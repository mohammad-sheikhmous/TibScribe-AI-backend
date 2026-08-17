<?php

namespace App\Http\Controllers\Doctor;

use App\Http\Controllers\Controller;
use App\Models\ClinicalSession;
use App\Models\FinalizedReport;
use App\Services\AiMedicalService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Validation\Rule;

class ClinicalArtifactController extends Controller
{
    private const AI_LABELS = [
        'symptom',
        'diagnosis',
        'medication',
        'history',
        'allergy',
        'lab',
        'vital',
        'plan',
        'procedure',
        'info',
        'treatment',
        'follow_up',
        'nutrition',
        'pregnancy_risk',
        'cardiology',
        'neurology',
        'gynecology',
        'infection',
        'postpartum',
        'emergency',
    ];

    private function owned(Request $request, int $id): ClinicalSession
    {
        $session = ClinicalSession::where('doctor_id', $request->user('doctor')->id)
            ->findOrFail($id);
        if (! $session->ai_job_id) {
            abort(409, 'AI job is not available for this session.');
        }

        return $session;
    }

    public function report(Request $request, int $session, AiMedicalService $ai)
    {
        return dataJson('report', $ai->report($this->owned($request, $session)), 'SOAP report.');
    }

    public function transcript(Request $request, int $session, AiMedicalService $ai)
    {
        return dataJson(
            'transcript',
            $ai->transcript($this->owned($request, $session), $request->boolean('include_raw')),
            'Transcript.',
        );
    }

    public function suggestions(Request $request, int $session, AiMedicalService $ai)
    {
        return dataJson(
            'suggestions',
            $ai->suggestions($this->owned($request, $session)),
            'KBS suggestions.',
        );
    }

    public function corrections(Request $request, int $session, AiMedicalService $ai)
    {
        return dataJson(
            'corrections',
            $ai->corrections($this->owned($request, $session)),
            'Corrections.',
        );
    }

    public function reviewQueue(Request $request, int $session, AiMedicalService $ai)
    {
        return dataJson(
            'review_queue',
            $ai->reviewQueue($this->owned($request, $session)),
            'Review queue.',
        );
    }

    public function correctItem(
        Request $request,
        int $session,
        string $item,
        AiMedicalService $ai,
    ) {
        $clinicalSession = $this->owned($request, $session);
        $lock = Cache::lock('clinical-session-mutation:' . $clinicalSession->id, 120);
        if (!$lock->get()) {
            abort(409, 'Another report mutation is already in progress.');
        }
        try {
            $clinicalSession->refresh();
            if ($clinicalSession->finalizedReport()->exists()) {
                abort(409, 'Finalized reports are immutable. Create an explicit amendment instead of editing the AI draft.');
            }

            $data = $request->validate([
                'text' => 'sometimes|string',
                'label' => ['sometimes', 'string', Rule::in(self::AI_LABELS)],
                'soap_section' => 'sometimes|in:subjective,objective,assessment,plan',
                'speaker' => 'sometimes|nullable|in:doctor,patient,unknown',
            ]);
            $data['actor'] = 'doctor:' . $request->user('doctor')->ai_external_id;

            return dataJson(
                'item',
                $ai->correctItem($clinicalSession, $item, $data),
                'Report item corrected and KBS suggestions refreshed.',
            );
        } finally {
            $lock->release();
        }
    }

    public function finalize(Request $request, int $session, AiMedicalService $ai)
    {
        $clinicalSession = $this->owned($request, $session);
        $lock = Cache::lock('clinical-session-mutation:' . $clinicalSession->id, 120);
        if (! $lock->get()) {
            abort(409, 'Another report mutation is already in progress.');
        }
        try {
            $clinicalSession->refresh();
            if ($clinicalSession->status !== 'complete') {
                $job = $ai->getJob($clinicalSession);
                $status = $job['status'] ?? $clinicalSession->status;
                $clinicalSession->update([
                    'status' => $status,
                    'stage' => $job['stage'] ?? null,
                    'ai_error' => $job['error'] ?? null,
                    'completed_at' => $status === 'complete' ? now() : $clinicalSession->completed_at,
                ]);
                $clinicalSession->refresh();
            }

            if ($clinicalSession->status !== 'complete') {
                abort(409, 'Session is not complete.');
            }
            if ($clinicalSession->finalizedReport()->exists()) {
                abort(409, 'Report has already been finalized.');
            }

            $report = $ai->report($clinicalSession);
            $suggestions = $ai->suggestions($clinicalSession);
            $final = FinalizedReport::create([
                'clinical_session_id' => $clinicalSession->id,
                'doctor_id' => $clinicalSession->doctor_id,
                'patient_id' => $clinicalSession->patient_id,
                'report_json' => $report,
                'suggestions_json' => $suggestions,
                'finalized_at' => now(),
            ]);

            return dataJson('finalized_report', $final, 'Report finalized.');
        } finally {
            $lock->release();
        }
    }

    public function finalized(Request $request, int $session)
    {
        $clinicalSession = $this->owned($request, $session);
        $final = $clinicalSession->finalizedReport;
        if (!$final) {
            abort(404, 'No finalized report.');
        }
        return dataJson('finalized_report', $final, 'Finalized report.');
    }
}
