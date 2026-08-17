<?php

namespace App\Http\Controllers\Doctor;

use App\Http\Controllers\Controller;
use App\Models\ClinicalSession;
use App\Models\Patient;
use App\Services\AiMedicalService;
use App\Services\AiServiceException;
use Illuminate\Database\QueryException;
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Validator;
use Throwable;

class ClinicalSessionController extends Controller
{
    private function owned(Request $request, int $id): ClinicalSession
    {
        return ClinicalSession::where('doctor_id', $request->user('doctor')->id)->findOrFail($id);
    }

    private function safeAiError(Throwable $e): string
    {
        if ($e instanceof ConnectionException) {
            return 'AI service is unreachable.';
        }
        if ($e instanceof AiServiceException) {
            return $e->getMessage();
        }
        return 'AI service request failed.';
    }

    private function requestFingerprint(Patient $patient, $audio, $visitAt): string
    {
        $audioHash = hash_file('sha256', $audio->getRealPath());
        if (! is_string($audioHash) || $audioHash === '') {
            abort(422, 'Unable to fingerprint the uploaded audio.');
        }
        // MySQL DATETIME does not preserve arbitrary timezone/microsecond forms. Canonicalise
        // to UTC seconds so the same clinical visit hashes identically before and after save.
        $visitKey = Carbon::parse($visitAt)->utc()->format('Y-m-d\TH:i:s\Z');
        return hash('sha256', $patient->ai_external_id . '|' . $audioHash . '|' . $visitKey);
    }

    private function idempotentResponse(ClinicalSession $session, string $fingerprint)
    {
        if ($session->client_request_fingerprint && ! hash_equals($session->client_request_fingerprint, $fingerprint)) {
            abort(409, 'This Idempotency-Key was already used for a different clinical-session request.');
        }
        $status = in_array($session->status, ['queued', 'running'], true) ? 202 : 200;
        return dataJson('session', $session, 'Existing clinical session returned for this Idempotency-Key.', true, $status);
    }

    public function index(Request $request)
    {
        $query = ClinicalSession::with('patient:id,first_name,last_name,mrn')
            ->where('doctor_id', $request->user('doctor')->id);
        if ($request->filled('patient_id')) {
            $query->where('patient_id', $request->integer('patient_id'));
        }
        if ($request->filled('status')) {
            $query->where('status', $request->string('status'));
        }
        return $query->orderByDesc('visit_at')->orderByDesc('id')
            ->paginate(min((int) $request->query('per_page', 20), 100));
    }

    public function store(Request $request, AiMedicalService $ai)
    {
        $data = $request->validate([
            'patient_id' => 'required|integer',
            'audio' => 'required|file|max:102400|mimes:wav,mp3,m4a,ogg,flac,webm,mp4,mpeg,mpga,oga,opus,aac,amr,wma',
            'visit_at' => 'nullable|date|before_or_equal:now',
        ]);
        $doctor = $request->user('doctor');
        $patient = Patient::where('doctor_id', $doctor->id)->findOrFail($data['patient_id']);

        // This public create is non-idempotent by nature (it creates a medical visit).
        // Require a browser-generated UUID so a lost HTTP response can be retried without
        // ever creating a second Laravel session or a second AI job.
        $clientRequestId = trim((string) $request->header('Idempotency-Key', ''));
        Validator::make(['idempotency_key' => $clientRequestId], [
            'idempotency_key' => ['required', 'uuid'],
        ])->validate();
        $effectiveVisitAt = isset($data['visit_at']) ? Carbon::parse($data['visit_at'])->utc() : now()->utc();
        $requestFingerprint = $this->requestFingerprint($patient, $request->file('audio'), $effectiveVisitAt);

        $existing = ClinicalSession::where('doctor_id', $doctor->id)
            ->where('client_request_id', $clientRequestId)
            ->first();
        if ($existing) {
            return $this->idempotentResponse($existing, $requestFingerprint);
        }

        try {
            $session = ClinicalSession::create([
                'doctor_id' => $doctor->id,
                'patient_id' => $patient->id,
                'client_request_id' => $clientRequestId,
                'client_request_fingerprint' => $requestFingerprint,
                'status' => 'queued',
                'original_filename' => $request->file('audio')->getClientOriginalName(),
                'visit_at' => $effectiveVisitAt,
                'started_at' => now(),
            ]);
        } catch (QueryException $e) {
            // Two identical browser retries can race between SELECT and INSERT. The DB unique
            // constraint is the final arbiter; only return an existing row if that is truly what
            // won the race, otherwise preserve the original database failure.
            if ($clientRequestId !== '') {
                $existing = ClinicalSession::where('doctor_id', $doctor->id)
                    ->where('client_request_id', $clientRequestId)
                    ->first();
                if ($existing) {
                    return $this->idempotentResponse($existing, $requestFingerprint);
                }
            }
            throw $e;
        }

        try {
            $job = $ai->createJob($session, $request->file('audio'));
            $session->update([
                'ai_job_id' => $job['job_id'],
                'status' => $job['status'] ?? 'queued',
            ]);
        } catch (Throwable $e) {
            $session->update([
                'status' => 'failed',
                'ai_error' => $this->safeAiError($e),
            ]);
            throw $e;
        }

        return dataJson('session', $session->fresh(), 'Clinical session queued.', true, 202);
    }

    public function retry(Request $request, int $session, AiMedicalService $ai)
    {
        $clinicalSession = $this->owned($request, $session);
        $lock = Cache::lock('clinical-session-mutation:' . $clinicalSession->id, 120);
        if (! $lock->get()) {
            abort(409, 'Another report mutation is already in progress.');
        }

        try {
            $clinicalSession->refresh();
            if ($clinicalSession->finalizedReport()->exists()) {
                abort(409, 'Finalized sessions cannot be retried.');
            }

            if ($clinicalSession->ai_job_id) {
                $job = $ai->retryJob($clinicalSession);
            } else {
                $request->validate([
                    'audio' => 'required|file|max:102400|mimes:wav,mp3,m4a,ogg,flac,webm,mp4,mpeg,mpga,oga,opus,aac,amr,wma',
                ]);
                if (! $clinicalSession->client_request_fingerprint) {
                    abort(409, 'This legacy session has no original audio fingerprint and cannot be safely recovered. Create a new clinical session.');
                }
                $retryFingerprint = $this->requestFingerprint($clinicalSession->patient, $request->file('audio'), $clinicalSession->visit_at);
                if (! hash_equals($clinicalSession->client_request_fingerprint, $retryFingerprint)) {
                    abort(409, 'Retry audio does not match the original clinical-session recording.');
                }
                $job = $ai->createJob($clinicalSession, $request->file('audio'));
                // If Laravel lost the original response, the idempotent upload can reveal an
                // already-failed AI job. Attach it first, then requeue that SAME durable job.
                if (($job['status'] ?? null) === 'failed' && ! empty($job['job_id'])) {
                    $clinicalSession->update(['ai_job_id' => $job['job_id']]);
                    $job = $ai->retryJob($clinicalSession->fresh());
                }
            }

            $clinicalSession->update([
                'ai_job_id' => $job['job_id'] ?? $clinicalSession->ai_job_id,
                'status' => $job['status'] ?? 'queued',
                'stage' => $job['stage'] ?? null,
                'ai_error' => null,
            ]);
            return dataJson('session', $clinicalSession->fresh(), 'Clinical session retry queued.', true, 202);
        } finally {
            $lock->release();
        }
    }

    public function show(Request $request, int $session, AiMedicalService $ai)
    {
        $clinicalSession = $this->owned($request, $session);
        if ($clinicalSession->ai_job_id && ! in_array($clinicalSession->status, ['complete', 'failed'], true)) {
            $job = $ai->getJob($clinicalSession);
            $status = $job['status'] ?? $clinicalSession->status;
            $clinicalSession->update([
                'status' => $status,
                'stage' => $job['stage'] ?? null,
                // FastAPI deliberately exposes only a generic processing error string.
                'ai_error' => $job['error'] ?? null,
                'completed_at' => $status === 'complete' ? now() : $clinicalSession->completed_at,
            ]);
        }
        return dataJson(
            'session',
            $clinicalSession->fresh()->load('patient:id,first_name,last_name,mrn'),
            'Clinical session.',
        );
    }
}
