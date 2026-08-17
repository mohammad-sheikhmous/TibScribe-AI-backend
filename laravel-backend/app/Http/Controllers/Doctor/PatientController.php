<?php

namespace App\Http\Controllers\Doctor;

use App\Http\Controllers\Controller;
use App\Models\Patient;
use App\Models\ClinicalSession;
use App\Services\AiMedicalService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Validation\Rule;
use Illuminate\Validation\Rules\Unique;

class PatientController extends Controller
{
    private function owned(Request $request, int $id): Patient
    {
        return Patient::where('doctor_id', $request->user('doctor')->id)->findOrFail($id);
    }

    private function mrnRule(Request $request, ?int $ignorePatientId = null): Unique
    {
        return Rule::unique('patients', 'mrn')
            ->where(fn($query) => $query->where('doctor_id', $request->user('doctor')->id))
            ->ignore($ignorePatientId);
    }

    public function index(Request $request)
    {
        $query = Patient::where('doctor_id', $request->user('doctor')->id);
        if ($search = $request->query('q')) {
            $query->where(fn($q) => $q
                ->where('mrn', 'like', "%{$search}%")
                ->orWhere('first_name', 'like', "%{$search}%")
                ->orWhere('last_name', 'like', "%{$search}%"));
        }

        return $query->latest()->paginate(min((int) $request->query('per_page', 20), 100));
    }

    public function store(Request $request)
    {
        $data = $request->validate([
            'mrn' => ['nullable', 'string', 'max:64', $this->mrnRule($request)],
            'first_name' => 'required|string|max:75',
            'last_name' => 'required|string|max:75',
            'birth_date' => 'nullable|date|before:today',
            'phone' => 'nullable|string|max:30',
            'notes' => 'nullable|string|max:5000',
        ]);
        $data['doctor_id'] = $request->user('doctor')->id;
        $patient = Patient::create($data);

        return dataJson('patient', $patient, 'Patient created.', true, 201);
    }

    public function show(Request $request, int $patient)
    {
        $record = $this->owned($request, $patient);
        return dataJson('patient', $record->loadCount('clinicalSessions'), 'Patient record.');
    }

    public function update(Request $request, int $patient)
    {
        $record = $this->owned($request, $patient);
        $data = $request->validate([
            'mrn' => ['sometimes', 'nullable', 'string', 'max:64', $this->mrnRule($request, $record->id)],
            'first_name' => 'sometimes|string|max:75',
            'last_name' => 'sometimes|string|max:75',
            'birth_date' => 'sometimes|nullable|date|before:today',
            'phone' => 'sometimes|nullable|string|max:30',
            'notes' => 'sometimes|nullable|string|max:5000',
        ]);
        $record->update($data);

        return dataJson('patient', $record->fresh(), 'Patient updated.');
    }

    public function timeline(Request $request, int $patient, AiMedicalService $ai)
    {
        $record = $this->owned($request, $patient);
        return dataJson('timeline', $ai->patientTimeline($record), 'Patient AI timeline.');
    }

    public function state(Request $request, int $patient, AiMedicalService $ai)
    {
        $record = $this->owned($request, $patient);
        $data = $request->validate([
            'pregnant' => 'sometimes|nullable|boolean',
            'postpartum' => 'sometimes|boolean',
            'ga_weeks' => 'sometimes|nullable|integer|min:1|max:44',
            'effective_at' => 'sometimes|nullable|date',
            'session_id' => 'sometimes|nullable|integer',
        ]);
        $hasSignal = (
            (array_key_exists('pregnant', $data) && $data['pregnant'] !== null)
            || (($data['postpartum'] ?? false) === true)
            || (($data['ga_weeks'] ?? null) !== null)
        );
        if (! $hasSignal) {
            abort(422, 'Set pregnant, postpartum=true, or ga_weeks.');
        }
        if (($data['postpartum'] ?? false) && (($data['pregnant'] ?? null) === true || isset($data['ga_weeks']))) {
            abort(422, 'Postpartum cannot be combined with pregnant=true or gestational age.');
        }

        $clinicalSession = null;
        if (! empty($data['session_id'])) {
            $clinicalSession = ClinicalSession::where('doctor_id', $request->user('doctor')->id)
                ->where('patient_id', $record->id)
                ->whereNotNull('ai_job_id')
                ->findOrFail((int) $data['session_id']);
            if ($clinicalSession->finalizedReport()->exists()) {
                abort(409, 'A finalized clinical report cannot be silently re-reasoned.');
            }
            // A session-linked override is a correction of THAT visit's clinical context.
            // Do not let an arbitrary timestamp silently refresh a different historical state.
            $effective = $clinicalSession->visit_at ?? $clinicalSession->created_at ?? now();
            $data['effective_at'] = $effective->toIso8601String();
        }
        unset($data['session_id']);

        if ($clinicalSession) {
            $lock = Cache::lock('clinical-session-mutation:' . $clinicalSession->id, 120);
            if (! $lock->get()) {
                abort(409, 'Another report mutation is already in progress.');
            }
            try {
                $clinicalSession->refresh();
                if ($clinicalSession->finalizedReport()->exists()) {
                    abort(409, 'A finalized clinical report cannot be silently re-reasoned.');
                }
                $state = $ai->setPatientState($record, $data, $clinicalSession);
            } finally {
                $lock->release();
            }
        } else {
            $state = $ai->setPatientState($record, $data);
        }

        return dataJson(
            'obstetric_state',
            $state,
            $clinicalSession ? 'Patient state updated and KBS suggestions refreshed.' : 'Patient state updated.',
            true,
            201,
        );
    }
}
