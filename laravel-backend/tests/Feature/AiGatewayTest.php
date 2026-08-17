<?php

namespace Tests\Feature;

use App\Models\ClinicalSession;
use App\Models\Doctor;
use App\Models\FinalizedReport;
use App\Models\Patient;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Str;
use Laravel\Sanctum\Sanctum;
use Ichtrojan\Otp\Otp;
use Tests\TestCase;

class AiGatewayTest extends TestCase
{
    use RefreshDatabase;

    private function doctor(): Doctor
    {
        return Doctor::factory()->create(['email_verified_at' => now()]);
    }

    private function patient(Doctor $doctor): Patient
    {
        return Patient::create([
            'doctor_id' => $doctor->id,
            'first_name' => 'Test',
            'last_name' => 'Patient',
        ]);
    }

    private function authenticate(Doctor $doctor): void
    {
        Sanctum::actingAs($doctor, ['*'], 'doctor');
    }

    public function test_unauthenticated_doctor_cannot_create_session(): void
    {
        $this->postJson('/api/doctor/sessions', [])->assertUnauthorized();
    }

    public function test_password_reset_token_cannot_access_clinical_apis(): void
    {
        $doctor = $this->doctor();
        $plain = $doctor->createToken('password-reset-token', ['password:reset'])->plainTextToken;

        $this->withToken($plain)->getJson('/api/doctor/patients')->assertForbidden();
    }

    public function test_doctor_cannot_use_another_doctors_patient(): void
    {
        $doctorA = $this->doctor();
        $doctorB = $this->doctor();
        $patientB = $this->patient($doctorB);
        $this->authenticate($doctorA);

        $this->post('/api/doctor/sessions', [
            'patient_id' => $patientB->id,
            'audio' => UploadedFile::fake()->create('a.wav', 10, 'audio/wav'),
        ])->assertNotFound();
    }

    public function test_duplicate_mrn_is_rejected_only_within_same_doctor(): void
    {
        $doctorA = $this->doctor();
        $doctorB = $this->doctor();
        Patient::create([
            'doctor_id' => $doctorA->id,
            'mrn' => 'MRN-1',
            'first_name' => 'A',
            'last_name' => 'One',
        ]);
        Patient::create([
            'doctor_id' => $doctorB->id,
            'mrn' => 'MRN-1',
            'first_name' => 'B',
            'last_name' => 'One',
        ]);

        $this->authenticate($doctorA);
        $this->postJson('/api/doctor/patients', [
            'mrn' => 'MRN-1',
            'first_name' => 'A',
            'last_name' => 'Duplicate',
        ])->assertUnprocessable();
    }

    public function test_session_upload_forwards_service_identity_headers(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $this->authenticate($doctor);

        Http::fake(['ai.test/jobs' => Http::response(['job_id' => 'job123', 'status' => 'queued'], 202)]);

        $this->withHeader('Idempotency-Key', (string) Str::uuid())->post('/api/doctor/sessions', [
            'patient_id' => $patient->id,
            'audio' => UploadedFile::fake()->create('a.wav', 10, 'audio/wav'),
        ])->assertStatus(202);

        Http::assertSent(fn ($request) =>
            $request->hasHeader('Authorization', 'Bearer '.config('services.ai.token'))
            && $request->hasHeader('X-TibScribe-Doctor-ID', (string) $doctor->ai_external_id)
            && $request->hasHeader('X-TibScribe-Patient-ID', (string) $patient->ai_external_id)
            && $request->hasHeader('X-TibScribe-Session-ID', (string) ClinicalSession::first()->ai_external_id)
        );
    }

    public function test_correction_actor_is_always_the_authenticated_doctor(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $session = ClinicalSession::create([
            'doctor_id' => $doctor->id,
            'patient_id' => $patient->id,
            'ai_job_id' => 'job-correct',
            'status' => 'complete',
        ]);
        $this->authenticate($doctor);

        Http::fake([
            'ai.test/jobs/job-correct/items/S-001' => Http::response([], 200),
        ]);

        $this->patchJson('/api/doctor/sessions/'.$session->id.'/items/S-001', [
            'text' => 'نص مصحح',
            'actor' => 'attacker',
        ])->assertOk();

        Http::assertSent(fn ($request) =>
            $request->method() === 'PATCH'
            && data_get($request->data(), 'actor') === 'doctor:'.$doctor->ai_external_id
        );
    }

    public function test_finalization_stores_an_immutable_laravel_snapshot(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $session = ClinicalSession::create([
            'doctor_id' => $doctor->id,
            'patient_id' => $patient->id,
            'ai_job_id' => 'job-final',
            'status' => 'complete',
        ]);
        $this->authenticate($doctor);

        Http::fake([
            'ai.test/jobs/job-final/report' => Http::response(['job_id' => 'job-final', 'soap' => []]),
            'ai.test/jobs/job-final/suggestions' => Http::response(['total' => 0, 'suggestions' => []]),
        ]);

        $this->postJson('/api/doctor/sessions/'.$session->id.'/finalize')->assertOk();
        $this->assertDatabaseHas('finalized_reports', ['clinical_session_id' => $session->id]);
        $this->postJson('/api/doctor/sessions/'.$session->id.'/finalize')->assertStatus(409);
        $this->assertSame(1, FinalizedReport::where('clinical_session_id', $session->id)->count());
    }
    public function test_confirmation_token_is_a_full_doctor_token(): void
    {
        $doctor = Doctor::factory()->create(['email_verified_at' => null]);
        $otp = (new Otp())->generate($doctor->email, 'numeric', 6, 20);

        $response = $this->postJson('/api/doctor/confirmation/verify', [
            'email' => $doctor->email,
            'OTP' => $otp->token,
        ])->assertOk();

        $plainToken = $response->json('token');
        $this->withToken($plainToken)->getJson('/api/doctor/patients')->assertOk();
    }

    public function test_patient_and_session_use_stable_opaque_ai_ids(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $session = ClinicalSession::create([
            'doctor_id' => $doctor->id,
            'patient_id' => $patient->id,
            'status' => 'queued',
        ]);

        $this->assertNotEmpty($doctor->ai_external_id);
        $this->assertNotEmpty($patient->ai_external_id);
        $this->assertNotEmpty($session->ai_external_id);
        $this->assertNotSame((string) $patient->id, $patient->ai_external_id);
        $this->assertNotSame((string) $session->id, $session->ai_external_id);
    }

    public function test_finalized_report_cannot_be_silently_corrected_after_approval(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $session = ClinicalSession::create([
            'doctor_id' => $doctor->id,
            'patient_id' => $patient->id,
            'ai_job_id' => 'job-locked',
            'status' => 'complete',
        ]);
        FinalizedReport::create([
            'clinical_session_id' => $session->id,
            'doctor_id' => $doctor->id,
            'patient_id' => $patient->id,
            'report_json' => ['soap' => []],
            'suggestions_json' => ['suggestions' => []],
            'finalized_at' => now(),
        ]);
        $this->authenticate($doctor);

        Http::fake();
        $this->patchJson('/api/doctor/sessions/'.$session->id.'/items/S-001', [
            'text' => 'should not be accepted',
        ])->assertStatus(409);
        Http::assertNothingSent();
    }

    public function test_disabling_doctor_invalidates_an_existing_full_token(): void
    {
        $doctor = $this->doctor();
        $plain = $doctor->createToken('doctor-token', ['*'])->plainTextToken;
        $doctor->update(['status' => false]);

        $this->withToken($plain)->getJson('/api/doctor/patients')->assertForbidden();
    }


    public function test_confirmation_endpoint_cannot_be_reused_as_passwordless_login(): void
    {
        $doctor = $this->doctor();
        $otp = (new Otp())->generate($doctor->email, 'numeric', 6, 20);

        $this->postJson('/api/doctor/confirmation/verify', [
            'email' => $doctor->email,
            'OTP' => $otp->token,
        ])->assertStatus(409);
    }

    public function test_failed_session_with_known_ai_job_can_be_retried(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $session = ClinicalSession::create([
            'doctor_id' => $doctor->id,
            'patient_id' => $patient->id,
            'ai_job_id' => 'job-retry',
            'status' => 'failed',
        ]);
        $this->authenticate($doctor);

        Http::fake([
            'ai.test/jobs/job-retry/retry' => Http::response([
                'job_id' => 'job-retry', 'status' => 'queued', 'stage' => null, 'error' => null,
            ], 202),
        ]);

        $this->postJson('/api/doctor/sessions/'.$session->id.'/retry')
            ->assertStatus(202);
        $this->assertSame('queued', $session->fresh()->status);
    }

    public function test_session_create_is_idempotent_at_the_laravel_public_boundary(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $this->authenticate($doctor);
        $key = (string) Str::uuid();

        Http::fake(['ai.test/jobs' => Http::response(['job_id' => 'job-idempotent', 'status' => 'queued'], 202)]);

        foreach ([1, 2] as $attempt) {
            $this->withHeader('Idempotency-Key', $key)->post('/api/doctor/sessions', [
                'patient_id' => $patient->id,
                'audio' => UploadedFile::fake()->createWithContent('same.wav', 'same-audio-bytes'),
            ])->assertStatus(202);
        }

        $this->assertSame(1, ClinicalSession::where('doctor_id', $doctor->id)->count());
        Http::assertSentCount(1);
    }

    public function test_reusing_idempotency_key_for_different_audio_is_rejected(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $this->authenticate($doctor);
        $key = (string) Str::uuid();

        Http::fake(['ai.test/jobs' => Http::response(['job_id' => 'job-one', 'status' => 'queued'], 202)]);

        $this->withHeader('Idempotency-Key', $key)->post('/api/doctor/sessions', [
            'patient_id' => $patient->id,
            'audio' => UploadedFile::fake()->createWithContent('same.wav', 'first-audio'),
        ])->assertStatus(202);

        $this->withHeader('Idempotency-Key', $key)->post('/api/doctor/sessions', [
            'patient_id' => $patient->id,
            'audio' => UploadedFile::fake()->createWithContent('same.wav', 'different-audio'),
        ])->assertStatus(409);

        $this->assertSame(1, ClinicalSession::where('doctor_id', $doctor->id)->count());
        Http::assertSentCount(1);
    }

    public function test_patient_state_tied_to_session_requests_kbs_refresh(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $session = ClinicalSession::create([
            'doctor_id' => $doctor->id,
            'patient_id' => $patient->id,
            'ai_job_id' => 'job-state',
            'status' => 'complete',
            'visit_at' => now(),
        ]);
        $this->authenticate($doctor);

        Http::fake([
            'ai.test/patients/by-external/laravel/*' => Http::response(['patient' => ['id' => 'ai-patient']]),
            'ai.test/patients/ai-patient/state' => Http::response(['status' => 'postpartum'], 201),
        ]);

        $this->postJson('/api/doctor/patients/'.$patient->id.'/state', [
            'postpartum' => true,
            'session_id' => $session->id,
        ])->assertCreated();

        Http::assertSent(fn ($request) =>
            $request->method() === 'POST'
            && str_ends_with($request->url(), '/patients/ai-patient/state')
            && data_get($request->data(), 'refresh_job_id') === 'job-state'
        );
    }

    public function test_lost_response_retry_rejects_different_audio_for_same_session(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $originalAudioHash = hash('sha256', 'original-audio');
        $visitAt = now()->utc()->startOfSecond();
        $visitKey = $visitAt->format('Y-m-d\TH:i:s\Z');
        $session = ClinicalSession::create([
            'doctor_id' => $doctor->id,
            'patient_id' => $patient->id,
            'status' => 'failed',
            'visit_at' => $visitAt,
            'client_request_fingerprint' => hash('sha256', $patient->ai_external_id.'|'.$originalAudioHash.'|'.$visitKey),
        ]);
        $this->authenticate($doctor);

        Http::fake();
        $this->post('/api/doctor/sessions/'.$session->id.'/retry', [
            'audio' => UploadedFile::fake()->createWithContent('visit.wav', 'different-audio'),
        ])->assertStatus(409);

        Http::assertNothingSent();
    }

    public function test_session_create_requires_idempotency_key(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $this->authenticate($doctor);

        Http::fake();
        $this->post('/api/doctor/sessions', [
            'patient_id' => $patient->id,
            'audio' => UploadedFile::fake()->createWithContent('visit.wav', 'audio-bytes'),
        ])->assertUnprocessable();
        Http::assertNothingSent();
    }

    public function test_reusing_idempotency_key_for_different_visit_time_is_rejected(): void
    {
        $doctor = $this->doctor();
        $patient = $this->patient($doctor);
        $this->authenticate($doctor);
        $key = (string) Str::uuid();
        $audioBytes = 'same-audio';

        Http::fake(['ai.test/jobs' => Http::response(['job_id' => 'job-time', 'status' => 'queued'], 202)]);

        $this->withHeader('Idempotency-Key', $key)->post('/api/doctor/sessions', [
            'patient_id' => $patient->id,
            'visit_at' => now()->subHours(2)->startOfSecond()->toIso8601String(),
            'audio' => UploadedFile::fake()->createWithContent('same.wav', $audioBytes),
        ])->assertStatus(202);

        $this->withHeader('Idempotency-Key', $key)->post('/api/doctor/sessions', [
            'patient_id' => $patient->id,
            'visit_at' => now()->subHour()->startOfSecond()->toIso8601String(),
            'audio' => UploadedFile::fake()->createWithContent('same.wav', $audioBytes),
        ])->assertStatus(409);

        $this->assertSame(1, ClinicalSession::where('doctor_id', $doctor->id)->count());
        Http::assertSentCount(1);
    }

}
