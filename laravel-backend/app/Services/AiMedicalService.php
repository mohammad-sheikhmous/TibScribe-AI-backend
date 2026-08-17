<?php

namespace App\Services;

use App\Models\ClinicalSession;
use App\Models\Patient;
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\Client\PendingRequest;
use Illuminate\Http\Client\Response;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\Http;

class AiMedicalService
{
    private function client(bool $retry = true): PendingRequest
    {
        $url = rtrim((string) config('services.ai.url'), ' /');
        $token = (string) config('services.ai.token');

        if ($url === '' || $token === '' || strlen($token) < 32) {
            throw new AiServiceException('AI service configuration is incomplete.', 503);
        }

        $client = Http::baseUrl($url)
            ->withToken($token)
            ->acceptJson()
            ->connectTimeout((int) config('services.ai.connect_timeout', 5))
            ->timeout((int) config('services.ai.timeout', 120));

        // Read-only/idempotent gateway calls can use Laravel's normal retry helper.
        // Mutating writes pass false; multipart uploads are retried manually so the stream is rewound.
        if ($retry) {
            $client = $client->retry(2, 250, throw: false);
        }

        return config('services.ai.verify_tls', true)
            ? $client
            : $client->withoutVerifying();
    }

    private function json(Response $response): array
    {
        if (!$response->successful()) {
            $body = $response->json();
            throw new AiServiceException(
                'AI service request failed.',
                $response->status(),
                is_array($body) ? $body : ['body' => $response->body()],
            );
        }

        return $response->json() ?? [];
    }

    private function headers(ClinicalSession $session): array
    {
        return [
            'X-TibScribe-Doctor-ID' => (string) $session->doctor->ai_external_id,
            'X-TibScribe-Patient-ID' => (string) $session->patient->ai_external_id,
            'X-TibScribe-Session-ID' => (string) $session->ai_external_id,
        ];
    }

    /** Strip internal FastAPI identifiers before data reaches the browser. */
    private function publicArtifact(ClinicalSession $session, array $payload): array
    {
        unset($payload['job_id'], $payload['patient_id'], $payload['external_session_id']);
        $payload['clinical_session_id'] = $session->id;
        if (isset($payload['patient_info']) && is_array($payload['patient_info'])) {
            unset($payload['patient_info']['patient_id']);
        }
        return $payload;
    }

    private function publicSuggestions(ClinicalSession $session, array $payload): array
    {
        foreach (($payload['suggestions'] ?? []) as $index => $suggestion) {
            if (is_array($suggestion)) {
                unset($suggestion['job_id'], $suggestion['patient_id']);
                $payload['suggestions'][$index] = $suggestion;
            }
        }
        unset($payload['job_id'], $payload['patient_id']);
        $payload['clinical_session_id'] = $session->id;
        return $payload;
    }

    public function createJob(ClinicalSession $session, UploadedFile $audio): array
    {
        $handle = fopen($audio->getRealPath(), 'rb');
        if (!is_resource($handle)) {
            throw new AiServiceException('Unable to open the uploaded audio.', 422);
        }

        try {
            $lastConnectionError = null;
            for ($attempt = 1; $attempt <= 2; $attempt++) {
                if ($attempt > 1) {
                    rewind($handle);
                    usleep(250_000);
                }

                try {
                    $response = $this->client(false)
                        ->withHeaders($this->headers($session))
                        ->attach(
                            'file',
                            $handle,
                            $audio->getClientOriginalName(),
                            ['Content-Type' => $audio->getMimeType() ?: 'application/octet-stream'],
                        )
                        ->post('/jobs', [
                            'visit_at' => $session->visit_at?->toIso8601String(),
                        ]);
                } catch (ConnectionException $e) {
                    $lastConnectionError = $e;
                    if ($attempt === 2) {
                        throw $e;
                    }
                    continue;
                }

                // A lost first response is safe to retry: FastAPI treats the Laravel
                // ClinicalSession id as its unique idempotency key and returns the
                // already-created job on the second request.
                if ($response->serverError() && $attempt < 2) {
                    continue;
                }

                return $this->json($response);
            }

            throw $lastConnectionError ?? new AiServiceException('AI upload failed.', 503);
        } finally {
            fclose($handle);
        }
    }

    public function getJob(ClinicalSession $session): array
    {
        return $this->json($this->client()->get('/jobs/' . $session->ai_job_id));
    }

    public function retryJob(ClinicalSession $session): array
    {
        if (!$session->ai_job_id) {
            throw new AiServiceException('AI job is not attached to this session.', 409);
        }

        return $this->json($this->client(false)->post('/jobs/' . $session->ai_job_id . '/retry'));
    }

    public function report(ClinicalSession $session): array
    {
        return $this->publicArtifact($session, $this->json($this->client()->get('/jobs/' . $session->ai_job_id . '/report')));
    }

    public function transcript(ClinicalSession $session, bool $raw = false): array
    {
        return $this->publicArtifact($session, $this->json($this->client()->get(
            '/jobs/' . $session->ai_job_id . '/transcript',
            ['include_raw' => $raw ? 'true' : 'false'],
        )));
    }

    public function suggestions(ClinicalSession $session): array
    {
        return $this->publicSuggestions($session, $this->json($this->client()->get('/jobs/' . $session->ai_job_id . '/suggestions')));
    }

    public function corrections(ClinicalSession $session): array
    {
        $rows = $this->json($this->client()->get('/jobs/' . $session->ai_job_id . '/corrections'));
        foreach ($rows as $index => $row) {
            if (is_array($row) && isset($row['actor']) && str_starts_with((string) $row['actor'], 'doctor:')) {
                $row['actor'] = 'doctor';
                $rows[$index] = $row;
            }
        }
        return $rows;
    }

    public function reviewQueue(ClinicalSession $session): array
    {
        $rows = $this->json($this->client()->get('/jobs/' . $session->ai_job_id . '/review-queue'));
        foreach ($rows as $index => $row) {
            if (is_array($row)) {
                unset($row['job_id']);
                $row['clinical_session_id'] = $session->id;
                $rows[$index] = $row;
            }
        }
        return $rows;
    }

    public function correctItem(ClinicalSession $session, string $item, array $payload): array
    {
        return $this->json($this->client(false)->patch(
            '/jobs/' . $session->ai_job_id . '/items/' . rawurlencode($item),
            $payload,
        ));
    }

    public function feedback(string $suggestionId, array $payload): array
    {
        return $this->json($this->client(false)->post(
            '/suggestions/' . rawurlencode($suggestionId) . '/feedback',
            $payload,
        ));
    }

    // Ensure the AI service has an opaque mapping for a Laravel patient.
    public function externalPatient(Patient $patient): array
    {
        return $this->json($this->client()->put(
            '/patients/by-external/laravel/' . rawurlencode((string) $patient->ai_external_id),
        ));
    }

    private function aiPatientId(Patient $patient): string
    {
        $mapped = $this->externalPatient($patient);
        $id = $mapped['patient']['id'] ?? null;
        if (!is_string($id) || $id === '') {
            throw new AiServiceException('AI patient mapping is invalid.', 502);
        }
        return $id;
    }

    public function patientTimeline(Patient $patient): array
    {
        $payload = $this->json($this->client()->get(
            '/patients/' . rawurlencode($this->aiPatientId($patient)) . '/timeline',
        ));
        $sessionMap = $patient->clinicalSessions()->pluck('id', 'ai_external_id');
        foreach (($payload['timeline'] ?? []) as $index => $row) {
            if (!is_array($row)) {
                continue;
            }
            $external = $row['external_session_id'] ?? null;
            $row['clinical_session_id'] = $external ? $sessionMap->get($external) : null;
            unset($row['job_id'], $row['external_session_id'], $row['visit_id']);
            $payload['timeline'][$index] = $row;
        }
        foreach (($payload['state_history'] ?? []) as $index => $row) {
            if (is_array($row)) {
                unset($row['job_id']);
                $payload['state_history'][$index] = $row;
            }
        }
        unset($payload['patient_id']);
        $payload['patient_id'] = $patient->id;
        return $payload;
    }

    public function setPatientState(Patient $patient, array $payload, ?ClinicalSession $session = null): array
    {
        if ($session?->ai_job_id) {
            $payload['refresh_job_id'] = $session->ai_job_id;
        }
        return $this->json($this->client(false)->post(
            '/patients/' . rawurlencode($this->aiPatientId($patient)) . '/state',
            $payload,
        ));
    }


    // Return an HTTP response backed by a streaming PSR body. The controller streams
    // this onward instead of materialising a potentially 100 MB recording in PHP RAM.
    public function audio(ClinicalSession $session, ?string $range = null): Response
    {
        $request = $this->client()
            ->withOptions(['stream' => true])
            ->withHeaders($range ? ['Range' => $range] : []);
        $response = $request->get('/jobs/' . $session->ai_job_id . '/audio');
        if (!$response->successful()) {
            throw new AiServiceException('AI audio request failed.', $response->status());
        }
        return $response;
    }

    public function itemAudio(ClinicalSession $session, string $item): Response
    {
        $response = $this->client()
            ->withOptions(['stream' => true])
            ->get('/jobs/' . $session->ai_job_id . '/items/' . rawurlencode($item) . '/audio');
        if (!$response->successful()) {
            throw new AiServiceException('AI audio segment request failed.', $response->status());
        }
        return $response;
    }

    public function health(): array
    {
        return $this->json($this->client()->get('/ready'));
    }
}
