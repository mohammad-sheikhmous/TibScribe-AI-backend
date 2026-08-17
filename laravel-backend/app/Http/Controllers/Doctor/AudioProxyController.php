<?php

namespace App\Http\Controllers\Doctor;

use App\Http\Controllers\Controller;
use App\Models\ClinicalSession;
use App\Services\AiMedicalService;
use Illuminate\Http\Client\Response as AiResponse;
use Illuminate\Http\Request;

class AudioProxyController extends Controller
{
    private function owned(Request $request, int $id): ClinicalSession
    {
        return ClinicalSession::where('doctor_id', $request->user('doctor')->id)
            ->whereNotNull('ai_job_id')
            ->findOrFail($id);
    }

    private function proxy(AiResponse $aiResponse)
    {
        $psr = $aiResponse->toPsrResponse();
        $stream = $psr->getBody();
        $headers = [];
        foreach (['Content-Type', 'Content-Length', 'Content-Range', 'Accept-Ranges', 'Content-Disposition'] as $name) {
            if ($value = $aiResponse->header($name)) {
                $headers[$name] = $value;
            }
        }

        return response()->stream(function () use ($stream): void {
            while (!$stream->eof()) {
                echo $stream->read(64 * 1024);
                if (function_exists('ob_flush')) {
                    @ob_flush();
                }
                flush();
            }
        }, $aiResponse->status(), $headers);
    }

    public function full(Request $request, int $session, AiMedicalService $ai)
    {
        return $this->proxy(
            $ai->audio($this->owned($request, $session), $request->header('Range')),
        );
    }

    public function item(Request $request, int $session, string $item, AiMedicalService $ai)
    {
        return $this->proxy($ai->itemAudio($this->owned($request, $session), $item));
    }
}
