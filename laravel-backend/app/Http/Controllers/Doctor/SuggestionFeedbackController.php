<?php

namespace App\Http\Controllers\Doctor;

use App\Http\Controllers\Controller;
use App\Models\ClinicalSession;
use App\Services\AiMedicalService;
use Illuminate\Http\Request;

class SuggestionFeedbackController extends Controller
{
    public function store(Request $r, int $session, string $suggestion, AiMedicalService $ai)
    {
        $s = ClinicalSession::where('doctor_id', $r->user('doctor')->id)->whereNotNull('ai_job_id')->findOrFail($session);
        $all = $ai->suggestions($s);
        $rows = $all['suggestions'] ?? $all;
        $owned = collect(is_array($rows) ? $rows : [])->contains(fn($row) => is_array($row) && (string)($row['id'] ?? '') === $suggestion);
        if (!$owned) abort(404, 'Suggestion not found for this clinical session.');
        $d = $r->validate(['action' => 'required|in:accepted,rejected,deferred,acted', 'reason' => 'nullable|string|max:2000']);
        $d['actor'] = 'doctor:' . $r->user('doctor')->ai_external_id;
        $feedback = $ai->feedback($suggestion, $d);
        if (is_array($feedback) && array_key_exists('actor', $feedback)) $feedback['actor'] = 'doctor';
        return dataJson('feedback', $feedback, 'Suggestion feedback recorded.', true, 201);
    }
}
