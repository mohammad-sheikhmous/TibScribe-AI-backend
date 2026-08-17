<?php

namespace App\Http\Controllers\Doctor;

use App\Http\Controllers\Controller;
use App\Services\AiMedicalService;

class AiHealthController extends Controller
{
    public function __invoke(AiMedicalService $ai)
    {
        return dataJson('ai', $ai->health(), 'AI service readiness.');
    }
}
