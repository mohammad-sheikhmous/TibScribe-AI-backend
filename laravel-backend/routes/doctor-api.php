<?php

use App\Http\Controllers\Doctor\AiHealthController;
use App\Http\Controllers\Doctor\AudioProxyController;
use App\Http\Controllers\Doctor\Auth\LoginController;
use App\Http\Controllers\Doctor\Auth\OtpController;
use App\Http\Controllers\Doctor\Auth\ProfileController;
use App\Http\Controllers\Doctor\Auth\RegisterController;
use App\Http\Controllers\Doctor\Auth\ResetPasswordController;
use App\Http\Controllers\Doctor\ClinicalArtifactController;
use App\Http\Controllers\Doctor\ClinicalSessionController;
use App\Http\Controllers\Doctor\PatientController;
use App\Http\Controllers\Doctor\SpecialtyController;
use App\Http\Controllers\Doctor\SuggestionFeedbackController;
use App\Http\Controllers\ImageController;
use Illuminate\Support\Facades\Route;

Route::middleware('throttle:auth-apis')->group(function () {
    Route::post('login', [LoginController::class, 'login'])->middleware('guest:doctor');
    Route::post('register', [RegisterController::class, 'register'])->middleware('guest:doctor');
    Route::post('confirmation/email', [OtpController::class, 'sendOTP']);
    Route::post('confirmation/verify', [OtpController::class, 'verify']);
    Route::post('passwords/email', [OtpController::class, 'sendOTP']);
    Route::post('passwords/verify', [OtpController::class, 'verify']);
});

Route::get('images/{path}', ImageController::class)->where('path', '[A-Za-z0-9._-]+');
Route::get('specialties', [SpecialtyController::class, 'index'])->middleware('throttle:normal-apis');

// Both full doctor tokens and short-lived reset tokens can authenticate here, but the
// reset controller itself requires the password:reset token name/ability.
Route::middleware(['auth:doctor', 'throttle:normal-apis'])
    ->post('passwords/reset', [ResetPasswordController::class, 'reset']);

// Everything below contains clinical/account data and requires a normal doctor-token.
Route::middleware(['auth:doctor', 'doctor-token', 'throttle:normal-apis'])->group(function () {
    Route::post('logout', [LoginController::class, 'logout']);
    Route::get('profile', [ProfileController::class, 'show']);
    Route::patch('profile', [ProfileController::class, 'update']);
    Route::post('profile/image', [ProfileController::class, 'updateDoctorImage']);

    Route::get('ai/ready', AiHealthController::class);
    Route::get('patients', [PatientController::class, 'index']);
    Route::post('patients', [PatientController::class, 'store']);
    Route::get('patients/{patient}', [PatientController::class, 'show']);
    Route::patch('patients/{patient}', [PatientController::class, 'update']);
    Route::get('patients/{patient}/timeline', [PatientController::class, 'timeline']);
    Route::post('patients/{patient}/state', [PatientController::class, 'state']);

    Route::get('sessions', [ClinicalSessionController::class, 'index']);
    Route::post('sessions', [ClinicalSessionController::class, 'store'])->middleware('throttle:ai-uploads');
    Route::get('sessions/{session}', [ClinicalSessionController::class, 'show']);
    Route::post('sessions/{session}/retry', [ClinicalSessionController::class, 'retry'])->middleware('throttle:ai-uploads');
    Route::get('sessions/{session}/report', [ClinicalArtifactController::class, 'report']);
    Route::get('sessions/{session}/transcript', [ClinicalArtifactController::class, 'transcript']);
    Route::get('sessions/{session}/suggestions', [ClinicalArtifactController::class, 'suggestions']);
    Route::get('sessions/{session}/corrections', [ClinicalArtifactController::class, 'corrections']);
    Route::get('sessions/{session}/review-queue', [ClinicalArtifactController::class, 'reviewQueue']);
    Route::patch('sessions/{session}/items/{item}', [ClinicalArtifactController::class, 'correctItem']);
    Route::get('sessions/{session}/audio', [AudioProxyController::class, 'full']);
    Route::get('sessions/{session}/items/{item}/audio', [AudioProxyController::class, 'item']);
    Route::post('sessions/{session}/finalize', [ClinicalArtifactController::class, 'finalize']);
    Route::get('sessions/{session}/finalized-report', [ClinicalArtifactController::class, 'finalized']);
    Route::post('sessions/{session}/suggestions/{suggestion}/feedback', [SuggestionFeedbackController::class, 'store']);
});
