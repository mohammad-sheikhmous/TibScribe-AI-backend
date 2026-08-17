<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class EnsureDoctorToken
{
    public function handle(Request $request, Closure $next): Response
    {
        $doctor = $request->user('doctor');

        // Normal doctor tokens are created with Sanctum's wildcard ability. A
        // password-reset token carries only `password:reset`, so it authenticates for
        // the reset endpoint but cannot cross this clinical/account boundary.
        if (! $doctor || ! $doctor->tokenCan('*')) {
            abort(403, 'A full doctor access token is required.');
        }
        if (!$doctor->status) {
            // Account suspension must invalidate already-issued bearer tokens as well;
            // checking only at login leaves a disabled clinician authenticated.
            abort(403, 'Doctor account is disabled.');
        }

        return $next($request);
    }
}
