<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Laravel\Sanctum\PersonalAccessToken;
use Symfony\Component\HttpFoundation\Response;

class SetUserLang
{
    public function handle(Request $request, Closure $next): Response
    {
        $locale = null;
        $token = $request->bearerToken();
        if ($token) {
            $access = PersonalAccessToken::findToken($token);
            $locale = $access?->tokenable?->language;
        }
        if (!$locale) {
            $candidate = substr((string)$request->header('Accept-Language', ''), 0, 2);
            $locale = in_array($candidate, ['en', 'ar'], true) ? $candidate : config('app.locale');
        }
        app()->setLocale($locale);
        return $next($request);
    }
}
