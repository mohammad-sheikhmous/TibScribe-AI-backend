<?php
namespace App\Exceptions;
use Exception;
use Illuminate\Auth\Access\AuthorizationException;
use Illuminate\Auth\AuthenticationException;
use Illuminate\Database\Eloquent\ModelNotFoundException;
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\Request;
use Illuminate\Validation\ValidationException;
use Symfony\Component\HttpKernel\Exception\HttpExceptionInterface;
use Throwable;
class ApiHandler extends Exception {
    public function __invoke(Throwable $e, Request $request) {
        if (!($request->is('api/*') || $request->expectsJson())) return null;
        if ($e instanceof ValidationException) return dataJson('errors',$e->errors(),$e->getMessage(),false,422);
        if ($e instanceof AuthenticationException) return messageJson('Unauthenticated.',false,401);
        if ($e instanceof AuthorizationException) return messageJson($e->getMessage() ?: 'Forbidden.',false,403);
        if ($e instanceof ModelNotFoundException) return messageJson('Resource not found.',false,404);
        if ($e instanceof HttpExceptionInterface) return messageJson($e->getMessage() ?: 'Request failed.',false,$e->getStatusCode());
        if ($e instanceof \App\Services\AiServiceException) return messageJson($e->getMessage(),false,$e->statusCode());
        if ($e instanceof ConnectionException) return messageJson('AI service is unreachable.',false,503);
        return null;
    }
}
