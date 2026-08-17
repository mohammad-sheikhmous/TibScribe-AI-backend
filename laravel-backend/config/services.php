<?php
return [
  'ai' => [
    'url' => env('AI_SERVICE_URL', 'http://127.0.0.1:8001'),
    'token' => env('AI_SERVICE_TOKEN'),
    'timeout' => (int)env('AI_SERVICE_TIMEOUT', 120),
    'connect_timeout' => (int)env('AI_SERVICE_CONNECT_TIMEOUT', 5),
    'verify_tls' => filter_var(env('AI_SERVICE_VERIFY_TLS', true), FILTER_VALIDATE_BOOL),
  ],
];
