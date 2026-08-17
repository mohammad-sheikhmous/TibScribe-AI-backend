<?php

namespace App\Services;

use RuntimeException;

class AiServiceException extends RuntimeException
{
    public function __construct(string $message, private int $status = 502, private array $details = [])
    {
        parent::__construct($message);
    }
    public function statusCode(): int
    {
        return $this->status;
    }
    public function context(): array
    {
        return $this->details;
    }
}
