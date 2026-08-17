<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Str;

class ClinicalSession extends Model
{
    protected $fillable = [
        'doctor_id',
        'patient_id',
        'ai_job_id',
        'client_request_id',
        'client_request_fingerprint',
        'status',
        'stage',
        'ai_error',
        'original_filename',
        'visit_at',
        'started_at',
        'completed_at',
    ];

    // All cross-service identifiers and client idempotency material are implementation details.
    protected $hidden = ['ai_job_id', 'ai_external_id', 'client_request_id', 'client_request_fingerprint'];

    protected static function booted(): void
    {
        static::creating(function (ClinicalSession $session) {
            $session->ai_external_id ??= (string) Str::uuid();
        });
    }

    protected function casts(): array
    {
        return [
            'visit_at' => 'datetime',
            'started_at' => 'datetime',
            'completed_at' => 'datetime',
        ];
    }

    public function doctor(): BelongsTo
    {
        return $this->belongsTo(Doctor::class);
    }

    public function patient(): BelongsTo
    {
        return $this->belongsTo(Patient::class);
    }

    public function finalizedReport()
    {
        return $this->hasOne(FinalizedReport::class);
    }
}
