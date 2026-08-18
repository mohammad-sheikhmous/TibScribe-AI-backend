<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Support\Str;

class Patient extends Model
{
    use SoftDeletes;
    protected $fillable = ['doctor_id', 'mrn', 'first_name', 'last_name', 'birth_date', 'phone', 'notes'];
    protected $hidden = ['ai_external_id'];
    protected static function booted(): void
    {
        static::creating(function (Patient $patient) {
            $patient->ai_external_id ??= (string) Str::uuid();
        });
    }
    protected function casts(): array
    {
        return ['birth_date' => 'date'];
    }
    public function doctor(): BelongsTo
    {
        return $this->belongsTo(Doctor::class);
    }
    public function clinicalSessions(): HasMany
    {
        return $this->hasMany(ClinicalSession::class);
    }
    public function getNameAttribute(): string
    {
        return trim(($this->first_name ?? '') . ' ' . ($this->last_name ?? ''));
    }
}
