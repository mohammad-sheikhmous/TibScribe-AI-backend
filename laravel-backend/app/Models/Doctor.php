<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Foundation\Auth\User as Authenticatable;
use Illuminate\Notifications\Notifiable;
use Laravel\Sanctum\HasApiTokens;
use Illuminate\Support\Str;

class Doctor extends Authenticatable
{
    use HasApiTokens, HasFactory, Notifiable;
    protected $fillable = ['first_name', 'last_name', 'email', 'phone', 'hospital_or_clinic', 'image', 'status', 'password', 'email_verified_at', 'language'];
    protected $hidden = ['password', 'remember_token', 'ai_external_id'];
    protected static function booted(): void
    {
        static::creating(function (Doctor $doctor) {
            $doctor->ai_external_id ??= (string) Str::uuid();
        });
    }
    protected function casts(): array
    {
        return ['email_verified_at' => 'datetime', 'password' => 'hashed', 'status' => 'boolean'];
    }
    public function getNameAttribute(): string
    {
        return trim($this->first_name . ' ' . $this->last_name);
    }
    public function specialties()
    {
        return $this->belongsToMany(Specialty::class);
    }
    public function patients()
    {
        return $this->hasMany(Patient::class);
    }
    public function clinicalSessions()
    {
        return $this->hasMany(ClinicalSession::class);
    }

    public function finalizedReports()
    {
        return $this->hasMany(FinalizedReport::class);
    }
}
