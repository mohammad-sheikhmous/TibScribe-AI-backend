<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class FinalizedReport extends Model
{
    protected $fillable = ['clinical_session_id', 'doctor_id', 'patient_id', 'report_json', 'suggestions_json', 'finalized_at'];
    protected function casts(): array
    {
        return ['report_json' => 'array', 'suggestions_json' => 'array', 'finalized_at' => 'datetime'];
    }
}
