<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        Schema::table('clinical_sessions', function (Blueprint $table) {
            $table->uuid('client_request_id')->nullable()->after('ai_external_id');
            $table->char('client_request_fingerprint', 64)->nullable()->after('client_request_id');
            $table->unique(['doctor_id', 'client_request_id'], 'clinical_sessions_doctor_request_unique');
            $table->index(['doctor_id', 'patient_id', 'visit_at'], 'clinical_sessions_doctor_patient_visit_idx');
        });
    }

    public function down(): void
    {
        Schema::table('clinical_sessions', function (Blueprint $table) {
            $table->dropUnique('clinical_sessions_doctor_request_unique');
            $table->dropIndex('clinical_sessions_doctor_patient_visit_idx');
            $table->dropColumn(['client_request_id', 'client_request_fingerprint']);
        });
    }
};
