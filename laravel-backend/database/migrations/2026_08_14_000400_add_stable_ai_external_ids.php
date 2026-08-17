<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Str;

return new class extends Migration {
    public function up(): void
    {
        foreach (['doctors', 'patients', 'clinical_sessions'] as $tableName) {
            if (! Schema::hasColumn($tableName, 'ai_external_id')) {
                Schema::table($tableName, function (Blueprint $table) use ($tableName) {
                    $table->uuid('ai_external_id')->nullable();
                    $table->unique('ai_external_id', $tableName.'_ai_external_id_unique');
                });
            }

            DB::table($tableName)
                ->whereNull('ai_external_id')
                ->orderBy('id')
                ->chunkById(100, function ($rows) use ($tableName) {
                    foreach ($rows as $row) {
                        DB::table($tableName)->where('id', $row->id)->update([
                            'ai_external_id' => (string) Str::uuid(),
                        ]);
                    }
                });
        }
    }

    public function down(): void
    {
        foreach (['clinical_sessions', 'patients', 'doctors'] as $tableName) {
            if (Schema::hasColumn($tableName, 'ai_external_id')) {
                Schema::table($tableName, function (Blueprint $table) use ($tableName) {
                    $table->dropUnique($tableName.'_ai_external_id_unique');
                    $table->dropColumn('ai_external_id');
                });
            }
        }
    }
};
