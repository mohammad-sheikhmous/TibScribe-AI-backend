<?php

namespace Database\Seeders;

use App\Models\Doctor;
use App\Models\Specialty;
use App\Models\User;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class DoctorSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        DB::transaction(function () {
            DB::statement('ALTER TABLE doctors DISABLE TRIGGER ALL;');

            DB::statement('TRUNCATE TABLE doctors RESTART IDENTITY;');
            DB::table('doctor_specialty')->truncate();

            DB::statement('ALTER TABLE doctors ENABLE TRIGGER ALL;');
        });

        $specialties = Specialty::all();

        Doctor::factory()
            ->count(30)
            // طريقة 1
            // ->hasAttached($specialties->random(rand(1, 2))) 
            ->create()
            // طريقة 2
            ->each(function ($doctor) use ($specialties) {
                $doctor->specialties()->attach(
                    $specialties->random(rand(1, 2))->pluck('id')->toArray()
                );
            });
    }
}
