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
        if (Docotr::count() == 0) {
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
}
