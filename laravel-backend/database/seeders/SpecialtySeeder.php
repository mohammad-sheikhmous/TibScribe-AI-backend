<?php

namespace Database\Seeders;

use App\Models\Specialty;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class SpecialtySeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $speciaties = [
            [
                'name' => [
                    'en' => 'General Practice',
                    'ar' => 'طب عام'
                ],
                'slug' => 'general_practice'
            ],
            [
                'name' => [
                    'en' => 'Internal Medicine',
                    'ar' => 'باطنية'
                ],
                'slug' => 'internal_medicine'
            ],
            [
                'name' => [
                    'en' => 'Dermatology',
                    'ar' => 'جلدية'
                ],
                'slug' => 'dermatology'
            ],
            [
                'name' => [
                    'en' => 'Pediatrics',
                    'ar' => 'أطفال'
                ],
                'slug' => 'pediatrics'
            ],
            [
                'name' => [
                    'en' => 'Ophthalmology',
                    'ar' => 'عيون'
                ],
                'slug' => 'ophthalmology'
            ],
            [
                'name' => [
                    'en' => 'Cardiology',
                    'ar' => 'قلبية'
                ],
                'slug' => 'cardiology'
            ],
            [
                'name' => [
                    'en' => 'Obstetrics',
                    'ar' => 'طب التوليد'
                ],
                'slug' => 'obstetrics'
            ],
        ];

        foreach ($speciaties as $specialty) {
            Specialty::updateOrCreate(
                [
                    'slug' => $specialty['slug'],
                ],
                [
                    'name' => $specialty['name'],
                ]
            );
        }
    }
}
