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
                ['slug' => 'general_practice'],
                [
                    'name' => [
                        'en' => 'General Practice',
                        'ar' => 'طب عام'
                    ],
                ]
            ],
            [
                ['slug' => 'internal_medicine',],
                [
                    'name' => [
                        'en' => 'Internal Medicine',
                        'ar' => 'باطنية'
                    ],
                ]
            ],
            [
                ['slug' => 'dermatology'],
                [
                    'name' => [
                        'en' => 'Dermatology',
                        'ar' => 'جلدية'
                    ]
                ]
            ],
            [
                ['slug' => 'pediatrics'],

                [
                    'name' => [
                        'en' => 'Pediatrics',
                        'ar' => 'أطفال'
                    ]
                ]
            ],
            [
                ['slug' => 'ophthalmology'],
                [
                    'name' => [
                        'en' => 'Ophthalmology',
                        'ar' => 'عيون'
                    ]
                ]
            ],
            [
                ['slug' => 'cardiology'],
                [
                    'name' => [
                        'en' => 'Cardiology',
                        'ar' => 'قلبية'
                    ],
                ]
            ],
            [
                ['slug' => 'obstetrics'],
                [
                    'name' => [
                        'en' => 'Obstetrics',
                        'ar' => 'طب التوليد'
                    ]
                ]
            ],
        ];

        foreach ($speciaties as $specialty) {
            Specialty::updateOrCreate($specialty);
        }
    }
}
