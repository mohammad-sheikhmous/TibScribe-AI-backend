<?php

namespace Database\Seeders;

use App\Models\Doctor;
use App\Models\Specialty;
use App\Models\User;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;

class DoctorSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        DB::transaction(function () {

            $specialties = Specialty::all();

            if ($specialties->isEmpty()) {
                throw new \Exception(
                    'No specialties found. Run SpecialtySeeder first.'
                );
            }

            $firstNames = [
                'Ahmad',
                'Mohammad',
                'Omar',
                'Ali',
                'Khaled',
                'Yousef',
                'Hassan',
                'Hussein',
                'Mahmoud',
                'Samer',
                'Rami',
                'Fadi',
                'Tarek',
                'Nabil',
                'Wael',
                'Lina',
                'Sara',
                'Nour',
                'Maya',
                'Rana',
                'Hala',
                'Reem',
                'Dima',
                'Rasha',
                'Aya',
                'Mariam',
                'Salma',
                'Dana',
                'Lama',
                'Ruba',
            ];

            $lastNames = [
                'Al Ahmad',
                'Al Hassan',
                'Al Ali',
                'Al Khaled',
                'Al Omar',
                'Sheikh',
                'Hamoud',
                'Saleh',
                'Yassin',
                'Nasser',
                'Ibrahim',
                'Darwish',
                'Hamad',
                'Mahmoud',
                'Khalil',
                'Saeed',
                'Mansour',
                'Abboud',
                'Rahman',
                'Ismail',
                'Haddad',
                'Farhat',
                'Kassem',
                'Bakri',
                'Sultan',
                'Najjar',
                'Hamdan',
                'Shami',
                'Masri',
                'Halabi',
            ];

            // نحسب الـ hash مرة واحدة فقط
            $password = Hash::make('Password1!');

            for ($i = 0; $i < 30; $i++) {

                $number = $i + 1;

                $doctor = Doctor::updateOrCreate(
                    [
                        'email' => sprintf(
                            'doctor%02d@example.com',
                            $number
                        ),
                    ],
                    [
                        'first_name' => $firstNames[$i],
                        'last_name' => $lastNames[$i],

                        'phone' => sprintf(
                            '093%07d',
                            $number
                        ),

                        'password' => $password,

                        'image' => 'default.png',

                        'status' => true,

                        'email_verified_at' => now(),

                        'language' => 'en',
                    ]
                );

                // ربط الطبيب بـ 1 أو 2 تخصص
                $count = random_int(
                    1,
                    min(2, $specialties->count())
                );

                $specialtyIds = $specialties
                    ->shuffle()
                    ->take($count)
                    ->pluck('id')
                    ->toArray();

                $doctor->specialties()->sync($specialtyIds);
            }
        });
    }
}
