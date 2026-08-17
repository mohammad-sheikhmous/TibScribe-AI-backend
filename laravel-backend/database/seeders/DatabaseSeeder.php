<?php

namespace Database\Seeders;

use App\Models\Specialty;
use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    public function run(): void
    {
        foreach ([['slug' => 'obgyn', 'name' => ['en' => 'Obstetrics & Gynecology', 'ar' => 'النسائية والتوليد']], ['slug' => 'family-medicine', 'name' => ['en' => 'Family Medicine', 'ar' => 'طب الأسرة']]] as $s) Specialty::updateOrCreate(['slug' => $s['slug']], ['name' => $s['name']]);
    }
}
