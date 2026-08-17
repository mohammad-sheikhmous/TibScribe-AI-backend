<?php

namespace App\Http\Controllers\Doctor;

use App\Http\Controllers\Controller;
use App\Models\Specialty;

class SpecialtyController extends Controller
{
    public function index()
    {
        return Specialty::select(['id', 'name', 'slug'])->get();
    }
}
