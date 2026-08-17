<?php

namespace App\Http\Controllers\Doctor\Auth;

use App\Http\Controllers\Controller;
use App\Http\Requests\DoctorRequest;
use App\Models\Doctor;
use App\Notifications\OtpNotification;
use Illuminate\Support\Facades\DB;
use Throwable;

class RegisterController extends Controller
{
    public function register(DoctorRequest $request)
    {
        $storedImage = null;
        try {
            if ($request->hasFile('image')) {
                $storedImage = storeImage($request->last_name, $request->file('image'), 'doctors');
            }

            $doctor = DB::transaction(function () use ($request, $storedImage) {
                $data = $request->except('status', 'image', 'specialty_ids');
                if ($storedImage) {
                    $data['image'] = $storedImage;
                }
                $doctor = Doctor::create($data);
                $doctor->specialties()->sync($request->specialty_ids);
                return $doctor;
            });
        } catch (Throwable $e) {
            // Filesystem writes are outside the SQL transaction. Remove a newly written image
            // if doctor/specialty persistence rolls back so registration never leaks orphans.
            if ($storedImage) {
                deleteImage($storedImage, 'doctors');
            }
            throw $e;
        }

        $doctor->notify(new OtpNotification());
        return dataJson(
            'doctor', $doctor,
            'the doctor created successfully, check verification code sent you to confirm the email',
            true, 201,
        );
    }
}
