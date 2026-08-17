<?php
namespace App\Http\Controllers\Doctor\Auth;
use App\Http\Controllers\Controller;
use App\Http\Requests\DoctorRequest;
use Illuminate\Http\Request;
class ProfileController extends Controller {
    public function show(Request $request){ return dataJson('profile_info',$request->user('doctor')->load('specialties'),'Profile Information'); }
    public function update(DoctorRequest $request){ $doctor=$request->user('doctor'); $data=$request->except('status','password','email','image','specialty_ids'); if($request->hasFile('image'))$data['image']=$doctor->image?updateImage($request->last_name??$doctor->last_name,$doctor->image,$request->file('image'),'doctors'):storeImage($request->last_name??$doctor->last_name,$request->file('image'),'doctors'); $doctor->update($data); if($request->has('specialty_ids'))$doctor->specialties()->sync($request->specialty_ids); return dataJson('profile_info',$doctor->fresh()->load('specialties'),'Profile information has been modified.'); }
    public function updateDoctorImage(Request $request){$request->validate(['image'=>'required|image|mimes:jpg,png,jpeg,gif,webp|max:5120']);$doctor=$request->user('doctor');$image=$doctor->image?updateImage($doctor->last_name,$doctor->image,$request->file('image'),'doctors'):storeImage($doctor->last_name,$request->file('image'),'doctors');$doctor->update(['image'=>$image]);return dataJson('image',$image,'Profile image updated successfully.');}
}
