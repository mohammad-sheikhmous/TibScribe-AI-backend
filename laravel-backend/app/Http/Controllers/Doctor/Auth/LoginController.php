<?php
namespace App\Http\Controllers\Doctor\Auth;
use App\Http\Controllers\Controller;
use App\Models\Doctor;
use App\Notifications\OtpNotification;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
class LoginController extends Controller {
    public function login(Request $request){
        $fields=$request->validate(['email'=>'required|email','password'=>'required|string']);
        $doctor=Doctor::where('email',$fields['email'])->first();
        if(!$doctor || !Hash::check($fields['password'],$doctor->password)) return messageJson('البريد أو كلمة المرور غير صحيحة',false,401);
        if(!$doctor->status) return messageJson('Doctor account is disabled.',false,403);
        if(!$doctor->email_verified_at){ $doctor->notify(new OtpNotification()); return messageJson(__('auth.You have not confirmed your email yet'),false,401); }
        $doctor->tokens()->where('name','doctor-token')->delete();
        $token=$doctor->createToken('doctor-token')->plainTextToken;
        return dataJson('token',$token,'logged in');
    }
    public function logout(Request $request){ $request->user('doctor')->currentAccessToken()?->delete(); return messageJson('logged out successfully...'); }
}
