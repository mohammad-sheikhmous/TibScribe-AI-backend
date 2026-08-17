<?php
namespace App\Http\Controllers\Doctor\Auth;
use App\Http\Controllers\Controller;
use App\Models\Doctor;
use App\Notifications\OtpNotification;
use Ichtrojan\Otp\Otp;
use Illuminate\Http\Request;
class OtpController extends Controller {
    private function isConfirmation(Request $request): bool { return $request->is('api/doctor/confirmation/*'); }
    public function sendOTP(Request $request){
        $request->validate(['email'=>'required|email']);
        $doctor=Doctor::where('email',$request->email)->first();
        if(!$doctor)return messageJson('The email is invalid.!',false,422);
        if($this->isConfirmation($request) && $doctor->email_verified_at){
            return messageJson('Email is already confirmed. Please sign in with your password.',false,409);
        }
        $doctor->notify(new OtpNotification());
        return messageJson('The code sent you successfully.');
    }
    public function verify(Request $request){
        $request->validate(['email'=>'required|email','OTP'=>'required|string']);
        $doctor=Doctor::where('email',$request->email)->first();
        if(!$doctor)return messageJson('The email is invalid.!',false,422);
        $isConfirmation=$this->isConfirmation($request);
        // Confirmation is enrollment, not a permanent passwordless-login path.
        if($isConfirmation && $doctor->email_verified_at){
            return messageJson('Email is already confirmed. Please sign in with your password.',false,409);
        }
        $otp=(new Otp())->validate($doctor->email,$request->OTP);
        if(!$otp->status)return messageJson('Code is invalid..!',false,401);
        if($isConfirmation && !$doctor->status)return messageJson('Doctor account is disabled.',false,403);
        if($isConfirmation){
            $doctor->forceFill(['email_verified_at'=>now()])->save();
            $doctor->tokens()->where('name','doctor-token')->delete();
            $token=$doctor->createToken('doctor-token',['*'])->plainTextToken;
        }else{
            $doctor->tokens()->where('name','password-reset-token')->delete();
            $token=$doctor->createToken('password-reset-token',['password:reset'])->plainTextToken;
        }
        return dataJson('token',$token,$isConfirmation?'The Email Confirmed Successfully.':'The code verified successfully., You are ready to reset password');
    }
}
