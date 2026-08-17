<?php
namespace App\Http\Controllers\Doctor\Auth;
use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Validation\Rules\Password;
class ResetPasswordController extends Controller {
    public function reset(Request $request){
        $request->validate(['password'=>['required','confirmed',Password::min(8)->letters()->numbers()->symbols()]]);
        $doctor=$request->user('doctor'); $token=$doctor->currentAccessToken();
        if(!$token || $token->name!=='password-reset-token' || !$token->can('password:reset')) return messageJson('Invalid reset token.',false,403);
        $doctor->update(['password'=>$request->password]); $doctor->tokens()->delete(); return messageJson('Your password changed successfully');
    }
}
