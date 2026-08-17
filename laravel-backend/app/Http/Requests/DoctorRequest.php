<?php
namespace App\Http\Requests;
use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Validation\Rule;
use Illuminate\Validation\Rules\Password;
class DoctorRequest extends FormRequest {
    public function authorize(): bool { return true; }
    public function rules(): array {
        $doctorId=$this->user('doctor')?->id; $register=$this->is('*/doctor/register');
        return [
          'first_name'=>[$register?'required':'sometimes','string','max:50'],
          'last_name'=>[$register?'required':'sometimes','string','max:50'],
          'phone'=>['nullable','string','max:20',Rule::unique('doctors','phone')->ignore($doctorId)],
          'email'=>[$register?'required':'sometimes','email','max:100',Rule::unique('doctors','email')->ignore($doctorId)],
          'image'=>['nullable','image','mimes:jpeg,gif,png,jpg,webp','max:5120'],
          'specialty_ids'=>[$register?'required':'sometimes','array','min:1'],
          'specialty_ids.*'=>['integer','exists:specialties,id'],
          'hospital_or_clinic'=>['nullable','string','max:150'],
          'password'=>[$register?'required':'sometimes','confirmed',Password::min(8)->letters()->numbers()->symbols()],
        ];
    }
}
