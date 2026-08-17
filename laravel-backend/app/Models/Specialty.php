<?php
namespace App\Models;
use Illuminate\Database\Eloquent\Model;
use Spatie\Translatable\HasTranslations;
class Specialty extends Model { use HasTranslations; protected $fillable=['name','slug']; public $translatable=['name']; public function doctors(){return $this->belongsToMany(Doctor::class);} }
