<?php
namespace App\Providers;
use Illuminate\Cache\RateLimiting\Limit; use Illuminate\Http\Request; use Illuminate\Support\Facades\RateLimiter; use Illuminate\Support\ServiceProvider;
class AppServiceProvider extends ServiceProvider { public function register():void{} public function boot():void{$this->rate('auth-apis',5);$this->rate('normal-apis',120);$this->rate('ai-uploads',10);} private function rate(string $name,int $attempts):void{RateLimiter::for($name,function(Request $r)use($attempts){$key=$r->user('doctor')?->id ?: $r->ip();return Limit::perMinute($attempts)->by((string)$key)->response(fn($req,$headers)=>dataJson('retry_after',$headers['Retry-After']??60,'Too many requests.',false,429));});} }
