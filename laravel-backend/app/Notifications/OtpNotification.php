<?php
namespace App\Notifications;
use Ichtrojan\Otp\Otp;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Notifications\Messages\MailMessage;
use Illuminate\Notifications\Notification;
class OtpNotification extends Notification implements ShouldQueue { use Queueable; public function via(object $notifiable):array{return ['mail'];} public function toMail(object $notifiable):MailMessage{$otp=(new Otp())->generate($notifiable->email,'numeric',6,20);return (new MailMessage)->greeting('Hello '.$notifiable->name)->subject('OTP Code')->line('Verify Your Email.')->line('Code : '.$otp->token)->line('The code is valid for 20 minutes and is used once.');} }
