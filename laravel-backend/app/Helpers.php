<?php

use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

if (!function_exists('messageJson')) {
    function messageJson(string|array $messageVal, bool $status = true, int $code = 200, string $messageKey = 'message')
    {
        return response()->json(['status' => $status, 'status_code' => $code, $messageKey => $messageVal], $code);
    }
}
if (!function_exists('dataJson')) {
    function dataJson(string $dataKey, mixed $data, string $message = '', bool $status = true, int $code = 200)
    {
        return response()->json(['status' => $status, 'status_code' => $code, 'message' => $message, $dataKey => $data], $code);
    }
}
if (!function_exists('storeImage')) {
    function storeImage(string $name, mixed $image, string $disk): string
    {
        $imageName = Str::slug($name) . '-' . Str::uuid() . '.' . strtolower($image->getClientOriginalExtension());
        $stored = Storage::disk($disk)->putFileAs('', $image, $imageName);
        if ($stored === false) {
            throw new RuntimeException('Failed to store uploaded image.');
        }
        return $imageName;
    }
}
if (!function_exists('deleteImage')) {
    function deleteImage(?string $name, string $disk): void
    {
        if ($name && Storage::disk($disk)->exists($name)) Storage::disk($disk)->delete($name);
    }
}
if (!function_exists('updateImage')) {
    function updateImage(string $newName, ?string $stored, mixed $image, string $disk): string
    {
        $replacement = storeImage($newName, $image, $disk);
        deleteImage($stored, $disk);
        return $replacement;
    }
}
