<?php

namespace App\Http\Controllers;

use Illuminate\Support\Facades\Storage;

class ImageController extends Controller
{
    public function __invoke($name)
    {
        foreach (['doctors'] as $disk) {
            if (Storage::disk($disk)->exists($name)) {
                return response(Storage::disk($disk)->get($name), 200)->header('Content-Type', Storage::disk($disk)->mimeType($name));
            }
        }
        return messageJson('Image not found.!', false, 404);
    }
}
