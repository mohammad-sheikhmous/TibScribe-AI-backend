<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class SwitchLangController extends Controller
{
    public function __invoke(Request $request)
    {
        $lang = $request->validate(['language' => 'required|in:en,ar'])['language'];
        $actor = $request->user('doctor') ?? $request->user();
        if ($actor) {
            $actor->update(['language' => $lang]);
        }
        app()->setLocale($lang);
        return dataJson('language', $lang, 'Language changed.');
    }
}
