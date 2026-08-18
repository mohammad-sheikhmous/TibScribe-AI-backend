<?php

namespace App\Http\Controllers\Doctor;

use App\Http\Controllers\Controller;
use App\Models\ClinicalSession;
use App\Models\FinalizedReport;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;

class DashboardController extends Controller
{
    public function __invoke(Request $request)
    {
        $doctor = $request->user('doctor');

        /*
         * Render/Laravel يعمل حالياً بـ UTC.
         * React يستطيع إرسال timezone الطبيب:
         *
         * X-Timezone: Europe/Berlin
         * X-Timezone: Asia/Damascus
         */
        $timezone = $request->header('X-Timezone', 'UTC');

        if (! in_array($timezone, timezone_identifiers_list(), true)) {
            $timezone = 'UTC';
        }

        $now = Carbon::now($timezone);

        $todayStart = $now->copy()->startOfDay()->utc();
        $todayEnd   = $now->copy()->endOfDay()->utc();

        $weekStart = $now->copy()
            ->subDays(6)
            ->startOfDay()
            ->utc();

        ///////////////     Summary cards       ///////////////

        $todayQuery = ClinicalSession::query()
            ->where('doctor_id', $doctor->id)
            ->whereBetween('visit_at', [$todayStart, $todayEnd]);

        $todaySessions = (clone $todayQuery)->count();

        $inProgress = (clone $todayQuery)
            ->whereIn('status', ['queued', 'running'])
            ->count();

        $approvedToday = FinalizedReport::query()
            ->where('doctor_id', $doctor->id)
            ->whereBetween('finalized_at', [$todayStart, $todayEnd])
            ->count();

        //  حالياً نعتبر failed sessions تحذيرات تشغيلية.
        $warnings = (clone $todayQuery)
            ->where('status', 'failed')
            ->count();

        ////////////////     Recent sessions     ///////////////////

        $recentSessions = ClinicalSession::query()
            ->with([
                'patient:id,first_name,last_name,mrn',
                'finalizedReport:id,clinical_session_id,finalized_at',
            ])
            ->where('doctor_id', $doctor->id)
            ->orderByDesc('visit_at')
            ->orderByDesc('id')
            ->limit(5)
            ->get()
            ->map(function (ClinicalSession $session) use ($timezone) {

                $dashboardStatus = match (true) {
                    $session->finalizedReport !== null
                    => 'approved',

                    $session->status === 'complete'
                    => 'pending_review',

                    in_array($session->status, ['queued', 'running'], true)
                    => 'in_progress',

                    $session->status === 'failed'
                    => 'warning',

                    default
                    => $session->status,
                };

                return [
                    'id' => $session->id,

                    'patient' => [
                        'id' => $session->patient->id,
                        'name' => $session->patient->name,
                        'mrn' => $session->patient->mrn,
                    ],

                    'visit_at' => $session->visit_at
                        ?->copy()
                        ->timezone($timezone)
                        ->toIso8601String(),

                    'status' => $dashboardStatus,

                    'processing_status' => $session->status,

                    'stage' => $session->stage,

                    'has_error' => $session->status === 'failed',

                    'approved_at' => $session->finalizedReport?->finalized_at
                        ?->copy()
                        ->timezone($timezone)
                        ->toIso8601String(),
                ];
            });

        ////////////////     Weekly activity — last 7 days      ////////////////////

        $weekSessions = ClinicalSession::query()
            ->where('doctor_id', $doctor->id)
            ->whereBetween('visit_at', [$weekStart, $todayEnd])
            ->get(['id', 'visit_at']);

        $weekApproved = FinalizedReport::query()
            ->where('doctor_id', $doctor->id)
            ->whereBetween('finalized_at', [$weekStart, $todayEnd])
            ->get(['id', 'finalized_at']);

        $weeklyActivity = collect(range(6, 0))
            ->map(function ($daysAgo) use (
                $now,
                $timezone,
                $weekSessions,
                $weekApproved
            ) {
                $date = $now->copy()
                    ->subDays($daysAgo);

                $dateKey = $date->format('Y-m-d');

                $sessionsCount = $weekSessions
                    ->filter(
                        fn($session) =>
                        $session->visit_at
                            ->copy()
                            ->timezone($timezone)
                            ->format('Y-m-d') === $dateKey
                    )
                    ->count();

                $approvedCount = $weekApproved
                    ->filter(
                        fn($report) =>
                        $report->finalized_at
                            ->copy()
                            ->timezone($timezone)
                            ->format('Y-m-d') === $dateKey
                    )
                    ->count();

                return [
                    'date' => $dateKey,
                    'day' => $date->format('D'),
                    'sessions' => $sessionsCount,
                    'approved' => $approvedCount,
                ];
            })
            ->values();

        ///////////////////    Doctor brief    //////////////////

        $doctor->load('specialties:id,name');

        $dashboard = [
            'doctor' => [
                'id' => $doctor->id,
                'name' => $doctor->name,
                'image' => $doctor->image,
                'specialties' => $doctor->specialties,
            ],

            'summary' => [
                'today_sessions' => $todaySessions,
                'in_progress' => $inProgress,
                'approved_today' => $approvedToday,
                'warnings' => $warnings,
            ],

            'recent_sessions' => $recentSessions,

            'weekly_activity' => $weeklyActivity,

            'generated_at' => $now->toIso8601String(),
            'timezone' => $timezone,
        ];

        return dataJson('dashboard', $dashboard, 'Doctor dashboard.');
    }
}
