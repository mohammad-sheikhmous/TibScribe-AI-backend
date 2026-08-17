#!/usr/bin/env sh
set -eu

: "${APP_KEY:?APP_KEY must be configured on Render}"
: "${DATABASE_URL:?DATABASE_URL must point to the Laravel Neon database}"
: "${AI_SERVICE_URL:?AI_SERVICE_URL must point to the deployed Modal FastAPI URL}"
: "${AI_SERVICE_TOKEN:?AI_SERVICE_TOKEN must match Modal SERVICE_TOKEN}"

php artisan config:clear
php artisan migrate --force
php artisan db:seed --force
php artisan config:cache

exec php artisan serve --host=0.0.0.0 --port="${PORT:-10000}"
