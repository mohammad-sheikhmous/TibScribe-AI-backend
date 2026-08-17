FROM php:8.3-cli
RUN apt-get update && apt-get install -y --no-install-recommends \
        git unzip libzip-dev libicu-dev libonig-dev \
    && docker-php-ext-install pdo_mysql zip intl mbstring pcntl \
    && rm -rf /var/lib/apt/lists/*
# Match the application/FastAPI 100 MB audio contract. Without this, PHP can reject
# a real consultation recording before Laravel validation sees it.
RUN printf "upload_max_filesize=105M\npost_max_size=110M\nmemory_limit=256M\nmax_execution_time=180\n" \
    > /usr/local/etc/php/conf.d/tibscribe-uploads.ini
COPY --from=composer:2 /usr/bin/composer /usr/bin/composer
WORKDIR /app
COPY . .
# Production image intentionally excludes PHPUnit/Pail/etc.; CI installs dev dependencies.
RUN composer install --no-dev --no-interaction --prefer-dist --optimize-autoloader \
    && mkdir -p storage/framework/cache storage/framework/sessions storage/framework/views storage/logs \
    && chmod -R ug+rwX storage bootstrap
EXPOSE 8000
CMD ["php", "artisan", "serve", "--host=0.0.0.0", "--port=8000"]
