#!/bin/bash

echo "🔁 Applying migrations..."
python manage.py migrate

echo "🟢 Starting server..."
gunicorn backend.wsgi:application --bind 0.0.0.0:$PORT
