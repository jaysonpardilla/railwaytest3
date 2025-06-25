
echo "🔁 Applying migrations..."
python manage.py migrate

echo "🟢 Starting server..."
gunicorn main.wsgi:application --bind 0.0.0.0:$PORT
