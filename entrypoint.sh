#!/bin/sh
set -e

# Function to run the main application with Gunicorn
run_app() {
    echo "Starting the application with Gunicorn..."
    gunicorn --bind 0.0.0.0:${PORT:-10000} app:app
    echo "Application started."
}

# Check the command passed to the container
case "$1" in
    run-app)
        run_app
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: run-app"
        exit 1
        ;;
esac