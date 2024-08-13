#!/bin/sh
set -e

# Function to run the main application
run_app() {
    echo "Starting the application..."
    python app.py
    echo "Application started."
}

run_app