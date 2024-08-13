#!/bin/sh
set -e

# Function to run data preparation scripts
run_data_scripts() {
    echo "Running data preparation scripts..."
    python scripts/scraper.py
    echo "Scraping done. Running vector_db.py..."
    sleep 2
    python scripts/vector_db.py
    echo "Vector database complete. Running rag.py..."
    sleep 2
    python scripts/rag.py
    echo "Data preparation completed."
}

# Function to run the main application
run_app() {
    echo "Starting the application..."
    python app.py
    echo "Application started."
}

# Check the command passed to the container
case "$1" in
    run-data)
        run_data_scripts
        ;;
    run-app)
        run_app
        ;;
    run-all)
        run_data_scripts
        run_app
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: run-data | run-app | run-all"
        exit 1
        ;;
esac
