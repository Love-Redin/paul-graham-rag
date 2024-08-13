from flask import Flask
import os

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Flask app is running!"

if __name__ == '__main__':
    # Render deployment
    # Get the port from the environment variable, or default to 10000
    port = int(os.environ.get('PORT', 10000))
    print(f"Attempting to start Flask app on port {port}...")
    app.run(debug=True, host='0.0.0.0', port=port)