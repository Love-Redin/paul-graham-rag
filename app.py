from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    # Bind to the environment port, or use 10000 by default
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)