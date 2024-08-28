from flask import Flask, render_template
import os

app = Flask(__name__)

# Route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Bind to the environment port, or use 10000 by default
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)