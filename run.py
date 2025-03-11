"""
Simple script to run the Flask application.
"""
from app import app

if __name__ == '__main__':
    print("Starting Code Analysis Tool...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000) 