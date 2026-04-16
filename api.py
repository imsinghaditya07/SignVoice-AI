import sys
import os

# Add backend to path so imports work
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.append(backend_path)

# Import the app from the backend api.py
# This assumes api.py has the 'app' flask instance
from api import app

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)
