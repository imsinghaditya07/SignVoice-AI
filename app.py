import sys
import os

# Ensure the backend directory is in the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(BASE_DIR, 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

# Import the 'app' instance from the backend/api.py file
try:
    from api import app
except ImportError as e:
    print(f"Error importing app from backend: {e}")
    sys.exit(1)

if __name__ == "__main__":
    # Render provides the port via environment variable
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)
