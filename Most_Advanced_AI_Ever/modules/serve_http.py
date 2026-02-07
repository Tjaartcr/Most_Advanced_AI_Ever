# serve_http.py
from backend import app, socketio

if __name__ == '__main__':
    print("▶️  Starting HTTP Socket.IO on port 5001 (for desktop GUI)…")
    socketio.run(app,
                 host='0.0.0.0',
                 port=5001)
