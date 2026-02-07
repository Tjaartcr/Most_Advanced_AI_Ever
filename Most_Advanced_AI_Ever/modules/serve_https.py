
# serve_https.py

import ssl
import warnings
from backend import app, socketio

# 🛠️ Optional patch to suppress annoying SSLEOFError logs
def patch_ssl_eof_error():
    original_send = ssl.SSLSocket.send

    def silent_send(self, data, *args, **kwargs):
        try:
            return original_send(self, data, *args, **kwargs)
        except ssl.SSLEOFError:
            warnings.warn("⚠️ SSL EOF error suppressed (client disconnected prematurely).")
            return 0

    ssl.SSLSocket.send = silent_send

# Apply the patch before starting server
patch_ssl_eof_error()

if __name__ == '__main__':
    print("🔐 Starting HTTPS Socket.IO on port 5000 (for browser WebUI)…")
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        ssl_context=('./cert.pem', './key.pem')  # 👈 Use absolute paths if needed
    )


