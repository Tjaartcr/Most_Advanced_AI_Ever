from flask import Flask, request, jsonify
from flask_cors import CORS
import threading, queue, datetime, os, sys

# 1) make sure modules/ is importable
BASE = os.path.dirname(__file__)
MODULES_PATH = os.path.join(BASE, "..", "modules")
if MODULES_PATH not in sys.path:
    sys.path.insert(0, MODULES_PATH)

from speech import speech
from listen import listen
from Repeat_Last import repeat
from home_automation import home_auto

app = Flask(__name__)
CORS(app)

# Queues for Server‑Sent Events (logs & queries)
log_queue   = queue.Queue()
query_queue = queue.Queue()

def push_log(msg):
    log_queue.put(f"{datetime.datetime.now():%H:%M:%S} {msg}")

def push_query(msg):
    query_queue.put(f"{datetime.datetime.now():%H:%M:%S} {msg}")

@app.route("/api/record-voice", methods=["POST"])
def record_voice():
    def worker():
        try:
            txt = listen()
            if txt:
                push_query("Voice: " + txt)
            else:
                push_log("No voice detected")
        except Exception as e:
            push_log(f"Voice error: {e}")
    threading.Thread(target=worker).start()
    return jsonify(status="recording"), 202

@app.route("/api/send-text", methods=["POST"])
def send_text():
    data = request.json or {}
    txt = data.get("text","").strip()
    if not txt:
        push_log("Empty text")
        return jsonify(error="empty"), 400
    push_query("Text: " + txt)
    listen.add_text(txt)
    return jsonify(status="ok"), 200

@app.route("/api/repeat-last", methods=["POST"])
def repeat_last():
    try:
        r = repeat()
        if r:
            push_query("Repeated: " + r)
            return jsonify(text=r), 200
        push_log("Nothing to repeat")
        return jsonify(error="none"), 404
    except Exception as e:
        push_log(f"Repeat error: {e}")
        return jsonify(error=str(e)), 500

# SSE endpoints (for live logs & queries)
@app.route("/stream/logs")
def stream_logs():
    def gen():
        while True:
            msg = log_queue.get()
            yield f"data: {msg}\n\n"
    return app.response_class(gen(), mimetype="text/event-stream")

@app.route("/stream/queries")
def stream_queries():
    def gen():
        while True:
            msg = query_queue.get()
            yield f"data: {msg}\n\n"
    return app.response_class(gen(), mimetype="text/event-stream")

def run_webui():
    """Called by main.py when run with --web."""
    app.run(host="0.0.0.0", port=5000, debug=True)
