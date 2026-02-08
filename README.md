1. ALFRED - Most Advanced AI Ever.:
    Alfred — Local assistant for automation, telepresence, and workshop integration

2. Short description:
    What Alfred is in one sentence: a local assistant that exposes a WebUI and desktop GUI, routes queries to models, logs sessions, supports audio transcription, translations, and integrates with embedded devices (ESP8266 / Mega).

3. Table of Contents:
    Short description
    Features
    Architecture / File map
    Requirements
    Installation
    Configuration
    Running Alfred (quick start)
    Usage examples (GUI, WebUI, socket events)
    Development notes (how to edit and test)
    Troubleshooting & common issues
    Contributing
    License & contact

4. Key features (bulleted)
    Socket.IO-backed backend for real-time events and multi-session support
    Desktop GUI (tkinter) for local control and session handling
    React WebUI (App.jsx) for remote/modern UI (model selection, history)
    Query logging per-user with recovery & per-user JSON logs (query_logger module)
    Audio upload & Whisper transcription pipeline (with Afrikaans↔English translation modules integrated)
    Reminders & scheduler support (reminders_module.py)
    Embedded device integration endpoints (ESP8266 + Arduino Mega) and macro playback/recording support
    Easy dev mode and production recommendations

5. Architecture / file map
    backend.py — socket server, session tracking (session_users[sid]), Whisper integration, log & emit responses, routes for uploads and       device endpoints.
    GUI.py — tkinter desktop GUI, stores current_user, emits gui_event, manages last_query.
    App.jsx — React WebUI: model selection, query history, and WebSocket/Socket.IO client.
    query_logger.py — append-safe JSON per-user logs with partial recovery for corrupt files.
    reminders_module.py — reminders scheduler and cleanup for run entries.
    en_to_af.py / af_to_en.py — translation wrappers used by transcription pipeline.
    esp8266_file_full — ESP8266 Web UI + Recorder sketch (LittleFS, MJPEG, macro endpoints).
    mega_file_full — Arduino Mega controller sketch (motor/relay control, PICKUP_MODE, object scanning).
    static/, templates/ — web UI static assets and Jinja templates (if present).
    requirements.txt — Python dependencies (Socket.IO server, whisper/whisperx or OpenAI bindings if used, etc.)

6. Requirements
    Python 3.11 (or as used by your environment)
    Node/npm for building React WebUI (if using App.jsx directly)
    System packages for Whisper/ASR (if used) and dependencies listed in requirements.txt
    Arduino toolchain (if building firmware)
    Recommended: run inside virtualenv / venv or docker-compose for production

7. Installation (quick)
    Clone the repo
    Create venv & install Python deps: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
    Build WebUI (if applicable): cd webui && npm install && npm run build
    Configure environment variables (see Configuration)

8. Configuration
    .env or environment variables:
    ALFRED_PORT — backend port (default 5000)
    SOCKET_IO_PATH — path for socket server (if changed)
    WHISPER_MODEL — base.en etc.
    LOG_DIR — path to user_logs/
    ESP8266_HOST / MEGA_SERIAL_PORT — device connection config
    Where files are stored: user_logs/, littlefs/ on ESP, etc.

9. Running Alfred (Quick start)
    Start backend: python backend.py
    Start GUI (local): python GUI.py
    Start WebUI dev: cd web && npm start or serve built assets with your webserver
    Verify logs appear in user_logs/<username>.json and socket events (gui_event, etc.) are exchanged.

10. Usage & Socket Events (examples)
    gui_event — GUI emits user queries + models; backend responds and logs.
    login / logout — session join/leave events mapping session_users[sid].
    audio_upload — route for uploading audio; backend transcribes, may call translation modules and emits resp.
    Device endpoints (HTTP): /macro/*, /record/*, /pickup/*, /telemetry for ESP8266 integration.
    Example flow: GUI/WebUI sends gui_event -> backend.py processes and logs -> emits resp back to client SID.


11. Development notes
    Tests: add unit tests for query_logger, translation wrappers, and backend event handlers.
    Local debugging: enable verbose logging in backend.py and run GUI locally.
    When changing firmware, keep esp8266_file_full and mega_file_full in firmware/ with version tags.

12. Troubleshooting
    Multiple users showing as same username: ensure session_users[sid] usage and avoid global new_user.
    Corrupt logs: query_logger has partial recovery — use provided recover() function (document where it lives).
    Whisper/ASR fails offline: ensure model files present and path configured; set fallback to af_to_en wrapper as needed.

13. Contributing
    Branching model: feature branches, PR to main, include unit tests for changes.
    Coding standards: keep small, focused changes; prefer small diffs.

14. License & contact

Suggested license (MIT or your choice)

Contact: Tjarrie @ tjaartcronje@gmail.com(AYEN @ ayen.ai.ml.web.soft@gmail.com) — where to report issues / request features.
