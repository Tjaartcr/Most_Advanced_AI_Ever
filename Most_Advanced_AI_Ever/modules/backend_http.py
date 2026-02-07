

import whisper  # ✅ This line is required!
##from listenWEBUI import WEBUIListenModule  # Your custom module
from listenWEBUI import WEBUIListenModule

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from auth_manager import login as do_login, signup as do_signup, get_users, delete_user, is_admin
from query_logger import log_user_query, read_logs
import os
import tempfile

from backend import app, socketio  # import your Flask/SocketIO objects


from listenWEBUI import WEBUIListenModule

webui_listener = WEBUIListenModule()  # ✅ instance created

app = Flask(__name__, static_folder='./webui-src/dist', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

##
whisper_model = whisper.load_model("base.en")
##

state = {
    'logs': [],
    'queries': [],
    'settings': {
        'use_whisper': False,
        'use_vosk': True,
        'enter_submits': True,
    }
}

@socketio.on('connect')
def connect():
    emit('full_state', state)

@socketio.on('gui_event')
def handle_event(data):
    print("Received gui_event:", data)
    user = data.get("user", "anonymous")

    if data['type'] == 'log':
        state['logs'].append(data['payload'])
    elif data['type'] == 'query':
        state['queries'].append(data['payload'])
        log_user_query(user, data['payload'])
    elif data['type'] == 'setting':
        state['settings'].update(data['payload'])
    
    socketio.emit('state_update', data)

@app.route('/login', methods=["POST"])
def login_user():
    data = request.json
    success, msg = do_login(data["username"], data["password"])
    return jsonify({"success": success, "message": msg})

@app.route('/signup', methods=["POST"])
def signup_user():
    data = request.json
    success, msg = do_signup(data["username"], data["password"])
    return jsonify({"success": success, "message": msg})

@app.route('/users', methods=["GET"])
def list_users():
    return jsonify(get_users())

@app.route('/delete_user', methods=["POST"])
def remove_user():
    data = request.json
    success = delete_user(data["username"])
    return jsonify({"success": success})

@app.route('/user_logs/<username>', methods=["GET"])
def get_user_logs(username):
    logs = read_logs(username)
    return jsonify(logs)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


@app.post("/upload_audio")
def upload_audio():

    webm_path = os.path.join("./tmp/temp_audio.webm")
    wav_path = os.path.join("./tmp/temp_audio.wav")

    # Save uploaded webm file
    file = request.files["audio"]
    file.save(webm_path)

    # Convert to WAV
    ffmpeg_cmd = f'ffmpeg -y -i "{webm_path}" -ar 16000 -ac 1 -f wav "{wav_path}"'
    os.system(ffmpeg_cmd)

    # Transcribe
    result = whisper_model.transcribe(wav_path)
    text = result.get("text", "").strip()

    # Process with your listen module
    from listenWEBUI import WEBUIlisten
    cleaned = WEBUIlisten.listen_text(text)
    
    return jsonify({"transcript": cleaned})


##if __name__ == '__main__':
##    from werkzeug.serving import run_simple
##    context = ('cert.pem', 'key.pem')  # Add this
##    socketio.run(app, host='0.0.0.0', port=5000, ssl_context=context)
####    socketio.run(app, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))


if __name__ == '__main__':
    # Plain HTTP, localhost only
##    socketio.run(app, host='127.0.0.1', port=5000)
    pass
