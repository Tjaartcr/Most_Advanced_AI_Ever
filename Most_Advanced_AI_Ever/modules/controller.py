

# ------------------------------------------
# File: modules/controller.py
# Central logic layer, shared by GUI.py and webui.py
# ------------------------------------------
import threading
import datetime
import importlib
from modules.listen import listen
from modules.speech import speech
from modules.Repeat_Last import repeat
from modules.memory import memory
from config import RAG_API_URL, IMAGE_RAG_API_URL
import requests

state = {
    'chat_history': [],  # list of {'time', 'query', 'response'}
    'log': [],           # system log messages
    'models': {
        'thinkbot': 'deepseek-r1:1.5b',
        'chatbot': 'tinyllama',
        'vision': 'moondream',
        'coding': 'stablelm2'
    },
    'toggles': {
        'whisper': False,
        'vosk': True,
        'bluetooth': False
    }
}

def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def add_log(msg):
    ts = timestamp()
    entry = f"{ts}  {msg}"
    state['log'].append(entry)
    memory.add('log', entry)

def add_chat(query, response):
    ts = timestamp()
    state['chat_history'].append({'time': ts, 'query': query, 'response': response})
    memory.add('chat', {'time': ts, 'query': query, 'response': response})

# Toggle features

def toggle(name):
    state['toggles'][name] = not state['toggles'][name]
    val = state['toggles'][name]
    if name == 'whisper':
        listen.set_listen_whisper(val)
    elif name == 'vosk':
        listen.set_listen_vosk(val)
    elif name == 'bluetooth':
        listen.set_mobile_speech(val)
    add_log(f"Toggled {name} => {val}")

# Voice recording and processing
def record_voice():
    add_log('Recording voice input...')
    def _rec():
        text = listen()
        if text:
            process_query(text)
        else:
            add_log('No voice input detected.')
    threading.Thread(target=_rec, daemon=True).start()

# Process a text query: chat or RAG depending on prefix

def process_query(text):
    add_log(f"Processing query: {text}")
    # simple prefix detection
    if text.startswith('/rag_image'):
        # trigger image RAG
        file_path = text.split(' ',1)[1]
        files = {'file': open(file_path,'rb')}
        resp = requests.post(IMAGE_RAG_API_URL, files=files).json()
        answer = resp.get('answer','')
    elif text.startswith('/rag'):
        # text RAG
        prompt = text.split(' ',1)[1]
        resp = requests.post(RAG_API_URL, json={'query': prompt}).json()
        answer = resp.get('answer','')
    else:
        # normal chat response stub
        answer = f"Echo: {text}"
    add_chat(text, answer)
    speech.AlfredSpeak(answer)
    return answer

# Reload GUI for desktop (when code changes) -- optional hot reload
def reload_gui():
    import modules.GUI as GUImod
    importlib.reload(GUImod)
    add_log('GUI module reloaded')
