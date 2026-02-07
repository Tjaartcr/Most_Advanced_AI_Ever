

import os
import re
import io
import json
import time
import base64
import mimetypes
import tempfile
from datetime import datetime

from flask import Flask, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename

# Optional libs (may not be installed in every env)
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None
    ChromaSettings = None

try:
    import ollama
except Exception:
    ollama = None

try:
    import whisper
except Exception:
    whisper = None

try:
    import chardet
except Exception:
    chardet = None

try:
    from pdfminer.high_level import extract_text
except Exception:
    extract_text = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import PyPDF2  # noqa: F401  (unused directly but kept for compatibility)
except Exception:
    PyPDF2 = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# Local modules
from listenWEBUI import WEBUIListenModule
from auth_manager import login as do_login, signup as do_signup, get_users, delete_user
from query_logger import log_user_query, log_user_response, read_logs

# -------------------- Paths & Globals --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, 'tmp')
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# Initialize listener and Whisper (if available)
webui_listener = WEBUIListenModule()
if whisper is not None:
    try:
        whisper_model = whisper.load_model("base.en")
    except Exception:
        whisper_model = None
else:
    whisper_model = None

app = Flask(__name__, static_folder='./webui-src/dist', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Module-level state
state = {
    'logs': [],
    'queries': [],
    'resp': [],
    'settings': {
        'use_whisper': False,
        'use_vosk': True,
        'enter_submits': True
    }
}

# Session tracking
session_users = {}  # Maps SID -> username

my_file_content = ""
collection = None   # Chroma collection shared across events
client = None       # Chroma client shared across events

model_selected = ""
thinkbot_model = ""
rag_clear = False

# Cached chroma client
_CHROMA_CLIENT = None


# -------------------- RAG / Embedding Helpers --------------------
def query_with_retrieval(prompt: str, n_results: int = 1):
    """Retrieve top chunks from Chroma and run generation with Ollama."""
    try:
        if ollama is None or chromadb is None:
            raise RuntimeError("Ollama or Chroma not available")

        q_emb = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)
        q_vector = q_emb.get("embedding") if isinstance(q_emb, dict) else q_emb
        client_local = chromadb.Client()
        coll = client_local.get_or_create_collection(name="docs")
        res = coll.query(query_embeddings=[q_vector], n_results=n_results)
        top_doc = res.get("documents", [[]])[0][0] if res.get("documents") else ""
    except Exception as e:
        print(f"[WARN] retrieval failed: {e}")
        top_doc = ""

    gen_prompt = f"Using this data: {top_doc}\n\nRespond to: {prompt}"
    try:
        if ollama is None:
            raise RuntimeError("Ollama not available")
        out = ollama.generate(model="tinyllama", prompt=gen_prompt, stream=False)
        text = out.get("response") or out.get("text") or str(out)
    except Exception as e:
        text = f"[generation failed: {e}]"
    return text


def extract_pdf_text_with_ocr(pdf_path):
    """Try text extraction first; if too short, fallback to OCR."""
    # Direct text extraction
    try:
        if extract_text is not None:
            text = extract_text(pdf_path)
            if text and len(text.strip()) > 10:
                return text
    except Exception as e:
        print(f"[ERROR] PDFminer extract_text failed: {e}")

    # OCR fallback
    try:
        print("[INFO] PDF text extraction empty or too short, using OCR fallback.")
        if convert_from_path is None or pytesseract is None:
            print("[WARN] OCR tools unavailable (pdf2image/pytesseract missing).")
            return ""
        pages = convert_from_path(pdf_path)
        ocr_text = ""
        for page in pages:
            ocr_text += pytesseract.image_to_string(page)
        return ocr_text
    except Exception as e:
        print(f"[ERROR] PDF extraction + OCR failed: {e}")
        return ""


def _save_tmp(filename, data_bytes):
    """Utility: safe save to tmp dir."""
    safe = secure_filename(filename)
    path = os.path.join(TMP_DIR, safe)
    with open(path, 'wb') as f:
        f.write(data_bytes)
    return path


# --- helper: singleton chroma client creator (safe) ---
def get_chroma_client(persist_directory=None, chroma_db_impl="duckdb+parquet"):
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is not None:
        return _CHROMA_CLIENT
    if chromadb is None:
        print("[RAG] chromadb not available")
        return None

    try:
        settings = ChromaSettings(chroma_db_impl=chroma_db_impl,
                                  persist_directory=persist_directory)
        _CHROMA_CLIENT = chromadb.Client(settings=settings)
        print(f"[RAG] Chroma client created with persist_directory={persist_directory}")
        return _CHROMA_CLIENT
    except Exception as e:
        print(f"[RAG] Embedded Chroma init failed: {e}; trying fallback chromadb.Client()")
        try:
            _CHROMA_CLIENT = chromadb.Client()
            print("[RAG] Chroma client created via fallback chromadb.Client()")
            return _CHROMA_CLIENT
        except Exception as e2:
            print(f"[RAG] chromadb.Client() fallback failed: {e2}")
            _CHROMA_CLIENT = None
            return None


# -------------------- Socket Handlers --------------------
@socketio.on('connect')
def connect():
    emit('full_state', state)


@socketio.on('disconnect')
def disconnect():
    sid = request.sid
    user = session_users.pop(sid, None)
    if user:
        leave_room(user)
        print(f"[DISCONNECT] {sid} left room {user}")


@socketio.on('gui_event')
def handle_event(data):
    global client, collection, thinkbot_model, model_selected, rag_clear, my_file_content
    sid = request.sid
    print(f"[DEBUG] Received gui_event from sid={sid}: {data!r}")

    user = data.get('username') if isinstance(data, dict) else None
    t = data.get('type') if isinstance(data, dict) else None
    payload = data.get('payload') if isinstance(data, dict) else None

    if not user:
        user = 'Tjaart'  # fallback

    # Track user per session
    session_users[sid] = user
    join_room(user)

    # --- LOGIN ---
    if t == 'login':
        print(f"[LOGIN] User: {user}")
        socketio.emit('state_update', {
            'type': 'login_ack',
            'username': user
        }, room=user)
        return

    # --- LOG ---
    if t == 'log':
        if isinstance(payload, str):
            log_user_response(user, payload)
            socketio.emit('state_update', {
                'type': 'log',
                'payload': payload,
                'username': user
            }, room=user)

        elif isinstance(payload, dict):
            entry = {
                "username": user,
                "timestamp": datetime.now().isoformat(),
                "query": payload.get("query", ""),
                "response": payload.get("response", ""),
                "models": payload.get("models", {})
            }
            log_user_query(user, entry)
            socketio.emit('state_update', {
                'type': 'log',
                'payload': entry,
                'username': user
            }, room=user)
            socketio.emit('state_update', {
                'type': 'query',
                'payload': entry,
                'username': user
            }, room=user)
        return

    # --- RESP ---
    if t == 'resp':
        if isinstance(payload, str):
            log_user_response(user, payload)
            socketio.emit('state_update', {
                'type': 'log',
                'payload': payload,
                'username': user
            }, room=user)

        elif isinstance(payload, dict):
            entry = {
                "username": user,
                "timestamp": datetime.now().isoformat(),
                "query": payload.get("query", ""),
                "response": payload.get("response", ""),
                "models": payload.get("models", {})
            }
            log_user_query(user, entry)
            socketio.emit('state_update', {
                'type': 'log',
                'payload': entry,
                'username': user
            }, room=user)
            socketio.emit('state_update', {
                'type': 'query',
                'payload': entry,
                'username': user
            }, room=user)
        return

    # --- SETTING (generic) and thinkbot model detection ---
    if t in ('setting', 'think_model'):
        data_payload = data.get('payload')
        who = data.get('username') or user

        selected = None
        if isinstance(data_payload, dict):
            if 'thinkbot_model' in data_payload:
                selected = data_payload['thinkbot_model']
            else:
                model_keys = [k for k in data_payload.keys() if k.endswith('_model')]
                if len(model_keys) == 1:
                    selected = data_payload[model_keys[0]]
        elif isinstance(data_payload, str):
            selected = data_payload

        if selected:
            thinkbot_model = selected
            print(f"[SETTING] thinkbot_model set to '{thinkbot_model}'")
            socketio.emit('state_update', {
                'type': 'setting',
                'payload': {'thinkbot_model': thinkbot_model},
                'username': who
            }, room=who)
        else:
            # forward raw settings payload
            print(f"[SETTING] received payload (no model detected): {data_payload}")
            socketio.emit('state_update', {
                'type': 'setting',
                'payload': data_payload,
                'username': who
            }, room=who)
        return

##    # --- RESET RAG ---
##    if t == 'reset_rag':
##        print(f"[RAG] reset_rag requested by {user}")
##
##        try:
##            if chromadb is None:
##                raise RuntimeError("chromadb not available in this environment")
##
##            # Ensure we have a client (create lazily if needed)
##            if client is None:
##                client = get_chroma_client(persist_directory=CHROMA_DIR)
##
##            # Try to delete the collection
##            try:
##                if client is not None and hasattr(client, "delete_collection"):
##                    client.delete_collection("docs")
##                    print("[RAG] client.delete_collection('docs') succeeded")
##                else:
##                    raise AttributeError("client.delete_collection not available")
##            except Exception as exc_delete:
##                print(f"[RAG] delete_collection not available/supported: {exc_delete!r}")
##                # Fallback: remove persist dir entirely
##                import shutil
##                if os.path.exists(CHROMA_DIR):
##                    shutil.rmtree(CHROMA_DIR)
##                os.makedirs(CHROMA_DIR, exist_ok=True)
##                print(f"[RAG] Persist directory {CHROMA_DIR} reset")
##
##            # Reset singletons
##            global _CHROMA_CLIENT
##            _CHROMA_CLIENT = None
##            client = None
##            collection = None
##
##            # Recreate empty client & collection
##            client = get_chroma_client(persist_directory=CHROMA_DIR)
##            if client is not None:
##                try:
##                    collection = client.get_or_create_collection(name="docs")
##                    # Some versions support collection.delete() to clear content
##                    try:
##                        if hasattr(collection, "delete"):
##                            collection.delete()
##                    except Exception:
##                        pass
##                except Exception as e_create:
##                    print(f"[RAG] Warning: could not create/clear collection after reset: {e_create}")
##                    collection = None
##
##            print("[RAG] Reset completed successfully")
##            socketio.emit('state_update', {
##                'type': 'reset_rag_ack',
##                'payload': 'RAG index cleared',
##                'username': user
##            }, room=user)
##
##        except Exception as e:
##            print(f"[RAG] Reset failed: {e}")
##            socketio.emit('state_update', {
##                'type': 'reset_rag_error',
##                'payload': f'RAG reset failed: {e}',
##                'username': user
##            }, room=user)
##        return


    # --- RESET RAG ---
    if t == 'reset_rag':
        print(f"[RAG] reset_rag requested by {user}")

        try:
            if chromadb is None:
                raise RuntimeError("chromadb not available in this environment")

            # Ensure we have a client (create lazily if needed)
            if client is None:
                client = get_chroma_client(persist_directory=CHROMA_DIR)

            # 1. Try to delete all collections
            try:
                for col in client.list_collections():
                    try:
                        client.delete_collection(col.name)
                        print(f"[RAG] Deleted collection: {col.name}")
                    except Exception as exc_delete:
                        print(f"[RAG] Could not delete {col.name}: {exc_delete}")
            except Exception as exc_list:
                print(f"[RAG] Could not list collections: {exc_list}")

            # 2. Fallback: remove persist dir entirely
            if os.path.exists(CHROMA_DIR):
                try:
                    import shutil
                    shutil.rmtree(CHROMA_DIR)
                    print(f"[RAG] Persist directory {CHROMA_DIR} wiped")
                except Exception as exc_rm:
                    print(f"[RAG] Could not remove persist dir: {exc_rm}")
            os.makedirs(CHROMA_DIR, exist_ok=True)

            # 3. Reset singletons
            global _CHROMA_CLIENT
            _CHROMA_CLIENT = None
            client = None
            collection = None

            # 4. Recreate empty client & default collection (docs)
            client = get_chroma_client(persist_directory=CHROMA_DIR)
            try:
                collection = client.get_or_create_collection(name="docs")
                # Optional: clear contents if supported
                if hasattr(collection, "delete"):
                    try:
                        collection.delete()
                    except Exception:
                        pass
            except Exception as e_create:
                print(f"[RAG] Warning: could not create/clear collection after reset: {e_create}")
                collection = None

            print("[RAG] ✅ Full reset completed successfully")
            socketio.emit('state_update', {
                'type': 'reset_rag_ack',
                'payload': 'All RAG collections cleared',
                'username': user
            }, room=user)

        except Exception as e:
            print(f"[RAG] ❌ Reset failed: {e}")
            socketio.emit('state_update', {
                'type': 'reset_rag_error',
                'payload': f'RAG reset failed: {e}',
                'username': user
            }, room=user)
        return


    # --- LOAD FILE ---
    if t == 'load_file':
        my_file_content = ""

        # Validate payload
        if not isinstance(payload, dict):
            print(f"[ERROR] load_file called with invalid payload: {payload!r}")
            socketio.emit('state_update', {
                'type': 'file_error',
                'payload': "Invalid or missing payload for load_file",
                'username': user
            }, room=user)
            return

        filename = payload.get('filename')
        filedata_b64 = payload.get('filedata')
        mime_hint = payload.get('mime') or payload.get('type')

        print(f"[DEBUG] load_file payload keys: {list(payload.keys())}")

        if not filename or filedata_b64 is None:
            print(f"[ERROR] load_file missing filename or filedata. filename={filename!r}, filedata_present={filedata_b64 is not None}")
            socketio.emit('state_update', {
                'type': 'file_error',
                'payload': {
                    'message': 'Missing filename or filedata',
                    'received_keys': list(payload.keys())
                },
                'username': user
            }, room=user)
            return

        # Decode base64 to bytes
        try:
            file_bytes = base64.b64decode(filedata_b64)
        except Exception as e:
            print(f"[ERROR] base64 decode failed: {e}; payload_snippet={str(filedata_b64)[:80]}")
            socketio.emit('state_update', {'type': 'file_error', 'payload': f'base64 decode failed: {e}', 'username': user}, room=user)
            return

        # Save raw bytes
        safe_name = secure_filename(filename)
        tmp_path = os.path.join(TMP_DIR, safe_name)
        try:
            with open(tmp_path, 'wb') as f:
                f.write(file_bytes)
            print(f"[INFO] Saved uploaded file to {tmp_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save uploaded file: {e}")
            socketio.emit('state_update', {'type': 'file_error', 'payload': f'Failed to save file: {e}', 'username': user}, room=user)
            return

        # Determine mime type
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type and mime_hint:
            mime_type = mime_hint

        # Extract text from PDF if applicable
        if (mime_type == 'application/pdf') or filename.lower().endswith('.pdf'):
            try:
                extracted_text = ""
                if pdfplumber is not None:
                    with pdfplumber.open(tmp_path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                extracted_text += text + "\n"
                else:
                    print("[WARN] pdfplumber not available; trying OCR fallback")
                    extracted_text = extract_pdf_text_with_ocr(tmp_path)

                my_file_content = (extracted_text or "").encode('utf-8', errors='replace').decode('utf-8')
                print(f"[INFO] Extracted text from PDF ({len(my_file_content)} chars)")
            except Exception as e:
                print(f"[ERROR] Failed to extract text from PDF: {e}")
                my_file_content = extract_pdf_text_with_ocr(tmp_path)  # last-chance OCR

        # If not PDF or extraction failed, try text file fallback
        elif mime_type and (mime_type.startswith('text') or filename.lower().endswith(('.txt', '.md', '.py'))):
            try:
                with open(tmp_path, 'rb') as f:
                    raw_data = f.read()
                    if chardet is not None:
                        detected_encoding = chardet.detect(raw_data).get('encoding') or 'utf-8'
                    else:
                        detected_encoding = 'utf-8'
                    print(f"[INFO] Detected file encoding: {detected_encoding}")
                    decoded_content = raw_data.decode(detected_encoding, errors='replace')
                    my_file_content = decoded_content.encode('utf-8').decode('utf-8')
            except Exception as e:
                print(f"[ERROR] Failed to decode text file: {e}")
                my_file_content = ""

        # Fallback decode raw bytes if still no content
        if not my_file_content:
            try:
                my_file_content = file_bytes.decode('utf-8', errors='ignore')
            except Exception:
                my_file_content = file_bytes.decode('latin-1', errors='ignore')

        # Use my_file_content for embedding / RAG
        file_text = my_file_content

        # === RAG embedding & storage ===
        try:
            # Initialize Chroma
            client = None
            if chromadb is not None:
                try:
                    client = chromadb.Client(ChromaSettings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=CHROMA_DIR
                    ))
                    print("[RAG] Connected to embedded Chroma (duckdb+parquet).")
                except Exception as e_init:
                    print(f"[RAG] Embedded Chroma init failed: {e_init}. Trying default chromadb.Client()...")
                    try:
                        client = chromadb.Client()
                        print("[RAG] Connected to Chroma via default client.")
                    except Exception as e_client:
                        print(f"[RAG] chromadb.Client() failed: {e_client}. Will fallback to disk storage.")
                        client = None
            else:
                print("[RAG] chromadb not installed; will fallback to local index")

            # Create embedding
            embedding = None
            emb_source = None
            if file_text and file_text.strip():
                try:
                    if ollama is None:
                        raise RuntimeError('ollama package not available')
                    emb_resp = ollama.embeddings(model="mxbai-embed-large", prompt=file_text)
                    embedding = emb_resp.get('embedding')
                    emb_source = 'ollama'
                    if embedding is None:
                        raise RuntimeError('Ollama returned no embedding')
                    print("[RAG] Got embedding from Ollama.")
                except Exception as e_oll:
                    print(f"[RAG] Ollama embeddings failed: {e_oll} — trying sentence-transformers fallback.")
                    try:
                        from sentence_transformers import SentenceTransformer
                        st_model = SentenceTransformer('all-MiniLM-L6-v2')
                        vect = st_model.encode(file_text)
                        embedding = vect.tolist() if hasattr(vect, "tolist") else list(vect)
                        emb_source = 'sentence-transformers'
                        print("[RAG] Got embedding from sentence-transformers fallback.")
                    except Exception as e_st:
                        print(f"[RAG] sentence-transformers fallback failed: {e_st}")
                        embedding = None
                        emb_source = None
            else:
                print("[RAG] Empty file_text; skipping embedding")

            # Store in Chroma or fallback
            doc_id = f"{user}_{safe_name}_{int(time.time())}"
            if embedding is not None and client is not None:
                try:
                    collection = client.get_or_create_collection(name="docs")
                    collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[file_text]
                    )
                    try:
                        client.persist()
                    except Exception:
                        pass
                    print(f"[RAG] Stored document in Chroma: id={doc_id} len={len(file_text)}")
                except Exception as e_store:
                    print(f"[RAG] Failed to store in Chroma: {e_store}")
                    client = None

            if embedding is None or client is None:
                # Fallback tiny local index
                fallback_index = os.path.join(TMP_DIR, "local_rag_index.json")
                try:
                    if os.path.exists(fallback_index):
                        with open(fallback_index, 'r', encoding='utf-8') as fh:
                            idx = json.load(fh)
                    else:
                        idx = []
                except Exception:
                    idx = []
                idx_entry = {
                    'doc_id': doc_id,
                    'username': user,
                    'filename': filename,
                    'path': tmp_path,
                    'length': len(file_text),
                    'embedding_source': emb_source,
                    'timestamp': datetime.now().isoformat()
                }
                idx.append(idx_entry)
                try:
                    with open(fallback_index, 'w', encoding='utf-8') as fh:
                        json.dump(idx, fh, ensure_ascii=False, indent=2)
                    print(f"[RAG] Saved document metadata to local_rag_index.json id={doc_id}")
                except Exception as e_idx:
                    print(f"[RAG] Failed to write fallback index: {e_idx}")

            # Log query history
            entry = {
                "username": user,
                "timestamp": datetime.now().isoformat(),
                "query": f"[file_upload] {filename}",
                "response": "[stored in RAG]" if (embedding is not None and client is not None) else "[saved - RAG unavailable]",
                "models": {"embed_model": emb_source or "none"}
            }
            log_user_query(user, entry)

            # Notify frontend
            socketio.emit('state_update', {
                'type': 'file_info',
                'payload': {
                    'message': f"Text '{filename}' processed for RAG" if (embedding is not None and client is not None) else f"Text '{filename}' saved (RAG unavailable)",
                    'doc_id': doc_id,
                    'length': len(file_text),
                    'rag_available': embedding is not None and client is not None,
                    'embed_source': emb_source
                },
                'username': user
            }, room=user)

        except Exception as e:
            print(f"[ERROR] RAG processing failed (unexpected): {e}")
            socketio.emit('state_update', {'type': 'file_error', 'payload': str(e), 'username': user}, room=user)
        return

    # --- QUERY ---
    if t == 'query':
        # Use selected model (fallback default)
        model_selected = thinkbot_model or "tinyllama"
        print(f"[EVENT] QUERY MODEL SELECTED is now {model_selected}")

        # Normalize payload -> get text and models dict
        if isinstance(payload, str):
            user_text = payload
            models_info = {}
        elif isinstance(payload, dict):
            user_text = payload.get('query', '') or payload.get('text', '')
            models_info = payload.get('models', {}) or {}
        else:
            user_text = ''
            models_info = {}

        if not user_text:
            socketio.emit('state_update', {
                'type': 'resp',
                'payload': {'error': 'Empty query'},
                'username': user
            }, room=user)
            return

        print(f"[EVENT] query from user='{user}' text='{user_text}' collection_present={bool(collection)}")

        # If we have a collection, run RAG retrieval first
        rag_context = None
        if collection is not None and client is not None and ollama is not None:
            try:
                emb_resp = ollama.embeddings(prompt="rag " + user_text, model="mxbai-embed-large")
                query_emb = emb_resp.get("embedding")
                if query_emb is not None:
                    results = collection.query(query_embeddings=[query_emb], n_results=1)
                    docs = results.get('documents') if isinstance(results, dict) else None
                    if docs and len(docs) > 0 and len(docs[0]) > 0:
                        rag_context = docs[0][0]
                        print(f"[RAG] Retrieved document len={len(rag_context)}")
                    else:
                        print("[RAG] No documents returned from collection.query()")
                else:
                    print("[RAG] embeddings call returned no embedding")
            except Exception as e:
                print(f"[RAG] retrieval failed: {e}")
                rag_context = None

        # Build clean one-line context (if any)
        clean_text = ""
        if rag_context:
            one_line = " ".join(line.strip() for line in str(rag_context).splitlines() if line.strip())
            clean_text = re.sub(r'[:\-\|\\/]', ' ', one_line)
            clean_text = re.sub(
                r'\bdear sirs, thank you for your calling\.?\s*please call again soon\.?\b',
                '',
                clean_text,
                flags=re.IGNORECASE
            )
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            print(f"[RAG] clean_text (len={len(clean_text)}): {clean_text[:200]}")

        # Final prompt to model
        if clean_text:
            model_prompt = f"Respond to this prompt: {user_text}\n\nUsing this data: {clean_text}"
        else:
            model_prompt = f"Respond to this prompt: {user_text}"

        # Log the incoming query
        entry = {
            "username": user,
            "timestamp": datetime.now().isoformat(),
            "query": model_prompt,
            "response": "",
            "models": models_info
        }
        log_user_query(user, entry)

        # Attempt to generate with ollama
        response_text = None
        try:
            if ollama is None:
                raise RuntimeError("Ollama not available")
            if hasattr(ollama, "generate"):
                gen = ollama.generate(model=model_selected, prompt=model_prompt)
                response_text = gen.get("text") or gen.get("content") or gen.get("response") or str(gen)
            elif hasattr(ollama, "completions") and hasattr(ollama.completions, "create"):
                gen = ollama.completions.create(model=model_selected, prompt=model_prompt, max_tokens=1024)
                response_text = gen.get("choices", [{}])[0].get("text") if isinstance(gen, dict) else str(gen)
            elif hasattr(ollama, "create"):
                gen = ollama.create(model=model_selected, prompt=model_prompt, max_tokens=512)
                response_text = gen.get("text") or str(gen)
            else:
                raise RuntimeError("No supported ollama generation method found (generate/create/completions.create).")
        except Exception as e:
            print(f"[ERROR] Model generation failed: {e}")
            response_text = f"[ERROR] Model generation failed: {e}"

        # Save response to logs
        try:
            entry['response'] = response_text
            log_user_query(user, entry)
        except Exception as e:
            print(f"[WARN] Failed to log response: {e}")

        # Emit the response back to the front end
        socketio.emit('state_update', {
            'type': 'resp',
            'payload': {
                'query': user_text,
                'response': response_text,
                'rag_used': bool(clean_text),
                'embed_source': models_info.get('embed_model')
            },
            'username': user
        }, room=user)

##        full_emb_prompt = f"Only respond exactly what was asked, nothing more. Respond to this prompt: {user_text} Using this data: {clean_text}"
##
##        socketio.emit('state_update', {
##            'type': 'query',
##            'payload': {
##                'query': full_emb_prompt,
##                'response': "",
##                'rag_used': bool(clean_text),
##                'embed_source': models_info.get('embed_model')
##            },
##            'username': user
##        }, room=user)
        
        full_emb_prompt = f"Only respond exactly what was asked, nothing more. Respond to this prompt: {user_text} Using this data: {clean_text}"

        socketio.emit('state_update', {
            'type': 'query',
            'payload': {
                'query': full_emb_prompt,
                'response': "",
                'rag_used': bool(clean_text),
                'embed_source': models_info.get('embed_model')
            },
            'username': user
        })
        return


##import ollama
##import re
##import os
##import json
##import time
##import base64
##import mimetypes
##import tempfile
##from datetime import datetime
##from werkzeug.utils import secure_filename
##
##import whisper
##from flask import Flask, request, jsonify, send_from_directory
##from flask_socketio import SocketIO, emit, join_room, leave_room
##
### Optional libs (may not be installed in every env)
##try:
##    import chromadb
##except Exception:
##    chromadb = None
##try:
##    import ollama
##except Exception:
##    ollama = None
##
##from listenWEBUI import WEBUIListenModule
##from auth_manager import login as do_login, signup as do_signup, get_users, delete_user
##from query_logger import log_user_query, log_user_response, read_logs
##
##import chardet
##from pdfminer.high_level import extract_text
##from pdf2image import convert_from_path
##import pytesseract
##import base64
##import io
##import PyPDF2
##from PyPDF2 import PdfFileReader
##import pdfplumber
##from langchain_chroma import Chroma
##
##BASE_DIR = os.path.dirname(os.path.abspath(__file__))
##TMP_DIR = os.path.join(BASE_DIR, 'tmp')
##os.makedirs(TMP_DIR, exist_ok=True)
##
### Initialize listener and Whisper
##webui_listener = WEBUIListenModule()
##whisper_model = whisper.load_model("base.en")
##
##app = Flask(__name__, static_folder='./webui-src/dist', static_url_path='')
##socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
##
##
####chroma_client = chromadb.Client(
####    settings=chromadb.config.Settings(anonymized_telemetry=False)
####)
##
### Module-level state
##state = {
##    'logs': [],
##    'queries': [],
##    'resp': [],
##    'settings': {
##        'use_whisper': False,
##        'use_vosk': True,
##        'enter_submits': True
##    }
##}
##
### Session tracking
##session_users = {}  # Maps SID -> username
##
##my_file_content = ""
##
##collection = None   # Chroma collection shared across events
##os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
##
##from langchain.text_splitter import RecursiveCharacterTextSplitter
##import PyPDF2
##
##
##model_selected = ""
##thinkbot_model = ""
##rag_clear = False
##
### ======= Query helper (can be called from elsewhere in backend) =======
##def query_with_retrieval(prompt: str, n_results: int = 1):
##    """Retrieve top chunks from Chroma and run generation with Ollama."""
##    try:
##        if ollama is None or chromadb is None:
##            raise RuntimeError("Ollama or Chroma not available")
##
##        q_emb = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)
##        q_vector = q_emb.get("embedding") if isinstance(q_emb, dict) else q_emb
##        client = chromadb.Client()
##        collection = client.get_or_create_collection(name="docs")
##        res = collection.query(query_embeddings=[q_vector], n_results=n_results)
##        top_doc = res.get("documents", [[]])[0][0] if res.get("documents") else ""
##    except Exception as e:
##        print(f"[WARN] retrieval failed: {e}")
##        top_doc = ""
##
##    gen_prompt = f"Using this data: {top_doc}\n\nRespond to: {prompt}"
##    try:
##        out = ollama.generate(model="tinyllama", prompt=gen_prompt, stream=False)
##        text = out.get("response") or out.get("text") or str(out)
##    except Exception as e:
##        text = f"[generation failed: {e}]"
##    return text
##
##
##
##
##def extract_pdf_text_with_ocr(pdf_path):
##    try:
##        text = extract_text(pdf_path)
##        if len(text.strip()) > 10:  # if decent text extracted, return it
##            return text
##        else:
##            # Fallback to OCR if text is too short or empty
##            print("[INFO] PDF text extraction empty or too short, using OCR fallback.")
##            pages = convert_from_path(pdf_path)
##            ocr_text = ""
##            for page in pages:
##                ocr_text += pytesseract.image_to_string(page)
##            return ocr_text
##    except Exception as e:
##        print(f"[ERROR] PDF extraction + OCR failed: {e}")
##        return ""
##
##
### Utility: safe save
##def _save_tmp(filename, data_bytes):
##    safe = secure_filename(filename)
##    path = os.path.join(TMP_DIR, safe)
##    with open(path, 'wb') as f:
##        f.write(data_bytes)
##    return path
##
### Socket handlers
##@socketio.on('connect')
##def connect():
##    emit('full_state', state)
##
##@socketio.on('disconnect')
##def disconnect():
##    sid = request.sid
##    user = session_users.pop(sid, None)
##    if user:
##        leave_room(user)
##        print(f"[DISCONNECT] {sid} left room {user}")
##
##
### --- helper: singleton chroma client creator (safe) ---
##
##_CHROMA_CLIENT = None
##
##def get_chroma_client(persist_directory=None, chroma_db_impl="duckdb+parquet"):
##    global _CHROMA_CLIENT
##    if _CHROMA_CLIENT is not None:
##        return _CHROMA_CLIENT
##    try:
##        import chromadb
##        from chromadb.config import Settings as ChromaSettings
##    except Exception as e:
##        print(f"[RAG] chromadb import failed: {e}")
##        return None
##
##    try:
##        settings = ChromaSettings(chroma_db_impl=chroma_db_impl,
##                                  persist_directory=persist_directory)
##        _CHROMA_CLIENT = chromadb.Client(settings=settings)
##        print(f"[RAG] Chroma client created with persist_directory={persist_directory}")
##        return _CHROMA_CLIENT
##    except Exception as e:
##        print(f"[RAG] Embedded Chroma init failed: {e}; trying fallback chromadb.Client()")
##        try:
##            _CHROMA_CLIENT = chromadb.Client()
##            print("[RAG] Chroma client created via fallback chromadb.Client()")
##            return _CHROMA_CLIENT
##        except Exception as e2:
##            print(f"[RAG] chromadb.Client() fallback failed: {e2}")
##            _CHROMA_CLIENT = None
##            return None
##
##@socketio.on('gui_event')
##def handle_event(data):
##    global client, collection, thinkbot_model, model_selected, rag_clear  # Declare globals at the very top
##    sid = request.sid
##    print(f"[DEBUG] Received gui_event from sid={sid}: {data!r}")
##
##    user = data.get('username') if isinstance(data, dict) else None
##    t = data.get('type') if isinstance(data, dict) else None
##    payload = data.get('payload') if isinstance(data, dict) else None
##
##    if not user:
##        user = 'Tjaart'  # fallback
##
##    # Track user per session
##    session_users[sid] = user
##    join_room(user)
##    global my_file_content
##    
##    # --- LOGIN ---
##    if t == 'login':
##        print(f"[LOGIN] User: {user}")
##        socketio.emit('state_update', {
##            'type': 'login_ack',
##            'username': user
##        }, room=user)
##        return
##
##    # --- LOG ---
##    if t == 'log':
##        if isinstance(payload, str):
##            log_user_response(user, payload)
##            socketio.emit('state_update', {
##                'type': 'log',
##                'payload': payload,
##                'username': user
##            }, room=user)
##
##        elif isinstance(payload, dict):
##            entry = {
##                "username": user,
##                "timestamp": datetime.now().isoformat(),
##                "query": payload.get("query", ""),
##                "response": payload.get("response", ""),
##                "models": payload.get("models", {})
##            }
##            log_user_query(user, entry)
##            socketio.emit('state_update', {
##                'type': 'log',
##                'payload': entry,
##                'username': user
##            }, room=user)
##            socketio.emit('state_update', {
##                'type': 'query',
##                'payload': entry,
##                'username': user
##            }, room=user)
##        return
##
##    # --- RESP ---
##    if t == 'resp':
##        if isinstance(payload, str):
##            log_user_response(user, payload)
##            socketio.emit('state_update', {
##                'type': 'log',
##                'payload': payload,
##                'username': user
##            }, room=user)
##
##        elif isinstance(payload, dict):
##            entry = {
##                "username": user,
##                "timestamp": datetime.now().isoformat(),
##                "query": payload.get("query", ""),
##                "response": payload.get("response", ""),
##                "models": payload.get("models", {})
##            }
##            log_user_query(user, entry)
##            socketio.emit('state_update', {
##                'type': 'log',
##                'payload': entry,
##                'username': user
##            }, room=user)
##            socketio.emit('state_update', {
##                'type': 'query',
##                'payload': entry,
##                'username': user
##            }, room=user)
##        return
##
##
##
##    # --- SETTING ---
##    if t == 'setting':
##        socketio.emit('state_update', {
##            'type': 'setting',
##            'payload': payload,
##            'username': user
##        }, room=user)
##        return
##
##
##    if t in ('setting', 'think_model'):
##        data_payload = data.get('payload')
##        user = data.get('username') or 'default_user'
##
##        # Try to extract the thinkbot model name from different payload shapes:
##        model_selected = None
##        
##        if isinstance(data_payload, dict):
##            # common: payload = { 'thinkbot_model': 'tinyllama' }
##            if 'thinkbot_model' in data_payload:
##                model_selected = data_payload['thinkbot_model']
##            else:
##                # fallback: payload may be { '<some_model_key>': '<name>' }
##                # find any key that ends with '_model' and pick its value
##                model_keys = [k for k in data_payload.keys() if k.endswith('_model')]
##                if len(model_keys) == 1:
##                    model_selected = data_payload[model_keys[0]]
##        elif isinstance(data_payload, str):
##            # sometimes frontend sends a raw string payload (e.g. from 'think_model' emit)
##            model_selected = data_payload
##
##        # If we found a model, assign and notify
##        if model_selected:
##            # ensure we set the module/global variable (declare `global thinkbot_model` above if needed)
##            try:
##                global thinkbot_model
##            except NameError:
##                # if thinkbot_model doesn't exist yet, create it
##                thinkbot_model = None
##
##            thinkbot_model = model_selected  # correct assignment (not ==)
##            print(f"The thinkbot_model for RAG is '{thinkbot_model}'")
##            print(f"The new thinkbot_model is now '{thinkbot_model}'")
##
##            # Emit a consistent state_update so clients can sync
##            socketio.emit('state_update', {
##                'type': 'setting',
##                'payload': {'thinkbot_model': thinkbot_model},
##                'username': user
##            }, room=user)
##        else:
##            # No model found — still forward the raw payload for other settings
##            print(f"[setting] received payload but no model detected: {data_payload}")
##            socketio.emit('state_update', {
##                'type': 'setting',
##                'payload': data_payload,
##                'username': user
##            }, room=user)
##
##        return
##
##
##    # --- SETTING ---
##    if t == 'think_model':
##
##        payload = data.get('payload')
##        user = data.get('username') or 'default_user'
##        model_selected = data.get('think_model')
##
##        print(f"The model for RAG is {model_selected}")
####        socketio.emit('state_update', {
####            'type': 'setting',
####            'payload': payload,
####            'username': user
####        }, room=user)
##
##        return
##
##
##    # --- RESET RAG ---
##    if t == 'reset_rag':
##        global client, collection, _CHROMA_CLIENT
##        user = data.get('username') or user
##        print(f"[RAG] reset_rag requested by {user}")
##
##        try:
##            # Ensure we have a client (create lazily if needed)
##            if chromadb is None:
##                raise RuntimeError("chromadb not available in this environment")
##
##            if client is None:
##                client = get_chroma_client(persist_directory=CHROMA_DIR)
##
##            # First attempt: use Chroma API to delete the collection if available
##            try:
##                if client is not None and hasattr(client, "delete_collection"):
##                    client.delete_collection("docs")
##                    print("[RAG] client.delete_collection('docs') succeeded")
##                else:
##                    # client API does not expose delete_collection in this version
##                    raise AttributeError("client.delete_collection not available")
##            except Exception as exc_delete:
##                print(f"[RAG] client.delete_collection failed or not available: {exc_delete!r}")
##                # Fallback: remove persist directory completely (will wipe stored index)
##                try:
##                    import shutil
##                    if os.path.exists(CHROMA_DIR):
##                        shutil.rmtree(CHROMA_DIR)
##                        os.makedirs(CHROMA_DIR, exist_ok=True)
##                        print(f"[RAG] Persist directory {CHROMA_DIR} removed and recreated")
##                    else:
##                        print(f"[RAG] Persist directory {CHROMA_DIR} did not exist; nothing to remove")
##                except Exception as exc_sh:
##                    raise RuntimeError(f"Failed to remove persist directory fallback: {exc_sh}")
##
##            # Reset client/collection singletons so next operations create a fresh client
##            _CHROMA_CLIENT = None
##            client = None
##            collection = None
##
##            # Recreate empty client & collection so the system remains usable immediately
##            client = get_chroma_client(persist_directory=CHROMA_DIR)
##            if client is not None:
##                try:
##                    collection = client.get_or_create_collection(name="docs")
##                    # Some chroma versions allow collection.delete() to clear content; call defensively
##                    try:
##                        if hasattr(collection, "delete"):
##                            # many implementations accept no-args delete() to clear all docs; if it errors, ignore
##                            collection.delete()
##                    except Exception:
##                        # ignore - collection already fresh or delete requires ids
##                        pass
##                except Exception as e_create:
##                    print(f"[RAG] Warning: could not create/clear collection after reset: {e_create}")
##                    collection = None
##
##            print("[RAG] Reset completed successfully")
##            socketio.emit('state_update', {
##                'type': 'reset_rag_ack',
##                'payload': 'RAG index cleared',
##                'username': user
##            }, room=user)
##
##        except Exception as e:
##            print(f"[RAG] Reset failed: {e}")
##            socketio.emit('state_update', {
##                'type': 'reset_rag_error',
##                'payload': f'RAG reset failed: {e}',
##                'username': user
##            }, room=user)
##
##        return
##
##
##    # --- LOAD FILE ---
##    
##   # inside your socketio event handler function, e.g.:
##    if t == 'load_file':
##        my_file_content = ""
##
##        # Validate payload
##        if not isinstance(payload, dict):
##            print(f"[ERROR] load_file called with invalid payload: {payload!r}")
##            socketio.emit('state_update', {
##                'type': 'file_error',
##                'payload': "Invalid or missing payload for load_file",
##                'username': user
##            }, room=user)
##            return
##
##        filename = payload.get('filename')
##        filedata_b64 = payload.get('filedata')
##        mime_hint = payload.get('mime') or payload.get('type')
##
##        print(f"[DEBUG] load_file payload keys: {list(payload.keys())}")
##
##        if not filename or filedata_b64 is None:
##            print(f"[ERROR] load_file missing filename or filedata. filename={filename!r}, filedata_present={filedata_b64 is not None}")
##            socketio.emit('state_update', {
##                'type': 'file_error',
##                'payload': {
##                    'message': 'Missing filename or filedata',
##                    'received_keys': list(payload.keys())
##                },
##                'username': user
##            }, room=user)
##            return
##
##        # Decode base64 to bytes
##        try:
##            file_bytes = base64.b64decode(filedata_b64)
##        except Exception as e:
##            print(f"[ERROR] base64 decode failed: {e}; payload_snippet={str(filedata_b64)[:80]}")
##            socketio.emit('state_update', {'type': 'file_error', 'payload': f'base64 decode failed: {e}', 'username': user}, room=user)
##            return
##
##        # Save raw bytes
##        safe_name = secure_filename(filename)
##        tmp_path = os.path.join(TMP_DIR, safe_name)
##        try:
##            with open(tmp_path, 'wb') as f:
##                f.write(file_bytes)
##            print(f"[INFO] Saved uploaded file to {tmp_path}")
##        except Exception as e:
##            print(f"[ERROR] Failed to save uploaded file: {e}")
##            return
##
##        # Determine mime type
##        mime_type, _ = mimetypes.guess_type(filename)
##        if not mime_type and mime_hint:
##            mime_type = mime_hint
##
##        # Extract text from PDF if applicable
##        if mime_type == 'application/pdf' or filename.lower().endswith('.pdf'):
##            try:
##                extracted_text = ""
##                with pdfplumber.open(tmp_path) as pdf:
##                    for page in pdf.pages:
##                        text = page.extract_text()
##                        if text:
##                            extracted_text += text + "\n"
##                my_file_content = extracted_text.encode('utf-8', errors='replace').decode('utf-8')
##                print(f"[INFO] Extracted text from PDF ({len(my_file_content)} chars)")
##            except Exception as e:
##                print(f"[ERROR] Failed to extract text from PDF: {e}")
##                my_file_content = ""
##
##        # If not PDF or extraction failed, try text file fallback
##        elif mime_type and (mime_type.startswith('text') or filename.lower().endswith(('.txt', '.md', '.py'))):
##            try:
##                with open(tmp_path, 'rb') as f:
##                    raw_data = f.read()
##                    import chardet
##                    detected_encoding = chardet.detect(raw_data).get('encoding') or 'utf-8'
##                    print(f"[INFO] Detected file encoding: {detected_encoding}")
##                    decoded_content = raw_data.decode(detected_encoding, errors='replace')
##                    my_file_content = decoded_content.encode('utf-8').decode('utf-8')
##            except Exception as e:
##                print(f"[ERROR] Failed to decode text file: {e}")
##                my_file_content = ""
##
##        # Fallback decode raw bytes if still no content
##        if not my_file_content:
##            try:
##                my_file_content = file_bytes.decode('utf-8', errors='ignore')
##            except Exception:
##                my_file_content = file_bytes.decode('latin-1', errors='ignore')
##
##        # Now you can use my_file_content for embedding / RAG as before
##        file_text = my_file_content
##
##        # === Your existing RAG embedding & storage code here ===
##        try:
##            # Initialize Chroma
##            try:
##                from chromadb.config import Settings as ChromaSettings
##                chroma_settings = ChromaSettings(
##                    chroma_db_impl="duckdb+parquet",
##                    persist_directory=os.path.join(BASE_DIR, "chroma_db")
##                )
##                client = chromadb.Client(chroma_settings)
##                print("[RAG] Connected to embedded Chroma (duckdb+parquet).")
##            except Exception as e_init:
##                print(f"[RAG] Embedded Chroma init failed: {e_init}. Trying default chromadb.Client()...")
##                try:
##                    client = chromadb.Client()
##                    print("[RAG] Connected to Chroma via default client.")
##                except Exception as e_client:
##                    print(f"[RAG] chromadb.Client() failed: {e_client}. Will fallback to disk storage.")
##                    client = None
##
##            # Create embedding
##            embedding = None
##            emb_source = None
##            try:
##                if ollama is None:
##                    raise RuntimeError('ollama package not available')
##                emb_resp = ollama.embeddings(model="mxbai-embed-large", prompt=file_text)
##                embedding = emb_resp.get('embedding')
##                emb_source = 'ollama'
##                if embedding is None:
##                    raise RuntimeError('Ollama returned no embedding')
##                print("[RAG] Got embedding from Ollama.")
##            except Exception as e_oll:
##                print(f"[RAG] Ollama embeddings failed: {e_oll} — falling back to sentence-transformers if available.")
##                try:
##                    from sentence_transformers import SentenceTransformer
##                    st_model = SentenceTransformer('all-MiniLM-L6-v2')
##                    vect = st_model.encode(file_text)
##                    embedding = vect.tolist() if hasattr(vect, "tolist") else list(vect)
##                    emb_source = 'sentence-transformers'
##                    print("[RAG] Got embedding from sentence-transformers fallback.")
##                except Exception as e_st:
##                    print(f"[RAG] sentence-transformers fallback failed: {e_st}")
##                    embedding = None
##                    emb_source = None
##
##            # Store in Chroma or fallback
##            doc_id = f"{user}_{safe_name}_{int(time.time())}"
##            if embedding is not None and client is not None:
##                try:
##                    collection = client.get_or_create_collection(name="docs")
##                    collection.add(
##                        ids=[doc_id],
##                        embeddings=[embedding],
##                        documents=[file_text]
##                    )
##                    try:
##                        client.persist()
##                    except Exception:
##                        pass
##                    print(f"[RAG] Stored document in Chroma: id={doc_id} len={len(file_text)}")
##                except Exception as e_store:
##                    print(f"[RAG] Failed to store in Chroma: {e_store}")
##                    client = None
##
##            if embedding is None or client is None:
##                fallback_index = os.path.join(TMP_DIR, "local_rag_index.json")
##                try:
##                    if os.path.exists(fallback_index):
##                        with open(fallback_index, 'r', encoding='utf-8') as fh:
##                            idx = json.load(fh)
##                    else:
##                        idx = []
##                except Exception:
##                    idx = []
##                idx_entry = {
##                    'doc_id': doc_id,
##                    'username': user,
##                    'filename': filename,
##                    'path': tmp_path,
##                    'length': len(file_text),
##                    'embedding_source': emb_source,
##                    'timestamp': datetime.now().isoformat()
##                }
##                idx.append(idx_entry)
##                try:
##                    with open(fallback_index, 'w', encoding='utf-8') as fh:
##                        json.dump(idx, fh, ensure_ascii=False, indent=2)
##                    print(f"[RAG] Saved document metadata to local_rag_index.json id={doc_id}")
##                except Exception as e_idx:
##                    print(f"[RAG] Failed to write fallback index: {e_idx}")
##
##            # Log query history
##            entry = {
##                "username": user,
##                "timestamp": datetime.now().isoformat(),
##                "query": f"[file_upload] {filename}",
##                "response": "[stored in RAG]" if embedding is not None else "[saved - RAG unavailable]",
##                "models": {"embed_model": emb_source or "none"}
##            }
##            log_user_query(user, entry)
##
##            # Notify frontend
##            socketio.emit('state_update', {
##                'type': 'file_info',
##                'payload': {
##                    'message': f"Text '{filename}' processed for RAG" if embedding is not None else f"Text '{filename}' saved (RAG unavailable)",
##                    'doc_id': doc_id,
##                    'length': len(file_text),
##                    'rag_available': embedding is not None and client is not None,
##                    'embed_source': emb_source
##                },
##                'username': user
##            }, room=user)
##
##        except Exception as e:
##            print(f"[ERROR] RAG processing failed (unexpected): {e}")
##            socketio.emit('state_update', {'type': 'file_error', 'payload': str(e), 'username': user}, room=user)
##        return
##
##
##    # --- QUERY ---
##    
##    if t == 'query':
##        model_selected = thinkbot_model
##        print(f"[EVENT] QUERY MODEL SELECTED is now {model_selected}")
##
##        # Normalize payload -> get text and models dict
##        if isinstance(payload, str):
##            user_text = payload
##            models_info = {}
##        elif isinstance(payload, dict):
##            user_text = payload.get('query', '') or payload.get('text', '')
##            models_info = payload.get('models', {}) or {}
##        else:
##            user_text = ''
##            models_info = {}
##
##        if not user_text:
##            socketio.emit('state_update', {
##                'type': 'resp',
##                'payload': {'error': 'Empty query'},
##                'username': user
##            }, room=user)
##            return
##
##        print(f"[EVENT] query from user='{user}' text='{user_text}' collection_present={bool(collection)}")
##
##        # If we have a collection, run RAG retrieval first
##        rag_context = None
##        if collection is not None and client is not None:
##            try:
##                emb_resp = ollama.embeddings(prompt="rag " + user_text, model="mxbai-embed-large")
##                query_emb = emb_resp.get("embedding")
##                if query_emb is not None:
##                    results = collection.query(query_embeddings=[query_emb], n_results=1)
##                    # safe extraction
##                    docs = results.get('documents') if isinstance(results, dict) else None
##                    if docs and len(docs) > 0 and len(docs[0]) > 0:
##                        rag_context = docs[0][0]
##                        print(f"[RAG] Retrieved document len={len(rag_context)}")
##                    else:
##                        print("[RAG] No documents returned from collection.query()")
##                else:
##                    print("[RAG] embeddings call returned no embedding")
##            except Exception as e:
##                print(f"[RAG] retrieval failed: {e}")
##                rag_context = None
##
##        # Build clean one-line context (if any)
##        clean_text = ""
##        if rag_context:
##            one_line = " ".join(line.strip() for line in str(rag_context).splitlines() if line.strip())
##            # remove punctuation characters you don't want (fixed regex)
##            clean_text = re.sub(r'[:\-\|\\/]', ' ', one_line)
##            # remove a known garbage sentence if present (case-insensitive)
##            clean_text = re.sub(
##                r'\bdear sirs, thank you for your calling\.?\s*please call again soon\.?\b',
##                '',
##                clean_text,
##                flags=re.IGNORECASE
##            )
##            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
##            print(f"[RAG] clean_text (len={len(clean_text)}): {clean_text[:200]}")
##
##        # Final prompt to model
##        if clean_text:
##            model_prompt = f"Respond to this prompt: {user_text}\n\nUsing this data: {clean_text}"
##        else:
##            model_prompt = f"Respond to this prompt: {user_text}"
##
##        # Log the incoming query
##        entry = {
##            "username": user,
##            "timestamp": datetime.now().isoformat(),
##            "query": model_prompt,
##            "response": "",
##            "models": models_info
##        }
##        log_user_query(user, entry)
##
##        # Attempt to generate with ollama (be tolerant of different client method names)
##        response_text = None
##        try:
##            # try common ollama generation method name(s)
##            if hasattr(ollama, "generate"):
##                gen = ollama.generate(model=model_selected, prompt=model_prompt)
##                # try common keys
##                response_text = gen.get("text") or gen.get("content") or gen.get("response") or str(gen)
##            elif hasattr(ollama, "completions") and hasattr(ollama.completions, "create"):
##                gen = ollama.completions.create(model="mxbai-embed-large", prompt=model_prompt, max_tokens=1024)
##                response_text = gen.get("choices", [{}])[0].get("text") if isinstance(gen, dict) else str(gen)
##            elif hasattr(ollama, "create"):
##                gen = ollama.create(model="mxbai-embed-large", prompt=model_prompt, max_tokens=512)
##                response_text = gen.get("text") or str(gen)
##            else:
##                raise RuntimeError("No supported ollama generation method found (generate/create/completions.create).")
##        except Exception as e:
##            print(f"[ERROR] Model generation failed: {e}")
##            # If model generation fails, return the error to frontend (and keep server-side log)
##            response_text = f"[ERROR] Model generation failed: {e}"
##
##        # Save response to logs
##        try:
##            entry['response'] = response_text
##            log_user_query(user, entry)
##        except Exception as e:
##            print(f"[WARN] Failed to log response: {e}")
##
##        # Emit the response back to the front end
##        socketio.emit('state_update', {
##            'type': 'resp',
##            'payload': {
##                'query': user_text,
##                'response': response_text,
##                'rag_used': bool(clean_text),
##                'embed_source': models_info.get('embed_model')
##            },
##            'username': user
##        }, room=user)
##
##
##        full_emb_prompt = f"Only respond exactly what was asked, nothing more. Respond to this prompt: {user_text} Using this data: {clean_text}"
##
##        # Emit the response back to the front end
##        socketio.emit('state_update', {
##            'type': 'query',
##            'payload': {
##                'query': full_emb_prompt,
##                'response': "",
##                'rag_used': bool(clean_text),
##                'embed_source': models_info.get('embed_model')
##            },
##            'username': user
##        })
##        
##        return




##    # --- QUERY ---
##    if t == 'query':
##
##        if collection is None:
##
##            text = payload if isinstance(payload, str) else payload.get('query', '')
##
##            text = text
##            print(f"[EVENT] No Collection !!!! query from RAG user='{user}' text='{text}'")
##
##            entry = {
##                "username": user,
##                "timestamp": datetime.now().isoformat(),
##                "query": text,
##                "response": "",
##                "models": payload.get('models', {}) if isinstance(payload, dict) else {}
##            }
##
##            log_user_query(user, entry)
##    ##        prompt=f"Using this data: {data}. Respond to this prompt: {prompt}",
##            full_emb_prompt = f" Respond to this prompt: {text}"
##            socketio.emit('state_update', {
##                'type': 'query',
##    ##            'payload': {'query': text, 'response': response_text},
##                'payload': full_emb_prompt,
##                'username': user
##            })
##            return
##    
##        text = payload if isinstance(payload, str) else payload.get('query', '')
##
##        text = " rag " + text
##        print(f"[EVENT] query from RAG user='{user}' text='{text}'")
##
##        # generate an embedding for the prompt and retrieve the most relevant doc
##        response = ollama.embeddings(
##            prompt=text,
##            model="mxbai-embed-large"
##        )
##        results = collection.query(
##            query_embeddings=[response["embedding"]],
##            n_results=1
##        )
##        data = results['documents'][0][0] if results['documents'] and results['documents'][0] else "No relevant documents found."
##        print(f"The data is : {data}")
##
##
##        # Remove newlines, collapse extra spaces
##        one_line = " ".join(line.strip() for line in data.splitlines() if line.strip())
##        print(f"My One sentance is {one_line}")
##
##        # Remove colon, dash, pipe, backslash, forward slash
##        clean_text = re.sub(r'[:\-\|\\/]_', '', one_line)
##
##        # Remove the specific sentence (case-insensitive)
##        clean_text = re.sub(
##            r'\bdear sirs, thank you for your calling\.?\s+please call again soon\.?\b',
##            '',
##            clean_text,
##            flags=re.IGNORECASE
##        )
##
##        # Remove extra spaces after deletion
##        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
##
##
##        print(f"My clean_text sentance is {clean_text}")
##
##        prompt = f" Respond to this prompt: {text} Using this data: {clean_text}."
##
##        entry = {
##            "username": user,
##            "timestamp": datetime.now().isoformat(),
##            "query": prompt,
####            "response": clean_text,
####            "response": "",
##            "models": payload.get('models', {}) if isinstance(payload, dict) else {}
##        }
##
##        log_user_query(user, entry)
##        full_emb_prompt = f"Only respond exactly what was asked, nothing more. Respond to this prompt: {text} Using this data: {clean_text}"
##        socketio.emit('state_update', {
##            'type': 'query',
####            'payload': {'query': text, 'response': response_text},
##            'payload': full_emb_prompt,
##            'username': user
##        })
##        return


##    # --- QUERY ---
##    if t == 'query':
##        text = payload if isinstance(payload, str) else payload.get('query', '')
##        print(f"[EVENT] query from user='{user}' text='{text}'")
##
##        entry = {
##            "username": user,
##            "timestamp": datetime.now().isoformat(),
##            "query": text  + " : " + '\n' + '\n' + my_file_content,
##            "response": "",
##            "models": payload.get('models', {}) if isinstance(payload, dict) else {}
##        }
##
##        log_user_query(user, entry)
##
##        socketio.emit('state_update', {
##            'type': 'query',
##            'payload': {'query': text  + " : " + '\n' + '\n' + my_file_content},
##            'username': user
##        })
##        return



##    # --- LOAD FILE ---
##
##    if t == 'load_file':
##        my_file_content = ""
##
##        # Validate payload
##        if not isinstance(payload, dict):
##            print(f"[ERROR] load_file called with invalid payload: {payload!r}")
##            socketio.emit('state_update', {
##                'type': 'file_error',
##                'payload': "Invalid or missing payload for load_file",
##                'username': user
##            }, room=user)
##            return
##
##        filename = payload.get('filename')
##        filedata_b64 = payload.get('filedata')
##        mime_hint = payload.get('mime') or payload.get('type')
##
##        print(f"[DEBUG] load_file payload keys: {list(payload.keys())}")
##
##        if not filename or filedata_b64 is None:
##            print(f"[ERROR] load_file missing filename or filedata. filename={filename!r}, filedata_present={filedata_b64 is not None}")
##            socketio.emit('state_update', {
##                'type': 'file_error',
##                'payload': {
##                    'message': 'Missing filename or filedata',
##                    'received_keys': list(payload.keys())
##                },
##                'username': user
##            }, room=user)
##            return
##
##        try:
##            file_bytes = base64.b64decode(filedata_b64)
##        except Exception as e:
##            print(f"[ERROR] base64 decode failed: {e}; payload_snippet={str(filedata_b64)[:80]}")
##            socketio.emit('state_update', {'type': 'file_error', 'payload': f'base64 decode failed: {e}', 'username': user}, room=user)
##            return
##
##        # Detect MIME type
##        mime_type, _ = mimetypes.guess_type(filename)
##        if not mime_type and mime_hint:
##            mime_type = mime_hint
##
##        # Handle PDF files directly in memory
##        if mime_type == 'application/pdf' or filename.lower().endswith('.pdf'):
##            try:
##                pdf_stream = io.BytesIO(file_bytes)
##                reader = PyPDF2.PdfReader(pdf_stream)
##                extracted_text = ""
##                for page in reader.pages:
##                    page_text = page.extract_text()
##                    if page_text:
##                        extracted_text += page_text + "\n"
##                my_file_content = extracted_text.encode('utf-8', errors='replace').decode('utf-8')
##                print(f"[INFO] Extracted text from PDF ({len(my_file_content)} chars)")
##            except Exception as e:
##                print(f"[ERROR] Failed to extract text from PDF: {e}")
##                my_file_content = ""
##
##        # Handle text files
##        elif mime_type and (mime_type.startswith('text') or filename.lower().endswith(('.txt', '.md', '.py'))):
##            try:
##                result = chardet.detect(file_bytes)
##                detected_encoding = result['encoding'] or 'utf-8'
##                decoded_content = file_bytes.decode(detected_encoding, errors='replace')
##                my_file_content = decoded_content.encode('utf-8').decode('utf-8')
##                print(f"[INFO] Detected encoding: {detected_encoding}")
##            except Exception as e:
##                print(f"[ERROR] Failed to decode text file: {e}")
##                my_file_content = ""
##
##        else:
##            # For unknown types, save the file for later or ignore text extraction
##            safe_name = secure_filename(filename)
##            tmp_path = os.path.join(TMP_DIR, safe_name)
##            try:
##                with open(tmp_path, 'wb') as f:
##                    f.write(file_bytes)
##                print(f"[INFO] Saved unknown file type to {tmp_path}")
##            except Exception as e:
##                print(f"[ERROR] Failed to save unknown file type: {e}")
##
##            my_file_content = ""
##
##        # Use extracted text or fallback decode
##        if my_file_content:
##            file_text = my_file_content
##        else:
##            try:
##                file_text = file_bytes.decode('utf-8', errors='ignore')
##            except Exception:
##                file_text = file_bytes.decode('latin-1', errors='ignore')
##
##        # Continue with your existing RAG embedding and storage code here
##        # (I assume your code for Chroma, Ollama, sentence-transformers embeddings, etc.)
##        # ...
##
##        # (Example: log and emit success)
##        doc_id = f"{user}_{filename}_{int(time.time())}"
##        print(f"[INFO] Ready to embed document id={doc_id} with length={len(file_text)}")
##        socketio.emit('state_update', {
##            'type': 'file_info',
##            'payload': {
##                'message': f"Text from '{filename}' processed for embedding",
##                'doc_id': doc_id,
##                'length': len(file_text),
##                'rag_available': True,
##                'embed_source': "unknown"  # update based on your embedding source
##            },
##            'username': user
##        }, room=user)
##
##
##        # RAG embedding & storage (same as your existing logic)
##        try:
##            # Initialize Chroma
##            try:
##                from chromadb.config import Settings as ChromaSettings
##                chroma_settings = ChromaSettings(
##                    chroma_db_impl="duckdb+parquet",
##                    persist_directory=os.path.join(BASE_DIR, "chroma_db")
##                )
##                client = chromadb.Client(chroma_settings)
##            except Exception:
##                client = chromadb.Client()
##
##            # Generate embedding (Ollama or fallback)
##            embedding = None
##            emb_source = None
##            try:
##                if ollama is None:
##                    raise RuntimeError('ollama package not available')
##                emb_resp = ollama.embeddings(model="mxbai-embed-large", prompt=file_text)
##                embedding = emb_resp.get('embedding')
##                emb_source = 'ollama'
##                if embedding is None:
##                    raise RuntimeError('Ollama returned no embedding')
##            except Exception:
##                from sentence_transformers import SentenceTransformer
##                st_model = SentenceTransformer('all-MiniLM-L6-v2')
##                vect = st_model.encode(file_text)
##                embedding = vect.tolist() if hasattr(vect, "tolist") else list(vect)
##                emb_source = 'sentence-transformers'
##
##            # Store in Chroma
##            doc_id = f"{user}_{safe_name}_{int(time.time())}"
##            collection = client.get_or_create_collection(name="docs")
##            collection.add(
##                ids=[doc_id],
##                embeddings=[embedding],
##                documents=[file_text]
##            )
##            try:
##                client.persist()
##            except Exception:
##                pass
##
##            # Log user query
##            entry = {
##                "username": user,
##                "timestamp": datetime.now().isoformat(),
##                "query": f"[file_upload] {filename}",
##                "response": "[stored in RAG]",
##                "models": {"embed_model": emb_source}
##            }
##            log_user_query(user, entry)
##
##            # Notify frontend success
##            socketio.emit('state_update', {
##                'type': 'file_info',
##                'payload': {
##                    'message': f"Text '{filename}' processed for RAG",
##                    'doc_id': doc_id,
##                    'length': len(file_text),
##                    'rag_available': True,
##                    'embed_source': emb_source
##                },
##                'username': user
##            }, room=user)
##
##        except Exception as e:
##            socketio.emit('state_update', {
##                'type': 'file_error',
##                'payload': f'RAG processing failed: {e}',
##                'username': user
##            }, room=user)
##
##        return
##
##


# HTTP endpoints
@app.route('/main_responses/<username>', methods=['GET'])
def main_responses(username):
    path = os.path.join(TMP_DIR, f"{username}_main.json")
    if not os.path.exists(path):
        return jsonify([])
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return jsonify(json.load(f))
        except json.JSONDecodeError:
            return jsonify([])

@app.route('/user_logs/<username>', methods=["GET"])
def get_user_logs(username):
    logs = read_logs(username)
    print(f"[DEBUG] Loaded logs for {username}: {len(logs)} entries")
    return jsonify(logs)

@app.post("/upload_audio")
def upload_audio():
    os.makedirs(TMP_DIR, exist_ok=True)
    webm_path = os.path.join(TMP_DIR, "temp_audio.webm")
    wav_path = os.path.join(TMP_DIR, "temp_audio.wav")

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    request.files["audio"].save(webm_path)

    # convert with ffmpeg if available
    if os.system(f'ffmpeg -y -i "{webm_path}" -ar 16000 -ac 1 -f wav "{wav_path}"') != 0:
        return jsonify({"error": "ffmpeg failed"}), 500

    try:
        text = whisper_model.transcribe(wav_path).get("text", "").strip()
        cleaned = webui_listener.listen_text(text)
    except Exception:
        cleaned = ""

    for p in (webm_path, wav_path):
        try:
            os.remove(p)
        except:
            pass

    return jsonify({"transcript": cleaned})

@app.post('/upload_file')
def upload_file_http():
    # HTTP multipart upload alternative to socket-based upload
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    username = request.form.get('username', 'Tjaart')
    filename = secure_filename(file.filename)
    tmp_path = os.path.join(TMP_DIR, filename)
    file.save(tmp_path)

    # Try to detect type and if text, process into Chroma
    mime_type, _ = mimetypes.guess_type(filename)

    if mime_type and mime_type.startswith('image'):
        print(f"[HTTP FILE] Image uploaded: {filename} by {username}")
        socketio.emit('state_update', {'type': 'file_info', 'payload': f"Image '{filename}' received", 'username': username}, room=username)
        return jsonify({"status": "ok", "type": "image"})

    if mime_type and (mime_type.startswith('text') or filename.lower().endswith(('.txt', '.md', '.py'))):
        print(f"[HTTP FILE] Text uploaded: {filename} by {username}")
        try:
            with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_text = f.read()
        except Exception as e:
            return jsonify({"error": f"Could not read file: {e}"}), 500

        # Same RAG logic as socket handler
        try:
            client = None
            if chromadb is not None:
                try:
                    from chromadb.config import Settings as ChromaSettings
                    chroma_settings = ChromaSettings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=os.path.join(BASE_DIR, "chroma_db")
                    )
                    client = chromadb.Client(chroma_settings)
                except Exception:
                    try:
                        client = chromadb.Client()
                    except Exception:
                        client = None

            embedding = None
            emb_source = None
            try:
                if ollama is not None:
                    emb_resp = ollama.embeddings(model="mxbai-embed-large", prompt=file_text)
                    embedding = emb_resp.get('embedding')
                    emb_source = 'ollama'
            except Exception:
                try:
                    from sentence_transformers import SentenceTransformer
                    st_model = SentenceTransformer('all-MiniLM-L6-v2')
                    vect = st_model.encode(file_text)
                    embedding = vect.tolist() if hasattr(vect, 'tolist') else list(vect)
                    emb_source = 'sentence-transformers'
                except Exception:
                    embedding = None
                    emb_source = None

            doc_id = f"{username}_{filename}_{int(time.time())}"
            if embedding is not None and client is not None:
                try:
                    collection = client.get_or_create_collection(name="docs")
                    collection.add(ids=[doc_id], embeddings=[embedding], documents=[file_text])
                    try:
                        client.persist()
                    except Exception:
                        pass
                except Exception:
                    client = None

            if embedding is None or client is None:
                fallback_index = os.path.join(TMP_DIR, "local_rag_index.json")
                try:
                    if os.path.exists(fallback_index):
                        with open(fallback_index, 'r', encoding='utf-8') as fh:
                            idx = json.load(fh)
                    else:
                        idx = []
                except Exception:
                    idx = []
                idx_entry = {
                    'doc_id': doc_id,
                    'username': username,
                    'filename': filename,
                    'path': tmp_path,
                    'length': len(file_text),
                    'embedding_source': emb_source,
                    'timestamp': datetime.now().isoformat()
                }
                idx.append(idx_entry)
                try:
                    with open(fallback_index, 'w', encoding='utf-8') as fh:
                        json.dump(idx, fh, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            log_user_query(username, {"username": username, "timestamp": datetime.now().isoformat(), "query": f"[file_upload] {filename}", "response": "[stored in RAG]" if embedding is not None else "[saved - RAG unavailable]", "models": {"embed_model": emb_source or "none"}})

    ##        prompt=f"Using this data: {data}. Respond to this prompt: {prompt}",
            full_emb_prompt = f"Using this data: {response_text}. Respond to this prompt: {text}"
            socketio.emit('state_update', {
                'type': 'query',
    ##            'payload': {'query': text, 'response': response_text},
                'payload': full_emb_prompt,
                'username': user
            })

##            socketio.emit('state_update', {'type': 'file_info', 'payload': {'message': f"Text '{filename}' processed for RAG" if embedding is not None else f"Text '{filename}' saved (RAG unavailable)", 'doc_id': doc_id}, 'username': username}, room=username)
            return jsonify({"status": "ok", "type": "text", "doc_id": doc_id})
        except Exception as e:
            print(f"[ERROR] HTTP RAG failed: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok", "type": "unknown"})

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

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    context = ('cert.pem', 'key.pem') if os.path.exists('cert.pem') and os.path.exists('key.pem') else None
    socketio.run(app, host='0.0.0.0', port=5001, ssl_context=context)

