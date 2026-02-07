



import speech_recognition as sr
import asyncio
from edge_tts import Communicate
import playsound
import os

def recognize_afrikaans_speech():
    """Recognizes Afrikaans speech from the microphone and returns the transcription."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Sê iets in Afrikaans...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio, language="af-ZA")
        print(f"Jy het gesê: {text}")
        return text
    except sr.UnknownValueError:
        print("Kon nie Afrikaanse spraak herken nie.")
    except sr.RequestError as e:
        print(f"Fout met versoek aan Google-diens: {e}")
        return None

async def speak(text, voice="af-ZA-WillemNeural", style="cheerful"):
    """Speaks the given text using edge-tts (Microsoft TTS)."""
    communicate = Communicate(text=text, voice=voice)
    try:
        output_file = "output.mp3"
        await communicate.save(output_file)
        playsound(output_file)
        os.remove(output_file)  # Clean up after playback
    except Exception as e:
        print(f"Kon nie teks lees nie: {e}")

if __name__ == "__main__":
    recognized_text = recognize_afrikaans_speech()
    if recognized_text:
        asyncio.run(speak(recognized_text))











##
##
##
##def _chunk_text(text, max_tokens=5000, max_chars=4000):
##    """
##    Simple character-based chunker. Returns list of text chunks.
##    - max_chars default covers long docs — tune as needed for your embed model.
##    """
##    if not text:
##        return []
##
##    chunks = []
##    start = 0
##    L = len(text)
##    while start < L:
##        end = start + max_chars
##        # try to cut at newline or sentence boundary for nicer chunks
##        if end < L:
##            # find last newline before end
##            nl = text.rfind("\n", start, end)
##            if nl > start + 200:  # not too small
##                end = nl
##            else:
##                # fallback to last space
##                sp = text.rfind(" ", start, end)
##                if sp > start + 50:
##                    end = sp
##        chunks.append(text[start:end].strip())
##        start = end
##    return [c for c in chunks if c]
##
##
##def _get_embeddings_for_chunks(chunks):
##    """
##    Robust embedding helper tries:
##      1) ollama.embeddings (if ollama module available)
##      2) sentence-transformers (all-MiniLM-L6-v2)
##      3) openai (if OPENAI_API_KEY is set)
##    Returns: (embeddings_list, source_str)
##    Raises RuntimeError with a clear message if all fail.
##    """
##    if not chunks:
##        return [], 'none'
##
##    # 1) Ollama
##    try:
##        import ollama
##        embeddings = []
##        for c in chunks:
##            try:
##                # try common signatures
##                resp = None
##                try:
##                    resp = ollama.embeddings(model="mxbai-embed-large", input=c)
##                except Exception:
##                    resp = ollama.embeddings("mxbai-embed-large", c)
##                # extract vector
##                vec = None
##                if isinstance(resp, dict):
##                    vec = resp.get('embedding') or resp.get('embeddings') or (resp.get('data') and resp['data'][0].get('embedding'))
##                if vec is None and isinstance(resp, (list, tuple)) and len(resp) > 0:
##                    vec = resp[0]
##                if vec is None:
##                    raise RuntimeError("No vector in ollama response")
##                embeddings.append(list(vec))
##            except Exception as e:
##                raise RuntimeError(f"Ollama embedding failed for a chunk: {e}")
##        return embeddings, 'ollama-mxbai'
##    except Exception as e:
##        # don't hard-fail; record and try next
##        ollama_err = str(e)
##
##    # 2) sentence-transformers
##    try:
##        from sentence_transformers import SentenceTransformer
##        model = SentenceTransformer('all-MiniLM-L6-v2')  # auto-downloads model on first run
##        vects = model.encode(chunks, show_progress_bar=False)
##        embeddings = [v.tolist() if hasattr(v, 'tolist') else list(v) for v in vects]
##        return embeddings, 'sentence-transformers'
##    except Exception as e:
##        st_err = str(e)
##
##    # 3) OpenAI (if API key present)
##    try:
##        import openai
##        key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_KEY')
##        if key:
##            openai.api_key = key
##            embeddings = []
##            for c in chunks:
##                resp = openai.Embedding.create(model="text-embedding-3-small", input=c)
##                vec = resp['data'][0]['embedding']
##                embeddings.append(vec)
##            return embeddings, 'openai-text-embedding-3-small'
##    except Exception as e:
##        openai_err = str(e)
##
##    # If we get here, all providers failed — raise helpful error with details
##    msg = (
##        "No embedding provider available. Tried:\n"
##        f" - ollama: {ollama_err if 'ollama_err' in locals() else 'not attempted'}\n"
##        f" - sentence-transformers: {st_err if 'st_err' in locals() else 'not attempted'}\n"
##        f" - openai: {openai_err if 'openai_err' in locals() else 'not attempted'}\n\n"
##        "Resolve by either:\n"
##        "  * Installing sentence-transformers: pip install sentence-transformers\n"
##        "  * Installing/starting Ollama and ensuring mxbai-embed-large is available\n"
##        "  * Setting OPENAI_API_KEY and installing openai: pip install openai\n"
##    )
##    raise RuntimeError(msg)
##
##
##
##
##
##
##
##
##
##
##
##
##
##    def load_file(self):
##        """
##        Open file dialog, extract text, chunk, embed, store into local Chroma.
##        Uses get_chroma_client() for consistent client creation.
##        """
##        path = filedialog.askopenfilename(title="Select a File", filetypes=(("All files","*.*"),))
##        if not path:
##            return
##
##        self.current_user = getpass.getuser().capitalize()
##        filename = os.path.basename(path)
##        try:
##            with open(path, "rb") as f:
##                file_bytes = f.read()
##        except Exception as e:
##            self.log_message(f"Failed to read file: {e}")
##            return
##
##        # --- extract text (PDF or text)
##        mime_type, _ = mimetypes.guess_type(filename)
##        text = ""
##        try:
##            if (mime_type == "application/pdf") or filename.lower().endswith(".pdf"):
##                if PyPDF2 is None:
##                    self.log_message("PyPDF2 not installed — cannot extract PDF text.")
##                    text = ""
##                else:
##                    try:
##                        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
##                        pages = []
##                        for p in reader.pages:
##                            try:
##                                pt = p.extract_text() or ""
##                            except Exception:
##                                pt = ""
##                            pages.append(pt)
##                        text = "\n".join(pages).strip()
##                    except Exception as e:
##                        self.log_message(f"PDF extraction failed: {e}")
##                        text = ""
##            else:
##                detect = chardet.detect(file_bytes[:4096] if len(file_bytes) > 4096 else file_bytes)
##                enc = detect.get("encoding") or "utf-8"
##                try:
##                    text = file_bytes.decode(enc, errors="replace")
##                except Exception:
##                    text = file_bytes.decode("utf-8", errors="ignore")
##        except Exception as e:
##            self.log_message(f"File parsing failed: {e}\n{traceback.format_exc()}")
##            text = ""
##
##        if not text:
##            try:
##                text = file_bytes.decode("latin-1", errors="replace")
##            except Exception:
##                text = ""
##
##        # preview
##        try:
##            preview = text[:2000] + ("\n\n[truncated]" if len(text) > 2000 else "")
##            self.loaded_file_box.delete("1.0", "end")
##            self.loaded_file_box.insert("1.0", preview)
##        except Exception:
##            pass
##
##        # chunk
##        chunks = _chunk_text(text, max_chars=3000)
##        if not chunks:
##            self.log_message("No text extracted — nothing to add to RAG.")
##            return
##
##        # get embeddings
##        try:
##            embeddings, emb_source = _get_embeddings_for_chunks(chunks)
##        except Exception as e:
##            self.log_message(f"Embedding generation failed: {e}")
##            return
##
##        # get chroma client (shared helper)
##        client, chroma_dir = get_chroma_client(persist_directory=getattr(self, "BASE_DIR", os.getcwd()))
##        if client is None:
##            self.log_message("chromadb client unavailable — cannot store RAG.")
##            return
##
##        # ensure collection in a version tolerant way
##        try:
##            if hasattr(client, "get_or_create_collection"):
##                collection = client.get_or_create_collection(name="docs")
##            else:
##                try:
##                    collection = client.get_collection("docs")
##                except Exception:
##                    try:
##                        collection = client.create_collection("docs")
##                    except Exception:
##                        collection = None
##        except Exception as e:
##            self.log_message(f"Failed to get/create collection: {e}")
##            collection = None
##
##        if collection is None:
##            self.log_message("Could not obtain collection object; aborting store.")
##            return
##
##        # prepare ids/metadatas/docs
##        ids = []
##        docs = []
##        metas = []
##        ts = int(time.time())
##        for i, chunk in enumerate(chunks):
##            doc_id = f"{self.current_user}_{filename}_{ts}_{i}_{uuid.uuid4().hex[:6]}"
##            ids.append(doc_id)
##            docs.append(chunk)
##            metas.append({"source_filename": filename, "username": self.current_user, "chunk_index": i})
##
##        # add (try several arg shapes)
##        try:
##            try:
##                collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
##            except Exception:
##                # alternate signature / older versions
##                collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embeddings)
##        except Exception as e:
##            self.log_message(f"Failed to add chunks to collection: {e}\n{traceback.format_exc()}")
##            return
##
##        # attempt to persist (best-effort)
##        try:
##            if hasattr(client, "persist"):
##                client.persist()
##            elif hasattr(collection, "persist"):
##                collection.persist()
##        except Exception:
##            pass
##
##        # quick debug: read back what was stored (very useful to debug shapes)
##        try:
##            # try to list stored documents for debug
##            try:
##                stored = collection.get(include=['ids', 'documents', 'metadatas', 'embeddings'])
##            except Exception:
##                stored = collection.get()
##            self.log_message(f"DEBUG after add - collection.get() returned type={type(stored)}")
##            # try to extract first docs block
##            got_docs = []
##            if isinstance(stored, dict) and 'documents' in stored:
##                db = stored.get('documents', [])
##                if isinstance(db, list) and len(db) > 0:
##                    first = db[0]
##                    if isinstance(first, list):
##                        got_docs = first
##                    else:
##                        got_docs = db
##            elif isinstance(stored, dict) and 'documents' not in stored:
##                # sometimes stored is {'ids': [...], 'metadatas': [...], 'documents': [[]] } handled above
##                got_docs = stored.get('documents', [])
##            elif isinstance(stored, list):
##                got_docs = stored
##            # sanitize and log sample
##            simple = []
##            for d in got_docs[:5]:
##                try:
##                    if isinstance(d, bytes):
##                        d = d.decode('utf-8', errors='ignore')
##                except Exception:
##                    d = str(d)
##                simple.append((len(str(d)), (str(d)[:200] + '...') if len(str(d))>200 else str(d)))
##            self.log_message(f"DEBUG stored docs sample (count={len(got_docs)}): {simple}")
##        except Exception as e:
##            self.log_message(f"DEBUG readback failed: {e}")
##
##        # mark rag enabled and store ids for deletion
##        self.rag_enabled = True
##        existing = getattr(self, "uploaded_doc_ids", [])
##        existing.extend(ids)
##        self.uploaded_doc_ids = existing
##
##        self.log_message(f"Stored {len(ids)} chunks from '{filename}' into Chroma (emb_source={emb_source}).")
##        try:
##            self.loaded_file_box.delete("1.0", "end")
##            self.loaded_file_box.insert("1.0", f"[Uploaded {filename} to local RAG ({len(ids)} chunks).]")
##        except Exception:
##            pass
##
##
##
##
##
##
##
##    def get_chroma_client(persist_directory=None):
##        """
##        Return (client, path). This tries PersistentClient, then Client(), then Client(Settings).
##        It also disables telemetry where possible to avoid noisy exceptions.
##        """
##        import os, traceback
##        os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
##        CHROMA_PATH = persist_directory or os.path.join(os.getcwd(), "chroma_db")
##
##        try:
##            import chromadb
##        except Exception as e:
##            print(f"[get_chroma_client] chromadb import failed: {e}")
##            return None, CHROMA_PATH
##
##        # Try to disable telemetry on module
##        try:
##            if hasattr(chromadb, "telemetry_enabled"):
##                try:
##                    chromadb.telemetry_enabled = False
##                except Exception:
##                    pass
##            telemetry = getattr(chromadb, "telemetry", None)
##            if telemetry is not None:
##                for name in ("capture", "capture_event", "send_event", "capture_exception"):
##                    if hasattr(telemetry, name):
##                        try:
##                            setattr(telemetry, name, lambda *a, **kw: None)
##                        except Exception:
##                            pass
##        except Exception:
##            pass
##
##        # Try new PersistentClient
##        try:
##            PersistentClient = getattr(chromadb, "PersistentClient", None)
##            if PersistentClient is not None:
##                try:
##                    client = PersistentClient(path=CHROMA_PATH)
##                    print(f"[get_chroma_client] Using chromadb.PersistentClient(path={CHROMA_PATH})")
##                    globals()['chromadb'] = chromadb
##                    return client, CHROMA_PATH
##                except Exception as e:
##                    print(f"[get_chroma_client] PersistentClient init failed: {e}")
##
##        except Exception:
##            pass
##
##        # Try high-level Client()
##        try:
##            client = chromadb.Client()
##            print("[get_chroma_client] Using chromadb.Client() fallback")
##            globals()['chromadb'] = chromadb
##            return client, CHROMA_PATH
##        except Exception as e:
##            print(f"[get_chroma_client] chromadb.Client() failed: {e}")
##
##        # Legacy Settings fallback
##        try:
##            from chromadb.config import Settings as ChromaSettings
##            chroma_settings = ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PATH)
##            client = chromadb.Client(chroma_settings)
##            print(f"[get_chroma_client] Using legacy chromadb.Client(Settings) at {CHROMA_PATH}")
##            globals()['chromadb'] = chromadb
##            return client, CHROMA_PATH
##        except Exception as e:
##            print(f"[get_chroma_client] legacy Client(Settings) fallback failed: {e}")
##
##        return None, CHROMA_PATH
##
##
##
##    def retrieve_rag_context(self, query_text, top_k=3):
##        """
##        Retrieve most relevant RAG chunks for the given query.
##        Works with both new and legacy Chroma clients.
##        """
##        import traceback  # ✅ make sure traceback is imported for error logs
##        try:
##            from sentence_transformers import SentenceTransformer
##            import chromadb
##            import numpy as np
##            import os
##
##            base_dir = getattr(self, "BASE_DIR", os.getcwd())
##            chroma_dir = os.path.join(base_dir, "chroma_db")
##
##            # use your existing helper
##            client, used_path = get_chroma_client(persist_directory=chroma_dir)
##            if client is None:
##                self.log_message("[retrieve_rag_context] client unavailable")
##                return ""
##
##            # get collection safely
##            try:
##                if hasattr(client, "get_or_create_collection"):
##                    coll = client.get_or_create_collection(name="docs")
##                else:
##                    coll = client.get_collection("docs")
##            except Exception as e:
##                self.log_message(f"[retrieve_rag_context] failed to get collection: {e}")
##                return ""
##
##            # embed query
##            st_model = SentenceTransformer('all-MiniLM-L6-v2')
##            query_emb = st_model.encode([query_text])
##            query_emb = np.array(query_emb).tolist()
##
##            # query collection (handle API variants)
##            results = None
##            try:
##                results = coll.query(query_embeddings=query_emb, n_results=top_k)
##            except Exception:
##                try:
##                    results = coll.query(embeddings=query_emb, n_results=top_k)
##                except Exception as e2:
##                    self.log_message(f"[retrieve_rag_context] query failed: {e2}")
##                    return ""
##
##            # normalize documents
##            docs = []
##            if isinstance(results, dict):
##                docs_field = results.get('documents') or []
##                if isinstance(docs_field, list):
##                    if len(docs_field) > 0 and isinstance(docs_field[0], list):
##                        docs = docs_field[0]
##                    else:
##                        docs = docs_field
##
##            docs = [d for d in docs if isinstance(d, str) and d.strip()]
##            if not docs:
##                self.log_message("[retrieve_rag_context] no non-empty documents returned after normalization")
##                return ""
##
##            context = "\n\n".join(docs)
##            preview = (context[:400] + '...') if len(context) > 400 else context
##            self.log_message(f"[retrieve_rag_context] returning {len(docs)} docs, preview:\n{preview}")
##            return context
##
##        except Exception as e:
##            self.log_message(f"[retrieve_rag_context] unexpected error: {e}\n{traceback.format_exc()}")
##            return ""
##
##
##
##
##
##
##
##
##    def send_text(self):
##        self.current_user = "Itf"
##        text = self.input_text.get("1.0", tk.END).strip()
##        if text:
##            self.last_query = text
##            self.models = {
##                "thinkbot": self.thinkbot_model.get(),
##                "chatbot": self.chatbot_model.get(),
##                "vision": self.vision_model.get(),
##                "coding": self.coding_model.get()
##            }
##
##            # Retrieve RAG context first
##            context = self.retrieve_rag_context(text)
##            if context:
##                full_query = f"{text}\n\n[Context from knowledge base:]\n{context}"
##            else:
##                full_query = text
##
##            text_msg = {
##                'username': self.current_user,
##                'query': full_query,
##            }
##
##            self.log_query(f"Text to send from GUI: {text_msg}")
##            listen.add_text(text_msg)
##            self.input_text.delete("1.0", tk.END)
##        else:
##            self.log_message("No text entered.")
##
##
##
##
##
##
##
##
##
##
##
##
##
