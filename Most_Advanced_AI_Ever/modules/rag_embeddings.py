
# rag_embeddings.py
import re
import numpy as np
from typing import List
import ollama


class file_embeddings:
    def __init__(self, embed_model: str = "mxbai-embed-large"):
        self.embed_model = embed_model
        self.store = []  # stores {text, embedding}

    # -------------------------
    # Public: split + embed once
    # -------------------------
    def build_store(self, text: str, chunk_size: int = 500, overlap: int = 50):
        """
        Clean document, extract meaningful lines, embed chunks, and store in memory
        """
        self.store.clear()

        # clean and extract action lines
        clean_text = self._clean_text(text)
        extracted_text = self._extract_actions(clean_text)

        if not extracted_text.strip():
            extracted_text = clean_text  # fallback: use cleaned text if no action lines found

        # split into chunks
        chunks = self._split_text(extracted_text, chunk_size, overlap)

        for chunk in chunks:
            emb = self._embed_text(chunk)
            self.store.append({"text": chunk, "embedding": emb})

        print(f"[RAG] Stored {len(self.store)} cleaned chunks")

    # -------------------------
    # Public: query retrieval
    # -------------------------
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k most relevant chunks for a query.
        Works whether self.store contains dicts or plain strings.
        """
        if not self.store:
            return []

        query_emb = self._embed_text(query)
        similarities = []

        # Compute similarities
        for i, entry in enumerate(self.store):
            if isinstance(entry, dict):
                text = entry.get("text", "")
                emb = entry.get("embedding")
            else:
                text = entry
                emb = self._embed_text(text)  # re-embed if only string stored
                self.store[i] = {"text": text, "embedding": emb}

            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            similarities.append(sim)

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return the chunk texts
        return [
            self.store[i]["text"] if isinstance(self.store[i], dict) else self.store[i]
            for i in top_indices
        ]

    # -------------------------
    # Private: helpers
    # -------------------------
    def _clean_text(self, text: str) -> str:
        """
        Clean noisy invoice/document text before splitting into chunks
        """
        # normalize whitespace and newlines
        text = re.sub(r'\s+', ' ', text)

        # remove boilerplate fields
        text = re.sub(r'(E-?MAIL|TEL NO|FAX NO|ATT|DATE|INVOICE NO|TOTAL|SUB TOTAL).*?:.*?(?=\s[A-Z]|$)',
                      '', text, flags=re.IGNORECASE)

        # strip currency lines like R 2000.00
        text = re.sub(r'R\s*\d+[.,]?\d*', '', text)

        # remove repeated dashes, underscores, symbols
        text = re.sub(r'[-_=]{2,}', '', text)

        return text.strip()

    def _extract_actions(self, text: str) -> str:
        """
        Extract only action lines with verbs like repair, fix, install, assist, replace, etc.
        """
        action_lines = []
        for line in text.split("."):
            if re.search(r"\b(repair|fix|install|assist|put|replace|inspect|update|reseal)\b",
                         line, re.IGNORECASE):
                action_lines.append(line.strip())
        return "\n".join(action_lines)

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks (line/paragraph aware)
        """
        words = text.split()
        chunks, current = [], []

        for word in words:
            current.append(word)
            if sum(len(w) + 1 for w in current) >= chunk_size:
                chunks.append(" ".join(current).strip())
                current = current[-overlap:] if overlap > 0 else []

        if current:
            chunks.append(" ".join(current).strip())

        return chunks

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Run Ollama embedding model on text
        """
        response = ollama.embeddings(model=self.embed_model, prompt=text)
        return np.array(response["embedding"])


##### file_embeddings.py
##### Minimal fixes so your original code "just works" while changing as little as possible.
####
####import sys
####import math
####import queue
####import time
####import os
####import re
####
####import ollama
####import chromadb
######from pypdf import PdfReader
####import PyPDF2
####import fitz
####import cv2
######import docx
##### from spire.doc import *        # keep commented if not installed
##### from spire.doc.common import *
######from exceptions import PendingDeprecationWarning
####
####import requests
####import lxml.html
####from langchain.text_splitter import RecursiveCharacterTextSplitter
####
############################################################################################################################
####import socketio, requests, ssl, urllib3, getpass
####import requests as _requests  # avoid shadowing requests above
##### requests already imported above
####import time as _time
####
##### Do NOT auto-connect socket.io clients at import time (user asked minimal changes)
##### session = requests.Session()
##### session.verify = False
##### sio_mobile = socketio.Client(http_session=session, reconnection=True)
##### sio_mobile.connect('https://localhost:5000')
##### sio = socketio.Client()
##### sio.connect('http://localhost:5001')
####
##### Add modules dir (keep behaviour)
####sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))
##### from speech import speech
##### from arduino_com import arduino_com
####from memory import memory
##### from Repeat_Last import repeat
##### from listen import listen
##### from GUI import gui
####
####from model_selector import get_models_by_type
####
####My_Joined_Text = []
####
####
####class file_embeddings:
####    """
####    Very small, minimally-modified wrapper around your original code.
####    Use: instantiate and call ollama_rag(prompt, file_path=...)
####    """
####
####    # preserve your print so it's obvious module loaded
####    print("RAG Embeddings loaded")
####
####    def __init__(self):
####        # kept simple: no GUI required (you originally passed gui; backend can instantiate without GUI)
####        self.response_queue = queue.Queue()
####        # in-memory fallback store if chromadb is not available
####        self._memory_store = {"ids": [], "embeddings": [], "documents": []}
####        self.client = None
####        self.collection = None
####
####    # minimal file reader helpers:
####    def _read_text_file(self, path):
####        with open(path, "r", encoding="utf-8", errors="ignore") as f:
####            return f.read()
####
####    def _read_pdf_file(self, path):
####        if PyPDF2 is None:
####            return ""
####        text_parts = []
####        with open(path, "rb") as f:
####            reader = PyPDF2.PdfReader(f)
####            for i, page in enumerate(reader.pages):
####                try:
####                    txt = page.extract_text() or ""
####                except Exception:
####                    txt = ""
####                text_parts.append(f"\n\n[Page {i+1}]\n" + txt)
####        return "\n".join(text_parts)
####
####    def _split_text(self, text, chunk_size=3000, chunk_overlap=200):
####        if not text:
####            return []
####        try:
####            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
####                                                     chunk_overlap=chunk_overlap,
####                                                     length_function=len)
####            return splitter.split_text(text)
####        except Exception:
####            # fallback naive splitter
####            chunks = []
####            L = len(text)
####            start = 0
####            while start < L:
####                end = min(start + chunk_size, L)
####                chunks.append(text[start:end])
####                start = end - chunk_overlap if end - chunk_overlap > start else end
####            return chunks
####
####    def _ensure_chroma(self, persist_directory=None):
####        # lazy init chroma client and collection; tolerant to different chromadb versions
####        if chromadb is None:
####            # chromadb not installed; use in-memory fallback
####            return None
####
####        if self.client is not None and self.collection is not None:
####            return self.collection
####
####        try:
####            # try to initialize with settings if available
####            try:
####                from chromadb.config import Settings as ChromaSettings
####                settings = ChromaSettings(persist_directory=persist_directory)
####                self.client = chromadb.Client(settings=settings)
####            except Exception:
####                self.client = chromadb.Client()
####        except Exception as e:
####            print(f"[RAG] chromadb client init failed: {e}")
####            self.client = None
####            self.collection = None
####            return None
####
####        # obtain collection
####        try:
####            if hasattr(self.client, "get_or_create_collection"):
####                self.collection = self.client.get_or_create_collection(name="docs")
####            else:
####                # older/newer APIs: try create then fallback to get
####                try:
####                    self.collection = self.client.create_collection(name="docs")
####                except Exception:
####                    self.collection = self.client.get_collection(name="docs")
####        except Exception as e:
####            print(f"[RAG] get/create collection failed: {e}")
####            self.collection = None
####
####        return self.collection
####
####    def _add_to_store(self, docs, embed_model="mxbai-embed-large"):
####        # compute embeddings via ollama and add to chroma (if available) or in-memory fallback
####        if ollama is None:
####            raise RuntimeError("ollama is required for embeddings")
####
####        # ensure chroma available
####        coll = self._ensure_chroma()
####
####        in_memory = coll is None
####
####        for i, d in enumerate(docs):
####            try:
####                resp = ollama.embeddings(model=embed_model, prompt=d)
####            except Exception as e:
####                raise RuntimeError(f"ollama.embeddings failed: {e}")
####
####            if isinstance(resp, dict) and "embedding" in resp:
####                emb = resp["embedding"]
####            elif hasattr(resp, "embedding"):
####                emb = resp.embedding
####            else:
####                emb = resp
####
####            uid = f"{int(time.time()*1000)}_{i}"
####            if not in_memory:
####                try:
####                    coll.add(ids=[uid], embeddings=[emb], documents=[d])
####                except Exception as ee:
####                    print(f"[RAG] failed to add to chroma collection: {ee}; falling back to in-memory")
####                    in_memory = True
####                    self._memory_store["ids"].append(uid)
####                    self._memory_store["embeddings"].append(emb)
####                    self._memory_store["documents"].append(d)
####            else:
####                self._memory_store["ids"].append(uid)
####                self._memory_store["embeddings"].append(emb)
####                self._memory_store["documents"].append(d)
####
####    def _query_store(self, prompt, embed_model="mxbai-embed-large", n_results=1):
####        # create embedding for prompt
####        if ollama is None:
####            raise RuntimeError("ollama is required for query embedding")
####
####        try:
####            r = ollama.embeddings(model=embed_model, prompt=prompt)
####        except Exception as e:
####            raise RuntimeError(f"ollama.embeddings failed for query: {e}")
####
####        if isinstance(r, dict) and "embedding" in r:
####            q_emb = r["embedding"]
####        elif hasattr(r, "embedding"):
####            q_emb = r.embedding
####        else:
####            q_emb = r
####
####        # Try chroma query first
####        if self.collection is not None:
####            try:
####                results = self.collection.query(query_embeddings=[q_emb], n_results=n_results)
####                docs = results.get("documents", [[]])[0]
####                return docs
####            except Exception as e:
####                print(f"[RAG] chroma query failed: {e}; falling back to in-memory")
####
####        # In-memory brute force cosine
####        embs = self._memory_store.get("embeddings", [])
####        docs = self._memory_store.get("documents", [])
####        if not embs:
####            return []
####
####        def cosine(a, b):
####            dot = sum(x * y for x, y in zip(a, b))
####            na = math.sqrt(sum(x * x for x in a))
####            nb = math.sqrt(sum(x * x for x in b))
####            if na == 0 or nb == 0:
####                return 0.0
####            return dot / (na * nb)
####
####        scored = []
####        for e, d in zip(embs, docs):
####            try:
####                sc = cosine(q_emb, e)
####            except Exception:
####                sc = 0.0
####            scored.append((sc, d))
####        scored.sort(key=lambda x: x[0], reverse=True)
####        return [d for _, d in scored[:n_results]]
####
####    def _generate(self, prompt, context=None, gen_model="tinyllama"):
####        # call ollama.generate and return response text
####        if ollama is None:
####            raise RuntimeError("ollama is required for generation")
####        gen_prompt = f"Using this data: {context}\n\nRespond to this prompt: {prompt}" if context else prompt
####        try:
####            out = ollama.generate(model=gen_model, prompt=gen_prompt, stream=False)
####        except Exception as e:
####            raise RuntimeError(f"ollama.generate failed: {e}")
####
####        if isinstance(out, dict):
####            return out.get("response") or out.get("text") or str(out)
####        elif hasattr(out, "response"):
####            return out.response
####        else:
####            return str(out)
####
####    # ---- the method you will call from backend.py ----
####    def ollama_rag(self, prompt, file_path=None, embed_model="mxbai-embed-large", gen_model="tinyllama"):
####        """
####        Minimal: read file_path (if given), index text using ollama embeddings -> chroma (or in-memory),
####        query for prompt and generate a reply with selected model. Returns the generated reply string.
####        """
####        # step 1: read file if provided (default path if not provided keeps backward compatibility)
####        if file_path is None:
####            # keep your original default path if you relied on it
####            file_path = "d:/Python_Env/New_Virtual_Env/Alfred_Offline_New_GUI/2025_08_17_WEBUI_RAG_New/New_V2_Home_Head_Movement_Smoothing/modules/backend.py"
####
####        if not os.path.exists(file_path):
####            raise FileNotFoundError(f"File not found: {file_path}")
####
####        ext = os.path.splitext(file_path)[1].lower()
####        if ext == ".pdf":
####            text = self._read_pdf_file(file_path)
####        else:
####            text = self._read_text_file(file_path)
####
####        if not text or len(text.strip()) == 0:
####            raise RuntimeError("No text extracted from file")
####
####        # step 2: split into chunks and add to store
####        chunks = self._split_text(text)
####        if not chunks:
####            raise RuntimeError("No chunks produced from file text")
####
####        # ensure chroma client/collection (non-fatal if fails; will use in-memory)
####        try:
####            self._ensure_chroma()
####        except Exception:
####            pass
####
####        # add documents (will call ollama.embeddings)
####        self._add_to_store(chunks, embed_model=embed_model)
####
####        # step 3: query and generate
####        docs = self._query_store(prompt, embed_model=embed_model, n_results=1)
####        context = docs[0] if docs else None
####
####        reply = self._generate(prompt, context=context, gen_model=gen_model)
####        return reply
####
####
####if __name__ == "__main__":
####    import sys
####    fe = file_embeddings()
####    if len(sys.argv) > 1:
####        demo_fp = sys.argv[1]
####        prompt = sys.argv[2] if len(sys.argv) > 2 else "Summarize the document"
####        print("Running RAG on:", demo_fp, "with prompt:", prompt)
####        try:
####            reply = fe.ollama_rag(prompt=prompt, file_path=demo_fp)
####            print("\n--- AI reply ---\n")
####            print(reply)
####        except Exception as e:
####            print(f"[ERROR] RAG failed: {e}")
####    else:
####        print("file_embeddings module loaded. No demo path set.")
####        print("Usage: python file_embeddings.py /path/to/file.pdf \"Your prompt here\"")
####


### end of file: no interactive main() at import time
##if __name__ == "__main__":
##    # small local demo if run directly (won't execute when imported)
##    fe = file_embeddings()
##    demo_fp = None  # set a path if you want to demo
##    if demo_fp:
##        print("Demo reply:", fe.ollama_rag("Summarize the file", file_path=demo_fp))
##    else:
##        print("file_embeddings module loaded. No demo path set.")
##
##




##############
##############
##############import ollama
##############import chromadb
################from pypdf import PdfReader
##############import PyPDF2
##############import fitz
##############import cv2
################import docx
##############from spire.doc import *
##############from spire.doc.common import *
################from exceptions import PendingDeprecationWarning
##############import os
##############import re
##############
##############import requests
##############import lxml.html
##############from langchain.text_splitter import RecursiveCharacterTextSplitter
##############
######################################################################################################################################
##############import socketio, requests, ssl, urllib3, getpass
##############import os
##############import requests
##############import time
##############import queue
##############
##############
##############
##############
################# Requests session (no SSL verify)
################session = requests.Session()
################session.verify = False
################
################# Socket.IO clients
################sio_mobile = socketio.Client(http_session=session, reconnection=True)
################sio_mobile.connect('https://localhost:5000')
################sio = socketio.Client()
################sio.connect('http://localhost:5001')
##############
############### Add modules dir
##############sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))
################from speech import speech
################from arduino_com import arduino_com
##############from memory import memory
################from Repeat_Last import repeat
################from listen import listen
################from GUI import gui
##############
##############from model_selector import get_models_by_type
##############
##############
##############
##############
##############
##############My_Joined_Text = []
##############
##############
##############class file_embeddings:
##############
##############    print("RAG Embeddings ended")
##############
##############    
################    def __init__(self, gui):
##############    def __init__(self):
##############        
################        if not gui:
################            raise ValueError("GUI instance must be provided to AI_Assistant!")
################        self.gui = gui
##############        self.response_queue = queue.Queue()
##############
##############    ##__________________________________________________________________________________
##############    ##__________________________________________________________________________________
##############
################    def ollama_rag(self, AlfredQueryOffline):
##############    def ollama_rag(self, prompt):
##############
##############        
##############
##############        ########################################################################################################################
##############        #     FOR WORD .doc FILES
##############
##############        ### Create a Document object
##############        ##document = Document()
##############        ### Load a Word document
##############        ##document.LoadFromFile("C:\\Python_Env\\AYEN\\SABS_SANS10142_ED1.docx")
##############        ##document.LoadFromFile("C:\\Python_Env\\AYEN\\Regulation_Very_Short.doc")
##############        ##
##############        ### Extract the text of the document
##############        ##document_text = document.GetText()
##############        ##print (f"document_text: {document_text}")
##############        ##
##############        ##My_Joined_Text = document_text
##############        ##My_Joined_Text = My_Joined_Text
##############        ##print(f"My_Joined_Text : {My_Joined_Text}")
##############        ##
##############        ##
##############        ##documents = [My_Joined_Text]
##############        ##print(f"documents : {documents}")
##############
##############
##############        ########################################################################################################################
##############        #     FOR PDF 1 x FILES
##############
##############
##############        ####My_Document_Pdf = open('C:/Python_Env/AYEN/SABS_SANS10142_ED1.pdf', 'rb') 
##############        ####My_Document_Pdf = open('C:/Python_Env/AYEN/Electrical_Installation_Regulations.pdf', 'rb')
##############        ####My_Document_Pdf = open('D:/Python_Env/AYEN/Ayen_Electrical/Invoices/Client Copy Pdf/Adrie_Woolfies_Full_91_257.pdf', 'rb')
##############        ####My_Document_Pdf = open('D:/Downloads/Afrikaans Graad 9/Antjie Somers en My Broer se Kraai/AntjieSomerswhereareyoufrom.pdf', 'rb')
##############        ##My_Document_Pdf = open('D:/Downloads/Afrikaans Graad 9/Antjie Somers en My Broer se Kraai/AntjieSomerswhereareyoufrom.pdf', 'rb')
##############        ##
##############        ##pdfReader = PyPDF2.PdfReader(My_Document_Pdf)
##############        ##Num_Pages = len(pdfReader.pages)
##############        ##
##############        ##print(f"Num_Pages : {Num_Pages}")
##############        ##
##############        ##for i in range(Num_Pages):
##############        ##  Page = pdfReader.pages[i]
##############        ##  My_Extracted_Text = Page.extract_text()
##############        ##  print(f"Page Number : {i}")
##############        ####  print(f"My_Extracted_Text : {My_Extracted_Text}")
##############        ##
##############        ##  My_Joined_Text.append(My_Extracted_Text)
##############        ####  print(f"My_Joined_Text : {My_Joined_Text}")
##############        ##
##############        ##documents = My_Joined_Text
##############        ##print(f"documents : {documents}")
##############
##############
##############
##############        ########################################################################################################################
##############        #     FOR MORE 1 PDF FILES
##############
##############        ####My_Document_Pdf = open('C:/Python_Env/AYEN/Testing/SABS_SANS10142_ED1.pdf', 'rb')
##############        ####My_Document_Pdf = open('C:/Python_Env/AYEN/Testing/Electrical_Installation_Regulations_Shorter.pdf', 'rb')
##############        ##My_Document_Pdf = open('D:/Python_Env/AYEN/Ayen_Electrical/Invoices/Client Copy Pdf/Adrie_Woolfies_Full_91_257.pdf', 'rb')
##############        ##
##############        ##pdfReader = PyPDF2.PdfReader(My_Document_Pdf)
##############        ##Num_Pages = len(pdfReader.pages)
##############        ##
##############        ##print(f"Num_Pages : {Num_Pages}")
##############        ##
##############        ##for i in range(Num_Pages):
##############        ##    Page = pdfReader.pages[i]
##############        ##    My_Extracted_Text = Page.extract_text()
##############        ##    print(f"Page Number : {i}")
##############        ##    print(f"My_Extracted_Text : {My_Extracted_Text}")
##############        ##
##############        ##    My_Joined_Text.append(My_Extracted_Text)
##############        ##    print(f"My_Joined_Text : {My_Joined_Text}")
##############        ##
##############        ##
##############        ##    # Split text into chunks
##############        ##    text_splitter = RecursiveCharacterTextSplitter(
##############        ##        chunk_size=10000,
##############        ##        chunk_overlap=200,
##############        ##        length_function=len
##############        ##    )
##############        ##    chunks = text_splitter.split_text(text=str(My_Joined_Text))
##############        ##
##############        ##
##############        ##
##############        ##    documents = chunks
##############        ##    print(f"documents : {documents}")
##############
##############
##############
##############
##############
##############
##############
##############        ########################################################################################################################
##############        #     FOR TEXT .txt FILES
##############
##############        # Create a Document object
##############        document = Document()
##############
##############        ### Load a TEXT document
##############        ####document.LoadFromFile("C:\\Python_Env\\AYEN\\SABS_SANS10142_ED1.txt")
##############        ####document.LoadFromFile("C:\\Python_Env\\AYEN\\Test_Regulations_Test.txt")
##############        ##document.LoadFromFile("d:/Python_Env/New_Virtual_Env/Alfred_Offline_New_GUI/2025_08_17_WEBUI_RAG_New/New_V2_Home_Head_Movement_Smoothing/modules/backend.py")
##############        ##
##############        ### Extract the text of the document
##############        ##document_text = document.GetText()
##############        ##print (f"document_text: {document_text}")
##############        ##
##############        ##My_Joined_Text = document_text
##############        ##My_Joined_Text = My_Joined_Text
##############        ##print(f"My_Joined_Text : {My_Joined_Text}")
##############        ##
##############        ##
##############        ##documents = [My_Joined_Text]
##############        ##print(f"documents : {documents}")
##############
##############        ########################################################################################################################
##############        #     FOR TEXT .PY FILES
##############
##############        # Path to your .py file
##############        file_path = "d:/Python_Env/New_Virtual_Env/Alfred_Offline_New_GUI/2025_08_17_WEBUI_RAG_New/New_V2_Home_Head_Movement_Smoothing/modules/backend.py"
##############
##############        # Open and read the file
##############        with open(file_path, "r", encoding="utf-8") as f:
##############            document_text = f.read()
##############
##############        print(f"document_text: {document_text[:500]}...")  # print first 500 chars for brevity
##############
##############        # Join text (here it's already one string, so this does nothing)
##############        My_Joined_Text = document_text
##############        print(f"My_Joined_Text: {My_Joined_Text[:500]}...")  # again first 500 chars
##############
##############        # Put it into a list
##############        documents = [My_Joined_Text]
##############        print(f"documents: {documents[:1]}")  # show only first element
##############
##############        ########################################################################################################################
##############        #     FOR WEB PAGE SCRAPING (URL's)
##############
##############        ##My_URL = ''
##############        ##
##############        ### Create a Document object
##############        ##document = Document()
##############        ##
##############        ##dom = lxml.html.fromstring(requests.get('http/www.?.com').content)
##############        ##page_list = [x for x in dom.xpath('//td/text()')]
##############        ##print(f"page_list: {page_list}")
##############
##############
##############        #######################################################################################################################
##############        #         GETTING EMBEDDINGS FOR LLM's
##############
##############        client = chromadb.Client()
##############        ##client = chromadb.HttpClient(host='localhost', port=8000)
##############
##############        collection = client.create_collection(name="docs")
##############        ##collection = client.get_or_create_collection(name="docs") # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
##############
##############        # store each document in a vector embedding database
##############        for i, d in enumerate(documents):
##############          response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
##############          embedding = response["embedding"]
##############          collection.add(
##############            ids=[str(i)],
##############            embeddings=[embedding],
##############            documents=[d]
##############          )
##############
##############          main() 
##############
##############
##############    ########################################################################################################################
##############
##############    def main():
##############
################      Input_From_User = input("ASK AI ANYTHING : ")
##############        Input_From_User = input("ASK AI ANYTHING : ")
##############        ##  cv2.waitkey(0)
##############
##############        prompt = Input_From_User
##############
##############        # generate an embedding for the prompt and retrieve the most relevant doc
##############        response = ollama.embeddings(
##############        prompt=prompt,
##############        model="mxbai-embed-large"
##############        )
##############        results = collection.query(
##############        query_embeddings=[response["embedding"]],
##############        n_results=1
##############        )
##############        data = results['documents'][0][0]
##############
##############        # generate a response combining the prompt and data we retrieved in step 2
##############        output = ollama.generate(
##############        model="tinyllama",
##############        prompt=f"Using this data: {data}. Respond to this prompt: {prompt}",
##############        stream=False
##############        )
##############
##############        print(output['response'])
##############
##############        ollama_rag(prompt)
##############
##############
##############if __name__ == "__main__":
##############
##############
##############    print("RAG Embeddings ended")
##############
##############    embed = file_embeddings
##############    ##    main()
##############    embed()
##############



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



############    # --- LOAD FILE ---
############    
############   # inside your socketio event handler function, e.g.:
############    if t == 'load_file':
############        my_file_content = ""
############
############        # Validate payload
############        if not isinstance(payload, dict):
############            print(f"[ERROR] load_file called with invalid payload: {payload!r}")
############            socketio.emit('state_update', {
############                'type': 'file_error',
############                'payload': "Invalid or missing payload for load_file",
############                'username': user
############            }, room=user)
############            return
############
############        filename = payload.get('filename')
############        filedata_b64 = payload.get('filedata')
############        mime_hint = payload.get('mime') or payload.get('type')
############
############        print(f"[DEBUG] load_file payload keys: {list(payload.keys())}")
############
############        if not filename or filedata_b64 is None:
############            print(f"[ERROR] load_file missing filename or filedata. filename={filename!r}, filedata_present={filedata_b64 is not None}")
############            socketio.emit('state_update', {
############                'type': 'file_error',
############                'payload': {
############                    'message': 'Missing filename or filedata',
############                    'received_keys': list(payload.keys())
############                },
############                'username': user
############            }, room=user)
############            return
############
############        # Decode base64 to bytes
############        try:
############            file_bytes = base64.b64decode(filedata_b64)
############        except Exception as e:
############            print(f"[ERROR] base64 decode failed: {e}; payload_snippet={str(filedata_b64)[:80]}")
############            socketio.emit('state_update', {'type': 'file_error', 'payload': f'base64 decode failed: {e}', 'username': user}, room=user)
############            return
############
############        # Save raw bytes
############        safe_name = secure_filename(filename)
############        tmp_path = os.path.join(TMP_DIR, safe_name)
############        try:
############            with open(tmp_path, 'wb') as f:
############                f.write(file_bytes)
############            print(f"[INFO] Saved uploaded file to {tmp_path}")
############        except Exception as e:
############            print(f"[ERROR] Failed to save uploaded file: {e}")
############            return
############
############        # Determine mime type
############        mime_type, _ = mimetypes.guess_type(filename)
############        if not mime_type and mime_hint:
############            mime_type = mime_hint
############
############        # Extract text from PDF if applicable
############        if mime_type == 'application/pdf' or filename.lower().endswith('.pdf'):
############            try:
############                extracted_text = ""
############                with pdfplumber.open(tmp_path) as pdf:
############                    for page in pdf.pages:
############                        text = page.extract_text()
############                        if text:
############                            extracted_text += text + "\n"
############                my_file_content = extracted_text.encode('utf-8', errors='replace').decode('utf-8')
############                print(f"[INFO] Extracted text from PDF ({len(my_file_content)} chars)")
############            except Exception as e:
############                print(f"[ERROR] Failed to extract text from PDF: {e}")
############                my_file_content = ""
############
############        # If not PDF or extraction failed, try text file fallback
############        elif mime_type and (mime_type.startswith('text') or filename.lower().endswith(('.txt', '.md', '.py'))):
############            try:
############                with open(tmp_path, 'rb') as f:
############                    raw_data = f.read()
############                    import chardet
############                    detected_encoding = chardet.detect(raw_data).get('encoding') or 'utf-8'
############                    print(f"[INFO] Detected file encoding: {detected_encoding}")
############                    decoded_content = raw_data.decode(detected_encoding, errors='replace')
############                    my_file_content = decoded_content.encode('utf-8').decode('utf-8')
############            except Exception as e:
############                print(f"[ERROR] Failed to decode text file: {e} \n lets try and se if it is PYTHON \n")
##############                my_file_content = ""
############
############                # Path to your .py file
############                file_path = "d:/Python_Env/New_Virtual_Env/Alfred_Offline_New_GUI/2025_08_17_WEBUI_RAG_New/New_V2_Home_Head_Movement_Smoothing/modules/backend.py"
############
############                # Open and read the file
############                with open(file_path, "r", encoding="utf-8") as f:
############                    document_text = f.read()
############
############                print(f"document_text: {document_text[:500]}...")  # print first 500 chars for brevity
############
############                # Join text (here it's already one string, so this does nothing)
############                My_Joined_Text = document_text
############                print(f"My_Joined_Text: {My_Joined_Text[:500]}...")  # again first 500 chars
############
############                # Put it into a list
############                documents = [My_Joined_Text]
############                print(f"documents: {documents[:1]}")  # show only first element
############
############                my_file_content =  documents
############                print(f"my_file_content: {my_file_content}")  # show only first element
############
############
############
############        # Fallback decode raw bytes if still no content
############        if not my_file_content:
############            try:
############                my_file_content = file_bytes.decode('utf-8', errors='ignore')
############            except Exception:
############                my_file_content = file_bytes.decode('latin-1', errors='ignore')
############
############        # Now you can use my_file_content for embedding / RAG as before
############        file_text = my_file_content
############
############        # === Your existing RAG embedding & storage code here ===
############        try:
############            # Initialize Chroma
############            try:
############                from chromadb.config import Settings as ChromaSettings
############                chroma_settings = ChromaSettings(
############                    chroma_db_impl="duckdb+parquet",
############                    persist_directory=os.path.join(BASE_DIR, "chroma_db")
############                )
############                client = chromadb.Client(chroma_settings)
############                print("[RAG] Connected to embedded Chroma (duckdb+parquet).")
############            except Exception as e_init:
############                print(f"[RAG] Embedded Chroma init failed: {e_init}. Trying default chromadb.Client()...")
############                try:
############                    client = chromadb.Client()
############                    print("[RAG] Connected to Chroma via default client.")
############                except Exception as e_client:
############                    print(f"[RAG] chromadb.Client() failed: {e_client}. Will fallback to disk storage.")
############                    client = None
############
############            # Create embedding
############            embedding = None
############            emb_source = None
############            try:
############                if ollama is None:
############                    raise RuntimeError('ollama package not available')
############                emb_resp = ollama.embeddings(model="mxbai-embed-large", prompt=file_text)
############                embedding = emb_resp.get('embedding')
############                emb_source = 'ollama'
############                if embedding is None:
############                    raise RuntimeError('Ollama returned no embedding')
############                print("[RAG] Got embedding from Ollama.")
############            except Exception as e_oll:
############                print(f"[RAG] Ollama embeddings failed: {e_oll} — falling back to sentence-transformers if available.")
############                try:
############                    from sentence_transformers import SentenceTransformer
############                    st_model = SentenceTransformer('all-MiniLM-L6-v2')
############                    vect = st_model.encode(file_text)
############                    embedding = vect.tolist() if hasattr(vect, "tolist") else list(vect)
############                    emb_source = 'sentence-transformers'
############                    print("[RAG] Got embedding from sentence-transformers fallback.")
############                except Exception as e_st:
############                    print(f"[RAG] sentence-transformers fallback failed: {e_st}")
############                    embedding = None
############                    emb_source = None
############
############            # Store in Chroma or fallback
############            doc_id = f"{user}_{safe_name}_{int(time.time())}"
############            if embedding is not None and client is not None:
############                try:
############                    collection = client.get_or_create_collection(name="docs")
############                    collection.add(
############                        ids=[doc_id],
############                        embeddings=[embedding],
############                        documents=[file_text]
############                    )
############                    try:
############                        client.persist()
############                    except Exception:
############                        pass
############                    print(f"[RAG] Stored document in Chroma: id={doc_id} len={len(file_text)}")
############                except Exception as e_store:
############                    print(f"[RAG] Failed to store in Chroma: {e_store}")
############                    client = None
############
############            if embedding is None or client is None:
############                fallback_index = os.path.join(TMP_DIR, "local_rag_index.json")
############                try:
############                    if os.path.exists(fallback_index):
############                        with open(fallback_index, 'r', encoding='utf-8') as fh:
############                            idx = json.load(fh)
############                    else:
############                        idx = []
############                except Exception:
############                    idx = []
############                idx_entry = {
############                    'doc_id': doc_id,
############                    'username': user,
############                    'filename': filename,
############                    'path': tmp_path,
############                    'length': len(file_text),
############                    'embedding_source': emb_source,
############                    'timestamp': datetime.now().isoformat()
############                }
############                idx.append(idx_entry)
############                try:
############                    with open(fallback_index, 'w', encoding='utf-8') as fh:
############                        json.dump(idx, fh, ensure_ascii=False, indent=2)
############                    print(f"[RAG] Saved document metadata to local_rag_index.json id={doc_id}")
############                except Exception as e_idx:
############                    print(f"[RAG] Failed to write fallback index: {e_idx}")
############
############            # Log query history
############            entry = {
############                "username": user,
############                "timestamp": datetime.now().isoformat(),
############                "query": f"[file_upload] {filename}",
############                "response": "[stored in RAG]" if embedding is not None else "[saved - RAG unavailable]",
############                "models": {"embed_model": emb_source or "none"}
############            }
############            log_user_query(user, entry)
############
############            # Notify frontend
############            socketio.emit('state_update', {
############                'type': 'file_info',
############                'payload': {
############                    'message': f"Text '{filename}' processed for RAG" if embedding is not None else f"Text '{filename}' saved (RAG unavailable)",
############                    'doc_id': doc_id,
############                    'length': len(file_text),
############                    'rag_available': embedding is not None and client is not None,
############                    'embed_source': emb_source
############                },
############                'username': user
############            }, room=user)
############
############        except Exception as e:
############            print(f"[ERROR] RAG processing failed (unexpected): {e}")
############            socketio.emit('state_update', {'type': 'file_error', 'payload': str(e), 'username': user}, room=user)
############        return
















