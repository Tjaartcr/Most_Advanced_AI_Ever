


# listen_module.py

import re
import os
import sys
import time
import json
import queue
import collections
import re
import string

import numpy as np
import sounddevice as sd
import webrtcvad
import vosk
import whisper
import serial

# Ensure these imports resolve in your project:
# - speech should provide speech.AlfredSpeak(text)
# - config should define VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
from speechWEBUI import WEBUISpeechModule
from WEBUI_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH

# Increase recursion limit if needed (as in your original code)
sys.setrecursionlimit(25000000)

print("speech module initialization start")



# 🧠 Convert digits after x into superscript form for display
def convert_to_superscript(expression):
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }

    def replace_power(match):
        base = match.group(1)
        power = ''.join(superscript_map.get(d, d) for d in match.group(2))
        return base + power

    return re.sub(r'([a-zA-Z])(\d+)', replace_power, expression)


# 🎙️ Fix all superscripts and minus signs for natural speech
def prepare_text_for_tts(text):
    # Superscript to natural speech
    superscript_to_words = {
        '⁰': ' to the power of 0',
        '¹': ' to the power of 1',
        '²': ' squared',
        '³': ' cubed',
        '⁴': ' to the power of 4',
        '⁵': ' to the power of 5',
        '⁶': ' to the power of 6',
        '⁷': ' to the power of 7',
        '⁸': ' to the power of 8',
        '⁹': ' to the power of 9',
    }

    # Replace superscript characters
    for sup, spoken in superscript_to_words.items():
        text = text.replace(sup, spoken)

    # Handle math phrases naturally
    text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 minus \2', text)        # 8 - 2 → 8 minus 2
    text = re.sub(r'(?<![\w])-(\d+)', r'negative \1', text)          # -8 → negative 8
    text = re.sub(r'(?<![\w])-([a-zA-Z])', r'negative \1', text)     # -x → negative x

    # Clean up any extra spaces
    text = re.sub(r'\s+', ' ', text)

    New_Speech_Text = text
    New_Speech_Text = New_Speech_Text.replace(" negative "," -")
    New_Speech_Text = New_Speech_Text.replace(" positive "," +")



    return New_Speech_Text.strip()
##    return text.strip()



##def add_text(self, text):
##    """Externally enqueue a text query to be processed next."""
##    if text:
##        self.text_queue.put(text)
##        print(f"Text added to queue: {text}")




def is_math_expression(text):
    """
    Determine if the input text looks like a math-related query.
    This can be expanded as needed.
    """
##    return bool(re.search(r'\d+[a-zA-Z]?\d*|[\+\-\*/\^=]', text))
    return bool(re.search(r'\d+[a-zA-Z]?\d*|[+\-*/^=]|\b(plus|minus|times|square|multiply|divide)\b', text))
    
class WEBUIListenModule:
    def __init__(self):
        """Initialize Vosk & Whisper Models, Bluetooth, and input queues."""
        # Vosk setup
        if not os.path.exists(VOSK_MODEL_PATH):
            raise FileNotFoundError(f"❌ Vosk model not found! Expected path: {VOSK_MODEL_PATH}")
        print(f"✅ Loading Vosk model from {VOSK_MODEL_PATH}...")
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)

        # Whisper setup
        print("✅ Loading Whisper model (this may take a moment)...")
        # You can choose "tiny.en", "base.en", etc. depending on your needs.
        self.whisper_model = whisper.load_model("base.en")

        # Audio settings
        self.samplerate = 16000
        # We no longer use a fixed-duration recording; we use VAD to detect end of speech.

        # Queues for external text injections
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()

        # Bluetooth setup
        self.bluetooth = None
        self.timeout_set = 3.05
        try:
            self.bluetooth = serial.Serial(SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH, timeout=self.timeout_set)
            print(f"✅ Bluetooth is now Connected to PORT {SERIAL_PORT_BLUETOOTH}")
        except serial.SerialException:
            print("❌ OH NO !!! Bluetooth connection failed.")

        # Recognizer flags
        self.use_whisper_listen = False
        self.use_vosk_listen = True
        self.mobile_speech_enabled = False


    def add_text(self, text):
        """Externally enqueue a text query to be processed next."""
        if text:
            self.text_queue.put(text)
            print(f"Text added to queue: {text}")

    # 🧠 Alfred input handler
    def listen_text(self, text):
        """Process text input directly, with math formatting if needed."""
        if not text:
            print("No text received.")
            return None

        else:
            return text
        
        print(f"Raw input received: {text}")

########        if is_math_expression(text):
########            print("Math detected ✅")
##########            speech.AlfredSpeak("Hmmm Thank you very much for this mathematic equation.")
##########            speech.AlfredSpeak("I am so excited, let's get started right away")
########
########            # Format display and voice versions
########            display_text = convert_to_superscript(text)
########            print(f"Superscripted for logic/display: {display_text}")
########
########            New_Display_Text = display_text
########            New_Display_Text = New_Display_Text.replace(" negative "," -")
########            New_Display_Text = New_Display_Text.replace(" positive "," +")
########
########
########            spoken_text = prepare_text_for_tts(display_text)
########            print(f"TTS output: {spoken_text}")
########
########            # Alfred speaks with math-awareness
##########            speech.AlfredSpeak(spoken_text)
########
########            # Return superscripted version for internal use
##########            return display_text
########            return New_Display_Text
########
########        else:
##########            speech.AlfredSpeak("Hmmmm, this is non mathematical.")
########            print("Standard (non-math) input detected")
########            return text

    def set_mobile_speech(self, enabled: bool):
        """Enable or disable sending speech output to a mobile device over Bluetooth."""
        self.mobile_speech_enabled = enabled
        print(f"mobile_speech_enabled : {self.mobile_speech_enabled}")
##        if enabled:
##            speech.AlfredSpeak(
##            "Remember to start the home automation application on your Android device and connect via Bluetooth.")

    def set_listen_whisper(self, enabled: bool):
        """
        Switch between Whisper-based listening (with VAD) and Vosk-based listening.
        Call with True to use Whisper, False to use Vosk.
        """
        self.use_whisper_listen = enabled
        self.use_vosk_listen = not enabled
        mode = "Whisper" if enabled else "Vosk"
        print(f"🎙️ Speech Recognizer set to: {mode}")

    def send_bluetooth(self, message: str):
        """Send a string message over Bluetooth serial if enabled."""
        if self.mobile_speech_enabled and self.bluetooth:
            try:
                self.bluetooth.write(message.encode('utf-8'))
                print(f"📡 Sent to Device via Bluetooth: {message}")
            except serial.SerialException as e:
                print(f"❌ Bluetooth error: {e}")
        else:
            print("Bluetooth speech output not enabled or no device connected.")

    def callback(self, indata, frames, time_info, status):
        """Callback for sounddevice.RawInputStream (used by Vosk)."""
        if status:
            print(f"❌ Audio callback error: {status}")
        # Put raw bytes into queue for Vosk
        self.audio_queue.put(bytes(indata))

    def listen_vosk(self):
        """
        Listen with Vosk until it accepts a waveform and returns a result.
        Passes through listen_text for math detection and formatting.
        """
        recognizer = vosk.KaldiRecognizer(self.vosk_model, self.samplerate)
        speech.AlfredSpeak("Listening Vosk...")
        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.callback
        ):
            print("🎤 Listening via Vosk... Speak now!")
            
            while True:
                data = self.audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    print(f"📝 Vosk recognized: {text}")
                    # ✅ Let listen_text handle further formatting
                    return self.listen_text(text) or None

    def record_until_silence(self,
                             frame_duration_ms: int = 30,
                             vad_mode: int = 3,
                             silence_duration_ms: int = 3000,
                             max_record_duration_s: int = 50) -> np.ndarray:
        """
        Record from microphone until end of speech detected by WebRTC VAD.
        Returns a float32 numpy array in [-1,1], sampled at self.samplerate.
        If no speech detected within max_record_duration_s, returns empty array.
        """
        vad = webrtcvad.Vad(vad_mode)
        samplerate = self.samplerate
        # frame_size in samples
        frame_size = int(samplerate * frame_duration_ms / 1000)
        bytes_per_frame = frame_size * 2  # because int16

        ring_buffer = collections.deque(maxlen=int(silence_duration_ms / frame_duration_ms))
        voiced_frames = []
        got_speech = False
        start_time = time.time()

        try:
            with sd.RawInputStream(samplerate=samplerate,
                                   blocksize=frame_size,
                                   dtype='int16',
                                   channels=1) as stream:
                
    ##            speech.AlfredSpeak("Listening whisper...")
                
                print("🎤 Listening (VAD) via microphone... Speak now!")
                while True:
                    # Safety stop if too long
                    if time.time() - start_time > max_record_duration_s:
                        print("⚠️ Max record duration reached, stopping recording.")
                        break
                    try:
                        frame_bytes, overflowed = stream.read(frame_size)
                    except Exception as e:
                        print(f"❌ Error reading audio frame: {e}")
                        break
                    if overflowed:
                        print("⚠️ Audio buffer overflowed")
                    # If incomplete frame, skip
                    if len(frame_bytes) < bytes_per_frame:
                        continue
                    # Check VAD
                    try:
                        is_speech = vad.is_speech(frame_bytes, sample_rate=samplerate)
                    except Exception as e:
                        print(f"⚠️ VAD error: {e}")
                        is_speech = False

                    if not got_speech:
                        # We haven't started speech yet: wait until speech detected
                        if is_speech:
                            got_speech = True
                            print("🔊 Speech started.")
                            voiced_frames.append(frame_bytes)
                            ring_buffer.clear()
                            ring_buffer.append(True)
                        else:
                            # still silence
                            continue
                    else:
                        # Already in speech: collect frames, track speech/silence
                        voiced_frames.append(frame_bytes)
                        ring_buffer.append(is_speech)
                        # If ring buffer full and all False => end of speech
                        if len(ring_buffer) == ring_buffer.maxlen and not any(ring_buffer):
                            print("🔇 Detected end of speech.")
                            break
                        # else continue collecting
        except Exception as e:
            print(f"❌ Exception opening RawInputStream: {e}")
            # Return any collected frames so far (or empty)
        # After loop
        if not voiced_frames:
            print("⚠️ No speech detected.")
            return np.array([], dtype=np.float32)
        # Combine PCM frames and convert to float32
        pcm_data = b"".join(voiced_frames)
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32

    def remove_consecutive_duplicate_words(self, sentence: str) -> str:
        """
        Remove consecutive duplicate words in a sentence.
        Comparison is case-insensitive and strips punctuation for comparison.
        Preserves the original word (with punctuation) in output.
        E.g., "I want to to remove double words" -> "I want to remove double words"
        """
        words = sentence.split()
        cleaned = []
        prev_norm = None
        for w in words:
            # Normalize for comparison: strip leading/trailing punctuation, lowercase
            w_norm = w.strip(string.punctuation).lower()
            if prev_norm is None or w_norm != prev_norm:
                cleaned.append(w)
            else:
                # Duplicate consecutive word detected
                print(f"🔁 Removed duplicate word: \"{w}\" in sentence")
            prev_norm = w_norm
        return ' '.join(cleaned)

    def remove_duplicate_sentences(self, text: str) -> str:
        """
        Remove duplicate sentences from `text`, keeping only the first occurrence.
        Before comparing sentences, cleans consecutive duplicate words in each sentence.
        Splits on punctuation (.!?), compares case-insensitively after word-cleaning.
        """
        # Split into sentences, keeping punctuation attached at end
        sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
        seen = set()
        unique_sentences = []
        for s in sentences:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            # First, remove consecutive duplicate words in this sentence
            cleaned_sentence = self.remove_consecutive_duplicate_words(s_stripped)
            # Normalize for comparison: lowercase; you may strip trailing punctuation if desired
            norm = cleaned_sentence.strip().lower()
            if norm not in seen:
                seen.add(norm)
                unique_sentences.append(cleaned_sentence)
            else:
                # Duplicate sentence detected
                print(f"🔁 Removed duplicate sentence: \"{cleaned_sentence}\"")
        # Join back with spaces (punctuation stays with sentences)
        return ' '.join(unique_sentences)


    def listen_whisper(self):
        """
        Use VAD-based recording, then Whisper transcription, removing
        consecutive duplicate words and duplicate sentences.
        Passes through listen_text for math detection and formatting.
        """
        print("🎤 Starting VAD-based recording for Whisper transcription...")
        audio = self.record_until_silence(
            frame_duration_ms=30,
            vad_mode=3,
            silence_duration_ms=3000,
            max_record_duration_s=50
        )
        if audio.size == 0:
            print("⚠️ No audio captured for Whisper.")
            return None

        print("🎤 Transcribing with Whisper...")
        try:
            result = self.whisper_model.transcribe(audio, language="en", fp16=False)
        except Exception as e:
            print(f"❌ Whisper transcription error: {e}")
            return None

        raw_text = result.get("text", "").strip()
        print(f"📝 Whisper raw recognized: {raw_text}")

        if not raw_text:
            return None

        # Remove duplicate sentences and words
        text_no_dup_sentences = self.remove_duplicate_sentences(raw_text)
        print(f"📝 After removing duplicate sentences: {text_no_dup_sentences}")

        # Final normalization
        output = text_no_dup_sentences.lower()
##        my_final_output = re.sub(r'[.,!?;:]', '', output)
        my_final_output = re.sub(r'[!?;:]', '', output)
        print(f"📝 Processed output: {my_final_output}")

        # ✅ Now let listen_text handle math formatting and TTS
        return self.listen_text(my_final_output) or None

        # 1) Remove duplicate sentences (internally also removes consecutive duplicate words)
        text_no_dup_sentences = self.remove_duplicate_sentences(raw_text)
        print(f"📝 After removing duplicate sentences and double words: {text_no_dup_sentences}")

        # 2) Further normalization: lowercase & remove punctuation if desired
        output = text_no_dup_sentences.lower()
        # Example: strip .,!?;:  (optional)
        my_final_output = re.sub(r'[.,!?;:]', '', output)
        print(f"📝 Processed output: {my_final_output}")

        return my_final_output or None

    def listen_bluetooth(self):
        """Read one line from Bluetooth serial, if available."""
        if self.bluetooth:
            try:
                data = self.bluetooth.readline().decode("utf-8").strip()
                if data:
                    print(f"Bluetooth input received: {data}")
                    return data
            except serial.SerialException as e:
                print(f"❌ Bluetooth error: {e}")
        return None

    def listen(self):
        """
        Main entry point: check queued text, Bluetooth input, then speech.
        Returns recognized text or None.
        """
        # 1) Check if there's queued text
        try:
            text = self.text_queue.get_nowait()
            if text:
                return self.listen_text(text)
        except queue.Empty:
            pass

        # 2) Check Bluetooth input
        bt = self.listen_bluetooth()
        if bt:
            return bt

        # 3) Speech recognition
        if self.use_whisper_listen:
            return self.listen_whisper()
        else:
            return self.listen_vosk()

    def listen_bluetooth_only(self):
        """Wait only for Bluetooth input."""
        print("🎙️ Waiting for a Bluetooth query...")
        result = self.listen_bluetooth()
        return result or "No input"


# Create a singleton instance for import
WEBUIlisten = WEBUIListenModule()

print("speech module initialization end")


