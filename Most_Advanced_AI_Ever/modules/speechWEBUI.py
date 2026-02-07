

###     COOLEST VOICE (VERY COOL)


# speech.py

import os
import json
import time
import queue
import asyncio
import vosk
import sounddevice as sd
import serial
import pyttsx3
from playsound import playsound
import edge_tts

from WEBUI_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
##from arduino_comWEBUI import WEBUIArduinoCommunicationModule

print("speech start")

class 	WEBUISpeechModule:
    
    def __init__(self):
        """Initialize Text-to-Speech and serial communication."""
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 190)
        self.engine.setProperty("voice", pyttsx3.init().getProperty("voices")[1].id)

        self.speak_queue = queue.Queue()
        self.Sending_On = False
        self.Start_Speech = False
        self.Stop_Speech = False

##    async def speak_edge_tts(self, text, voice="en-US-GuyNeural", style="cheerful"):
    async def speak_edge_tts(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
        """Speak using Microsoft Edge TTS with emotion and style."""

        ssml = f"{text}"

        communicate = edge_tts.Communicate(text=ssml, voice=voice)
        await communicate.save("output.mp3")
        playsound("output.mp3")

    def AlfredSpeak_Onboard(self, text):
        """Fallback TTS using onboard voice."""
        self.engine.say(text)
        self.engine.runAndWait()

    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
        """Main Alfred speaking function with expressive voice and Arduino control."""
        data_start_talk = "T"
        self.Sending_On = True


        ###  ARDUINO MOUTH SPEAKING CONTROL 

        while self.Sending_On:
            self.Start_Speech = True
##            print(f"data_start_talk : {data_start_talk}")
##
####            for i in range(10):
####                arduino.send_arduino(data_start_talk)
####                print(f" i : {i + 1}")

            if self.Start_Speech:
                time.sleep(0.01)
                try:
                    # Run the async TTS
                    asyncio.run(self.speak_edge_tts(text, voice=voice, style=style))
                    self.Stop_Speech = True
                    self.Start_Speech = False
                except Exception as e:
                    print("Error during speech:", e)

            if self.Stop_Speech:
##                data_start_talk = "S"
##                print(f"data_start_talk : {data_start_talk}")
##
####                for i in range(10):
####                    arduino.send_arduino(data_start_talk)
####                    print(f" i : {i + 1}")

                self.Stop_Speech = False

            time.sleep(0.01)
            self.Stop_Speech = False
            self.Start_Speech = False
            self.Sending_On = False






    def AlfredSpeak_Bluetooth(self, text):
        listen.send_bluetooth(text)
        """Converts text to speech."""
        self.engine.say(text)
        self.engine.runAndWait()

        
    def callback(self, indata, frames, time, status):
        """Callback function to store speech input."""
        if status:
            print(f"❌ Error: {status}")
        self.queue.put(bytes(indata))
        

print("speech end")

# ✅ Initialize Speech Module
speechWEBui = WEBUISpeechModule()


