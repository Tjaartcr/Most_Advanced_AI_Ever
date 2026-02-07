



import datetime
import random
import pyjokes
import keyboard

from speech import speech
##from communication import comm
from memory import memory
from Repeat_Last import repeat
##from assistant import assistant
from listen import listen
from GUI import gui
import Alfred_config
import afr_eng_detect

import openai
import pdb
import pyttsx3  #pip install pyttsx3
import speech_recognition as sr
import datetime  #module
##import wikipedia    ?????
import smtplib
import webbrowser as wb
import os  #inbuilt
from os import listdir
import psutil  #pip install psutil
import pyjokes  # pip install pyjokes
import requests, json  #inbuilt
import time
from PIL import Image, ImageGrab
import subprocess
import string
import pyautogui
import wolframalpha
import pyperclip
import sys
import google
import operator
import random
import tkinter as tk
import argparse
import json
##import googlesearch    ??????
import serial
import randfacts
import curses
import cv2
import keyboard
import PyPDF2
from PyPDF2 import PdfFileReader
import numpy as np
import re
import urllib3
from os import startfile

##from vosk import Model, KaldiRecognizer
import pyttsx3
import datetime
import pyaudio
import json
import os
import time
import playsound
from playsound import playsound
import msvcrt as m
from word2number import w2n
import wordtodigits
import sounddevice as sd

import subprocess

import wave
import sys
##import pyaudio

import ollama
import nltk
from ollama import chat

from vosk import Model, KaldiRecognizer, SetLogLevel

import threading
import multiprocessing
from multiprocessing import Process, Value, Array, Lock
import queue

import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

import cameras_flash_on_off

print("assistant")

MyToDoListEdited = ""



class AlfredAssistant(gui):

    Alfred_No_Ollama = 0
    Silence_Counter_Quiet = 0
    Alfred_No_Ollama = 0
    Silence_Counter2 = 0
    AddPage = 0
    PreviousPage = 1

    Memory = []
    Do_History = []
    Did_History = []
    ToDoList = []

    MyToDoListEditedWithout = ""
    MyToDoListEdited = ""
    AlfredQueryOfflineNew = ""
    AlfredQueryOfflineToDoList = ""

    global Alfred_Repeat_Previous_Response
    global log_queue

    Alfred_Repeat_Previous_Response = ""

    # Flags and queues for threading
    stop_flag = False
    muted = False  # Global flag for text-to-speech
    log_queue = queue.Queue()

                
    def __init__(self, gui):
        self.gui = None  # This will be set later (in main.py)


    def assistant_extract_text_from_query(self, AlfredQueryOffline):
        """
        Return: (message, username)
        Always returns strings: message (possibly empty) and username (fallback 'ITF').
        """
        print(f"assistant_extract_text_from_query input: {AlfredQueryOffline!r}")

        if not AlfredQueryOffline:
            return "", "ITF"

        # If dict — prefer explicit fields
        if isinstance(AlfredQueryOffline, dict):
            message = str(AlfredQueryOffline.get("text") or AlfredQueryOffline.get("query") or AlfredQueryOffline.get("q") or "").strip()
            username = str(AlfredQueryOffline.get("username") or AlfredQueryOffline.get("user") or getattr(self, "current_user", "ITF"))
            print(f"[DICT] user={username} msg={message}")
            return message, username

        # If string — try timestamped pattern first, then simpler forms
        if isinstance(AlfredQueryOffline, str):
            s = AlfredQueryOffline.strip()

            # 1) Timestamped: YYYY-MM-DD : HH:MM:SS : message : 'username':Name
            m = re.match(r"^(\d{4}-\d{2}-\d{2})\s*:\s*(\d{2}:\d{2}:\d{2})\s*:\s*(.*?)\s*:\s*'username':\s*(\w+)\s*$", s)
            if m:
                date_str, time_str, message, username = m.groups()
                print(f"[TS PARSED w/username] date={date_str} time={time_str} user={username} msg={message}")
                return message.strip(), username.strip()

            # 1b) Timestamped: YYYY-MM-DD : HH:MM:SS : message : Username  (no 'username' label)
            m1b = re.match(r"^'?(\d{4}-\d{2}-\d{2})\s*:\s*(\d{2}:\d{2}:\d{2})\s*:\s*(.*?)\s*:\s*(.+?)'?$", s)
            if m1b:
                date_str, time_str, message, username = m1b.groups()
                print(f"[TS PARSED bare] date={date_str} time={time_str} user={username} msg={message}")
                return message.strip(), username.strip()

            # 2) Simple: message : 'username':Name
            m2 = re.match(r"^(.*?)\s*:\s*'username':\s*(\w+)\s*$", s)
            if m2:
                message, username = m2.groups()
                return message.strip(), username.strip()

            # 3) Other username patterns: username=Name or user:Name
            m3 = re.search(r"(?i)(?:\busername\b|\buser\b)\s*[:=]\s*'?(?P<u>\w+)'?", s)
            if m3:
                username = m3.group("u")
                # remove username fragment from message
                message = re.sub(r"(?i)(?:[:\s]*\busername\b\s*[:=]\s*'?\w+'?)", "", s).strip(" :")
                return message, username

            # 4) fallback: treat whole string as message, use current_user as username
            return s, getattr(self, "current_user", "ITF")

        # fallback for other types
        return str(AlfredQueryOffline).strip(), getattr(self, "current_user", "ITF")


    def handle_send_email(self):

        affirmatives = {"yes", "correct", "awesome", "fantastic", "great"}
        negatives = {"no", "nope", "incorrect", "false"}

        while True:
            speech.AlfredSpeak("To whom would you like to send the email, sir?")
            query_email_to = listen.listen()
            if not query_email_to:
                continue

            if "chart" in query_email_to or "boss" in query_email_to:
                to = "tjaartcronje@gmail.com"
            else:
                to = query_email_to.strip()

            time.sleep(1)

            speech.AlfredSpeak("What would you like it to be about, sir?")
            query_email_content = listen.listen()
            if not query_email_content:
                continue
            content = query_email_content.strip()

            time.sleep(1)

            speech.AlfredSpeak("Is this what you want to send, sir?")
            speech.AlfredSpeak(content)

            query_reply = listen.listen()
            if not query_reply:
                continue

            reply = query_reply.lower()

            def sendEmail(to, content):
                if to is None or content is None:
                    print("ERROR: Email or content is None.")
                    return
                Message_to_Send = f".  To : {to}. Message : {content}"
                print(f"After : {Message_to_Send}")


                print(f"Email User: {Alfred_config.EMAIL_HOST_USER}, Password is {'set' if Alfred_config.EMAIL_HOST_PASSWORD else 'missing'}")
                if not Alfred_config.EMAIL_HOST_USER or not Alfred_config.EMAIL_HOST_PASSWORD:
                    print("Missing email credentials.")
                    return


                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.ehlo()
                server.starttls()
                server.login(Alfred_config.EMAIL_HOST_USER, Alfred_config.EMAIL_HOST_PASSWORD)
                server.sendmail("tjaartcronje@gmail.com", to, content)
                server.close()

            if any(word in reply for word in affirmatives):
                Message_to_Send = f".  To : {to}. Message : {content}"
                print(f"Before : {Message_to_Send}")
                speech.AlfredSpeak("Email sent, sir.")
                sendEmail(to, content)
                break

            elif any(word in reply for word in negatives):
                speech.AlfredSpeak("Okay, let's start over.")
                return

            else:
                speech.AlfredSpeak("I didn't catch that. Please say yes or no.")


    def greet(self, AlfredQueryOffline):
        """Greet the user based on the time of day."""
        hour = datetime.datetime.now().hour
        if hour < 12:
            listen.send_bluetooth("Good morning, sir!")
            speech.AlfredSpeak("Good morning, sir!")
        elif hour < 18:
            listen.send_bluetooth("Good afternoon, sir!")
            speech.AlfredSpeak("Good afternoon, sir!")
        else:
            lipesten.send_bluetooth("Good evening, sir!")
            speech.AlfredSpeak("Good evening, sir!")


    def wait(self, AlfredQueryOffline):

        global Alfred_Repeat_Previous_Response

        Alfred_Repeat_Previous_Response = f"paused sir, Please click on CMD box and  press the escape key to continue"
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        speech.AlfredSpeak("paused sir, Please click on CMD box and  press the escape key to continue")
        listen.send_bluetooth("paused sir, Please click on CMD box and  press the escape key to continue")
        print("paused sir, Please click on CMD box and press the 'Esc' key to continue")

        time.sleep(2)
        
        key = keyboard.read_key()

        while True:
            
            if key == "esc":
                cls
                Wake_up_Greeting()
                return 
                    
    def welcome(self, AlfredQueryOffline):

        hour=datetime.datetime.now().hour
        if hour >=3 and hour <12:
            listen.send_bluetooth("Good morning, sir!")
            speech.AlfredSpeak('Good morning sir')
        elif hour >=12 and hour <18:
            listen.send_bluetooth("Good afternoon, sir!")
            speech.AlfredSpeak('Good afternoon sir')
        elif hour >=18 and hour <21:
            listen.send_bluetooth("Good evening, sir!")
            speech.AlfredSpeak('Good evening sir')
        elif hour >=21 and hour <24:
            listen.send_bluetooth("Good night and have a nice dream, sir!")
            speech.AlfredSpeak('Good night and have a nice dream, sir!')
        elif hour >=0 and hour <3:
            listen.send_bluetooth('It is late sir, let us take a nap')
            speech.AlfredSpeak('It is late sir, let us take a nap')
        listen.send_bluetooth("How can I help you, now sir?")
        speech.AlfredSpeak('How can I help you now')

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I greeted you and I said listening..."
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Greetings....", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
        
        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
            f"and I replied:\n\n{Alfred_Repeat_Previous_Response}\n\n"
        )
           
        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")
        
        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

        print('')
        print('listening ...')
        print('')
        
        main.main(assistant, gui)

    import socket


    def Send_Message(self, AlfredQueryOffline):
        
        my_socket = socket.socket()

        while True:

            port = 80

            ip = IP_ADRESS

            my_socket.connect((ip, port))
        
            msg = input("Message: ") + '\n'
            my_socket.send(msg.encode())

            return Send_Message()

    def Receive_Message(self, AlfredQueryOffline):
        
        my_socket = socket.socket()

        while True:

            port = 80

            ip = IP_ADRESS

            my_socket.connect((ip, port))
        
            msg = (my_socket.recv(1024).decode())
            print("Message Received: ", msg)
            
            return Send_Message()

    def set_flash_on(self):
        cameras_flash_on_off.set_flash_on_left(Alfred_config.LEFT_EYE_CAMERA_INPUT_NEW)
        cameras_flash_on_off.set_flash_on_right(Alfred_config.RIGHT_EYE_CAMERA_INPUT_NEW)

    def set_flash_off(self):
        cameras_flash_on_off.set_flash_off_left(Alfred_config.LEFT_EYE_CAMERA_INPUT_NEW)
        cameras_flash_on_off.set_flash_off_right(Alfred_config.RIGHT_EYE_CAMERA_INPUT_NEW)

    

    def timeOfDay(self, AlfredQueryOffline):




        if afr_eng_detect.afrikaans_spoken:
            print("[DEBUG ASSISTANT timeOfDay] Afrikaans detected!")
        elif afr_eng_detect.english_spoken:
            print("[DEBUG ASSISTANT timeOfDay] English detected!")


        print(f"[DEBUG LISTEN TEXT] Time of Day AlfredQueryOffline : {AlfredQueryOffline}")

        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"[DEBUG ASSISTANT timeOfDay] NEW  message : {message!r}")
        print(f"[DEBUG ASSISTANT timeOfDay] NEW  username: {username!r}")

        # -------------------------------------------------
        # From here you have BOTH:
        #   username = "Tjaart"
        #   message  = "tell me the date today."
        # -------------------------------------------------

        import datetime
        global log_queue
        global Alfred_Repeat_Previous_Response

        now = datetime.datetime.now()
        hour = now.hour
        minute = now.minute
        second = now.second

        # Number → words helper
        def number_to_words(n):
            ones = [
                "zero", "one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen", "nineteen"
            ]
            tens = ["", "", "twenty", "thirty", "forty", "fifty"]
            if n < 20:
                return ones[n]
            ten, one = divmod(n, 10)
            return tens[ten] if one == 0 else f"{tens[ten]} {ones[one]}"

        # 12-hour conversion
        hour_12 = hour % 12 or 12
        if hour < 12:
            period = "in the morning"
        elif hour < 17:
            period = "in the afternoon"
        else:
            period = "in the evening"


##        if afr_eng_detect.english_spoken:
##
##
##            # Ensure hour_12 is 1..12 (not 0..11)
##            # assuming `hour` is 0..23 originally:
##            hour_12 = hour % 12 or 12
##
##            # compute next_hour once (wrap 12 -> 1)
##            next_hour = (hour_12 % 12) + 1
##
##            # Spoken clock building
##            if minute == 0:
##                spoken_clock = (
##                    f"{number_to_words(hour_12)} o clock exactly {period}"
##                    if second == 0
##                    else f"{number_to_words(hour_12)} o clock and {number_to_words(second)} seconds {period}"
##                )
##            elif minute == 15:
##                spoken_clock = f"quarter past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"
##            elif minute == 30:
##                # Use next_hour for "half <next_hour>" style (e.g. 13:30 -> "half two")
##                spoken_clock = f"half {number_to_words(next_hour)} and {number_to_words(second)} seconds {period}"
##            elif minute == 45:
##                # already used next_hour style for quarter before
##                spoken_clock = f"quarter before {number_to_words(next_hour)} and {number_to_words(second)} seconds {period}"
##            else:
##                spoken_clock = f"{number_to_words(minute)} minutes past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"



##            # Spoken clock
##            if minute == 0:
##                spoken_clock = (
##                    f"{number_to_words(hour_12)} o clock exactly {period}"
##                    if second == 0
##                    else f"{number_to_words(hour_12)} o clock and {number_to_words(second)} seconds {period}"
##                )
##            elif minute == 15:
##                spoken_clock = f"quarter past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"
##            elif minute == 30:
##                spoken_clock = f"half past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"
##            elif minute == 45:
##                next_hour = (hour_12 % 12) + 1
##                spoken_clock = f"quarter to {number_to_words(next_hour)} and {number_to_words(second)} seconds {period}"
##            else:
##                spoken_clock = f"{number_to_words(minute)} past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"


        if afr_eng_detect.afrikaans_spoken:


            # Ensure hour_12 is 1..12 (not 0..11)
            # assuming `hour` is 0..23 originally:
            hour_12 = hour % 12 or 12

            # compute next_hour once (wrap 12 -> 1)
            next_hour = (hour_12 % 12) + 1

            # Spoken clock building
            if minute == 0:
                spoken_clock = (
                    f"{number_to_words(hour_12)} o clock exactly {period}"
                    if second == 0
                    else f"{number_to_words(hour_12)} o clock and {number_to_words(second)} seconds {period}"
                )
            elif minute == 15:
                spoken_clock = f"quarter past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"
            elif minute == 30:
                # Use next_hour for "half <next_hour>" style (e.g. 13:30 -> "half two")
                spoken_clock = f"half {number_to_words(next_hour)} and {number_to_words(second)} seconds {period}"
            elif minute == 45:
                # already used next_hour style for quarter before
                spoken_clock = f"quarter before {number_to_words(next_hour)} and {number_to_words(second)} seconds {period}"
            else:
                spoken_clock = f"{number_to_words(minute)} minutes past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"


            printed_time = now.strftime("%H:%M:%S")
            Alfred_Repeat_Previous_Response = f"I said the current time is {spoken_clock}"
            print(f"[DEBUG ASSISTANT timeOfDay] Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

            # Log to memory
            response = Alfred_Repeat_Previous_Response
            current_date = now.strftime("%Y-%m-%d")
            current_time = printed_time

            chat_entry = {
                "date": current_date,
                "time": current_time,
                "query": message,     # << use parsed message
                "username": username, # << log username too
                "response": response
            }
            memory.add_to_memory(chat_entry)
            repeat.add_to_repeat(chat_entry)

            query_msg = (
                f"At {current_date} :  {current_time} : You Asked: {message} "
            )

            print(f"[DEBUG ASSISTANT timeOfDay] \n query_msg : {query_msg} \n")

            model = "Alfred"
            query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

            print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
            print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")
        
            print(f"[DEBUG ASSISTANT timeOfDay] \n query_resp : {query_resp} \n")

            try:
                self.gui.log_message(query_msg)
                self.gui.log_response(query_resp)
            except Exception as e:
                print(f"error in assistant gui : {e}")

            # Speak out loud
            listen.send_bluetooth(f"The time now is {spoken_clock}")
            speech.AlfredSpeak(f"The time now is {spoken_clock}")

            return username, message


        else:

            # Ensure hour_12 is 1..12 (not 0..11)
            # assuming `hour` is 0..23 originally:
            hour_12 = hour % 12 or 12

            # compute next_hour once (wrap 12 -> 1)
            next_hour = (hour_12 % 12) + 1

            # Spoken clock building
            if minute == 0:
                spoken_clock = (
                    f"{number_to_words(hour_12)} o clock exactly {period}"
                    if second == 0
                    else f"{number_to_words(hour_12)} o clock and {number_to_words(second)} seconds {period}"
                )
            elif minute == 15:
                spoken_clock = f"quarter past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"
            elif minute == 30:
                # Use next_hour for "half <next_hour>" style (e.g. 13:30 -> "half two")
                spoken_clock = f"half {number_to_words(next_hour)} and {number_to_words(second)} seconds {period}"
            elif minute == 45:
                # already used next_hour style for quarter before
                spoken_clock = f"quarter before {number_to_words(next_hour)} and {number_to_words(second)} seconds {period}"
            else:
                spoken_clock = f"{number_to_words(minute)} minutes past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"

            printed_time = now.strftime("%H:%M:%S")
            Alfred_Repeat_Previous_Response = f"I said the current time is {spoken_clock}"
            print(f"[DEBUG ASSISTANT timeOfDay] Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

            # Log to memory
            response = Alfred_Repeat_Previous_Response
            current_date = now.strftime("%Y-%m-%d")
            current_time = printed_time

            chat_entry = {
                "date": current_date,
                "time": current_time,
                "query": message,     # << use parsed message
                "username": username, # << log username too
                "response": response
            }
            memory.add_to_memory(chat_entry)
            repeat.add_to_repeat(chat_entry)

            query_msg = (
                f"At {current_date} :  {current_time} : You Asked: {message} "
            )

            print(f"[DEBUG ASSISTANT timeOfDay] \n query_msg : {query_msg} \n")

            model = "Alfred"
            query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

            print(f"[DEBUG ASSISTANT timeOfDay] \n query_resp : {query_resp} \n")

            print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
            print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

            try:
                self.gui.log_message(query_msg)
                self.gui.log_response(query_resp)
            except Exception as e:
                print(f"error in assistant gui : {e}")

            # Speak out loud
            listen.send_bluetooth(f"The time now is {spoken_clock}")
            speech.AlfredSpeak(f"The time now is {spoken_clock}")

            return username, message


##            # Spoken clock
##            if minute == 0:
##                spoken_clock = (
##                    f"{number_to_words(hour_12)} o clock exactly {period}"
##                    if second == 0
##                    else f"{number_to_words(hour_12)} o clock and {number_to_words(second)} seconds {period}"
##                )
##            elif minute == 15:
##                spoken_clock = f"quarter past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"
##            elif minute == 30:
##                spoken_clock = f"half {number_to_words(hour_12 - 1)} and {number_to_words(second)} seconds {period}"
##            elif minute == 45:
##                next_hour = (hour_12 % 12) + 1
##                spoken_clock = f"quarter before {number_to_words(next_hour)} and {number_to_words(second)} seconds {period}"
##            else:
##                spoken_clock = f"{number_to_words(minute)} minutes past {number_to_words(hour_12)} and {number_to_words(second)} seconds {period}"



##        printed_time = now.strftime("%H:%M:%S")
##        Alfred_Repeat_Previous_Response = f"I said the current time is {spoken_clock}"
##        print(f"[DEBUG ASSISTANT timeOfDay] Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")
##
##        # Log to memory
##        response = Alfred_Repeat_Previous_Response
##        current_date = now.strftime("%Y-%m-%d")
##        current_time = printed_time
##
##        chat_entry = {
##            "date": current_date,
##            "time": current_time,
##            "query": message,     # << use parsed message
##            "username": username, # << log username too
##            "response": response
##        }
##        memory.add_to_memory(chat_entry)
##        repeat.add_to_repeat(chat_entry)
##
##        query_msg = (
##            f"At {current_date} :  {current_time} : You Asked: {message} "
##        )
##
##        print(f"[DEBUG ASSISTANT timeOfDay] \n query_msg : {query_msg} \n")
##
##        model = "Alfred"
##        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"
##
##        print(f"[DEBUG ASSISTANT timeOfDay] \n query_resp : {query_resp} \n")
##
####        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "
##        
##        try:
##            self.gui.log_message(query_msg)
##            self.gui.log_response(query_resp)
##        except Exception as e:
##            print(f"error in assistant gui : {e}")
##
##        # Speak out loud
##        listen.send_bluetooth(f"The time now is {spoken_clock}")
##        speech.AlfredSpeak(f"The time now is {spoken_clock}")
##
##        return username, message


    def date(self, AlfredQueryOffline):
        global log_queue

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")
       
        # Get current datetime
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        day_num = now.day
        weekday_num = now.weekday()

        # Weekday names
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday = weekdays[weekday_num]

        # Month name
        month_name = now.strftime('%B')

        # Ordinal suffix function
        def ordinal(n):
            if 10 <= n % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
            return f"{n}{suffix}"

        # Prepare spoken vs printed strings
        spoken_date = f"{ordinal(day_num)} of {month_name} {year}"
        print(f"[DEBUG DATE] spoken_date : {spoken_date}")
        
        printed_date = f"{day_num:02d} {month_name} {year}"

        # Debug prints
        print(f"[DEBUG DATE] {printed_date}, {weekday}")

        # Build the response
        The_date = spoken_date  # what Alfred will say
        Alfred_Repeat_Previous_Response = f"I said the current date is {weekday} the {The_date}"
        print(Alfred_Repeat_Previous_Response)

        response = Alfred_Repeat_Previous_RESPONSE = Alfred_Repeat_Previous_Response
        current_date = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M:%S')

        # Log to memory and repeat buffers
        chat_entry = {
            "date": current_date,
            "time": current_time,
            "query": "The current date...?",
            "response": response
        }
        memory.add_to_memory(chat_entry)
        repeat.add_to_repeat(chat_entry)

        # GUI logging if available
        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
            f"and I replied:\n\n{Alfred_Repeat_Previous_Response}\n\n"
        )

        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")


##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "
        
        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

        # Speak and send via Bluetooth
        listen.send_bluetooth(f"The current date is {weekday} the {The_date}")
        speech.AlfredSpeak(f"The current date is {weekday} the {The_date}")



        # Show remaining to‑do list items if any
        print("")
        print("[DEBUG DATE] MyToDoListEdited New : " + str(MyToDoListEdited))
        print("")

        try:
            if MyToDoListEdited:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("[DEBUG DATE] AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
                print("")
        except Exception:
            pass


    def day(self, AlfredQueryOffline):
        global log_queue

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        if not AlfredQueryOffline:
            print("No text received.")
            return None, None

        username = "ITF"
        message = ""

        # Case 1: Proper dict
        if isinstance(AlfredQueryOffline, dict):
            username = AlfredQueryOffline.get("username", "ITF")
            message = AlfredQueryOffline.get("query", "")
            print(f"[DICT DAY] Received from {username}: {message}")

        # Case 2: String that looks like "query : 'username':Tjaart"

        elif isinstance(AlfredQueryOffline, str):
            import re

            # Match: date : time : message : 'username':Name
            pattern = r"^(\d{4}-\d{2}-\d{2}) : (\d{2}:\d{2}:\d{2}) : (.*?) : 'username':(\w+)$"
            match = re.match(pattern, AlfredQueryOffline.strip())

            if match:
                date_str, time_str, message, username = match.groups()
                message = message.strip()
                username = username.strip()
                print(f"[PARSED STRING DAY] {date_str} {time_str} | User={username} | Msg={message}")
            else:
                # fallback
                message = AlfredQueryOffline.strip()
                try:
                    username = self.current_user
                except AttributeError:
                    username = "unknown"
                print(f"[RAW STRING DAY] Received from {username}: {message}")

        
        year = int(datetime.datetime.now().year)
        month = int(datetime.datetime.now().month)
        date = int(datetime.datetime.now().day)
        day = int (datetime.datetime.now().weekday())

        if (day == 0):
            day = 'Monday'
        elif (day == 1):
            day = 'Tuesday'
        elif (day == 2):
            day = 'Wednessday'
        elif (day == 3):
            day = 'Thursday'
        elif (day == 4):
            day = 'Friday'
        elif (day == 5):
            day = 'Saterday'
        elif (day == 6):
            day = 'Sunday'

        print(day)

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I said the current day is {day}"
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "The current day...?", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response

        
        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
            f"and I replied:\n\n{Alfred_Repeat_Previous_Response}\n\n"
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")


##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "
        
        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        
        listen.send_bluetooth(f"It is {day} today, sir")
        speech.AlfredSpeak(f"It is {day} today, sir")


##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening ...")

        print("")
        print("MyToDoListEdited New : " + str(MyToDoListEdited))
        print('')
            
        try :
                
            if MyToDoListEdited != []:
            
                MyToDoListEdited.pop(0)

                AlfredQueryOfflineNew = MyToDoListEdited
                
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            
            pass


    def checktime(self, tt):
        global log_queue

        hour = datetime.datetime.now().hour
        
        if ("morning" in tt):
            if (hour >= 6 and hour < 12):
                listen.send_bluetooth("Good morning sir")
                speech.AlfredSpeak("Good morning sir")
            else:
                if (hour >= 12 and hour < 18):
                    listen.send_bluetooth("it's Good afternoon sir")
                    speech.AlfredSpeak("it's Good afternoon sir")
                elif (hour >= 18 and hour < 24):
                    listen.send_bluetooth("it's Good Evening sir")
                    speech.AlfredSpeak("it's Good Evening sir")
                else:
                    listen.send_bluetooth("it's Goodnight sir")
                    speech.AlfredSpeak("it's Goodnight sir")
        elif ("afternoon" in tt):
            if (hour >= 12 and hour < 18):
                listen.send_bluetooth("it's Good afternoon sir")
                speech.AlfredSpeak("it's Good afternoon sir")
            else:
                if (hour >= 6 and hour < 12):
                    listen.send_bluetooth("Good morning sir")
                    speech.AlfredSpeak("Good morning sir")
                elif (hour >= 18 and hour < 24):
                    listen.send_bluetooth("it's Good Evening sir")
                    speech.AlfredSpeak("it's Good Evening sir")
                else:
                    listen.send_bluetooth("it's Goodnight sir")
                    speech.AlfredSpeak("it's Goodnight sir")
                    
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening ...")


    #greeting function
    def Geeting(self, AlfredQueryOffline):
        global log_queue

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
    #    speech.AlfredSpeak("Welcome Back")
        hour = datetime.datetime.now().hour
        if (hour >= 6 and hour < 12):
            listen.send_bluetooth("Good morning sir!")
            speech.AlfredSpeak("Good morning sir!")
        elif (hour >= 12 and hour < 18):
            listen.send_bluetooth("Good afternoon sir")
            speech.AlfredSpeak("Good afternoon sir")
        elif (hour >= 18 and hour < 24):
            listen.send_bluetooth("Good evening sir")
            speech.AlfredSpeak("Good evening sir")
        else:
            listen.send_bluetooth("Good evening sir")
            speech.AlfredSpeak("Good evening sir")

        listen.send_bluetooth("How are you doing, sir?")
        speech.AlfredSpeak("How are you doing, sir?")

        Response()

    def Response(self, AlfredQueryOffline):

        data = stream.read(8000, exception_on_overflow=False)
        
        Serial_Bluetooth.flushInput()
        Bluetooth_RX = Serial_Bluetooth.readline()

        print(Bluetooth_RX)

        command = Bluetooth_RX
        command = str(command, 'utf-8')    
        print("command : " ,command)
                  
        AlfredQueryOffline = command
        print('sir: ' + AlfredQueryOffline)


        
        if ("great" in AlfredQueryOffline and "and you" in AlfredQueryOffline):
            speech.AlfredSpeak("I am happy that you are doing great sir")         
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Return 3.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())
            
        elif ("awesome" in AlfredQueryOffline and "and you" in AlfredQueryOffline):
            speech.AlfredSpeak("I am happy that you are feeling awesome sir")         
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Return 3.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())
            
        elif ("fantastic" in AlfredQueryOffline and "and you" in AlfredQueryOffline):
            speech.AlfredSpeak("I am happy that you are feeling fantastic sir")         
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Return 3.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())
            
        elif ("angry" in AlfredQueryOffline and "and you" in AlfredQueryOffline):
            speech.AlfredSpeak("Please don't be angry sir")    
            speech.AlfredSpeak("calm down, relax..?")    
            speech.AlfredSpeak("breath in ... and ... out ...?")    
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Mad.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())
            
        elif ("sad" in AlfredQueryOffline and "and you" in AlfredQueryOffline):
            speech.AlfredSpeak("I am so sorry that you are sad sir")
            speech.AlfredSpeak("let's try to cheer you up?")
            JokesAllSad()      
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Sad.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())
            main.main(assistant, gui)
            
        elif ("happy" in AlfredQueryOffline and "and you" in AlfredQueryOffline):
            speech.AlfredSpeak("I am glad that you are happy sir")
            speech.AlfredSpeak("I am happy that we could chat sir")
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Return 3.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())

        elif ("great" in AlfredQueryOffline ):
            speech.AlfredSpeak("I am happy that you are great")
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Return 1.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())

        elif ("awesome" in AlfredQueryOffline):
            speech.AlfredSpeak("I am happy that you are awesome")
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Return 1.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())


        elif ("fantastic" in AlfredQueryOffline):
            speech.AlfredSpeak("I am happy that you are fantastic")
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Return 1.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())

        elif ("angry" in AlfredQueryOffline):
            speech.AlfredSpeak("Please don't be angry sir")    
            speech.AlfredSpeak("calm down, relax..?")    
            speech.AlfredSpeak("breath in ... and ... out ...?")    
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Mad.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())
            os.startfile(Alfred_config.DRIVE_LETTER+'Python_Env//New_Virtual_Env//Project_Files//My_Mp3//chill songs.mp3')

        elif ("sad" in AlfredQueryOffline):
            speech.AlfredSpeak("I am so sorry that you are sad sir")
            speech.AlfredSpeak("let's try to cheer you up?")
            JokesAllSad()
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Sad.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())
            os.startfile(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Project_Files//My_Mp3//Sad song.mp3")

        elif ("happy" in AlfredQueryOffline):
            speech.AlfredSpeak("I am glad that you are happy sir")
            speech.AlfredSpeak("Let's party")
            resGreeting = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Greeting Return 1.txt")
            listen.send_bluetooth(resGreeting.read())
            speech.AlfredSpeak(resGreeting.read())
            os.startfile(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Project_Files//My_Mp3//102-coldplay-viva_la_vida.mp3")
        
        elif ("thanks" in AlfredQueryOffline or "thank you" in AlfredQueryOffline or "ok" in AlfredQueryOffline
            or "not" in AlfredQueryOffline or "no thanks" in AlfredQueryOffline or "ok" in AlfredQueryOffline
            or "no" in AlfredQueryOffline or "nope" in AlfredQueryOffline or "ok" in AlfredQueryOffline):
              
            listen.send_bluetooth("I am glad that you could chat, sir. Have an awesome time and live life to it's fullest")
            speech.AlfredSpeak("I am glad that we could chat sir")
            speech.AlfredSpeak("have an awesome time and live life to it's fullest")
                    
        try:
            self.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
        
##            listen.send_bluetooth("Listening...")
##            speech.AlfredSpeak("Listening...")

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")

        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass
        
            return Response()        

        else:

            return Response() 


    #welcome function
    def wishme(self):

        hour = datetime.datetime.now().hour
        if (hour >= 6 and hour < 12):
##            speech.AlfredSpeak_Onboard("Good morning sir!")
            speech.AlfredSpeak("Good morning sir!")
                               
        elif (hour >= 12 and hour < 18):
            speech.AlfredSpeak("Good afternoon sir")

        elif (hour >= 18 and hour < 24):
            speech.AlfredSpeak("Good evening sir!")

        else:
            speech.AlfredSpeak("Good evening sir!")

        time.sleep(1.5)
        
        listen.send_bluetooth("Please, check your Query's and Responses on the graphics user interface. Also don't forget to check out our Newest, Web User Interface Running on our localhost. On Port, 5000")
        speech.AlfredSpeak("Please, check your Query's and Responses on the graphics user interface. Also don't forget to check out our Newest, Web User Interface Running on our localhost. On Port, 5000")

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I said Let's connect...? and Listening...."
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Greetings....", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
 

    #welcome function
    def Wake_up_Greeting(self, AlfredQueryOffline):
        global log_queue

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        hour = datetime.datetime.now().hour
        if (hour >= 6 and hour < 12):
            listen.send_bluetooth("Good morning sir! Welcome back. what a wonderful morning... . I hope you have a fantastic day. How can i assist you, sir")
            speech.AlfredSpeak("Good morning sir! Welcome back. what a wonderful morning... . I hope you have a fantastic day. How can i assist you, sir")


        elif (hour >= 12 and hour < 18):
            listen.send_bluetooth("Good afternoon sir! Welcome back. what an Awesome afternoo... . I hope you have an awesome afternoon. How can i assist you, sir")
            speech.AlfredSpeak("Good afternoon sir! Welcome back. what an Awesome afternoo... . I hope you have an awesome afternoon. How can i assist you, sir")


        elif (hour >= 18 and hour < 22):
            listen.send_bluetooth("Good evening sir! Welcome back. Hope you had a great day... . I hope you have an fantastic evening. How can i assist you, sir")
            speech.AlfredSpeak("Good evening sir! Welcome back. Hope you had a great day... . I hope you have an fantastic evening. How can i assist you, sir")

                        
        elif (hour >= 22 and hour < 24):
            listen.send_bluetooth("Good evening sir! Welcome back. What a great evening... . It is very late, sir. You should go to bed... How can i assist you, sir")
            speech.AlfredSpeak("Good evening sir! Welcome back. What a great evening...  . It is very late, sir. You should go to bed... How can i assist you, sir")

                        
##        listen.send_bluetooth('listening ...')
##        speech.AlfredSpeak('listening ...')


        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I welcomed you back...? and I said Listening...."
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Wake up Greeting....", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
  
        GUI.log_message(f"You Asked: Greetings to you and I replied : {Alfred_Repeat_Previous_Response}")


    def SecretaryWishme(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")


        
    #    speech.AlfredSpeak("Welcome Back")
        hour = datetime.datetime.now().hour
        if (hour >= 6 and hour < 12):
            SecretarySpeak("Good morning sir!")
        elif (hour >= 12 and hour < 18):
            SecretarySpeak("Good afternoon sir")
        elif (hour >= 18 and hour < 24):
            SecretarySpeak("Good evening sir")
        else:
            SecretarySpeak("Good evening sir")

        SecretarySpeak("Welcome back.")
        SecretarySpeak("I am your Secretary. My name is Wifi. I am Einsteins wife. Please ask me who to call to assist you? Alfred or Einstein sir?")

        return Waiting_Room()


    def wishme_end(self, AlfredQueryOffline):
        ("signing off")
        hour = datetime.datetime.now().hour
        if (hour >= 6 and hour < 12):
            listen.send_bluetooth("Have a good Morning")
            speech.AlfredSpeak("Have a good Morning")
        elif (hour >= 12 and hour < 18):
            listen.send_bluetooth("Have a good afternoon")
            speech.AlfredSpeak("Have a good afternoon")
        elif (hour >= 18 and hour < 24):
            listen.send_bluetooth("Have a good Evening")
            speech.AlfredSpeak("Have a good Evening")
        else:
            listen.send_bluetooth("Goodnight.. Rest well Sir")
            speech.AlfredSpeak("Goodnight.. Rest well Sir")
        quit()

    def SecretaryWishme_end(self, AlfredQueryOffline):
        ("signing off")
        hour = datetime.datetime.now().hour
        if (hour >= 6 and hour < 12):
            SecretarySpeak("Have a good Morning")
        elif (hour >= 12 and hour < 18):
            SecretarySpeak("Have a good afternoon")
        elif (hour >= 18 and hour < 24):
            SecretarySpeak("Have a good Evening")
        else:
            SecretarySpeak("Goodnight.. Rest well Sir")
        quit()


    def sendEmail(self, to, content):

        email_username = os.environ.get('EMAIL_HOST_USER')
        email_password = os.environ.get('EMAIL_HOST_PASSWORD')
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(email_username, email_password)
        server.sendmail("tjaartcronje@gmail.com", to, content)
        server.close()

        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            
            pass
        
    #screenshot function
    def screenshot(self, AlfredQueryOffline):
        img = pyautogui.screenshot()
        img.save(
            Alfred_config.DRIVE_LETTER+"Users\\Alfred-AI-using-python3-\\screenshots\\ss.png"
        )
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass
        

    #battery and cpu usage
    def cpu(self, AlfredQueryOffline):

        usage = str(psutil.cpu_percent())
        speech.AlfredSpeak('CPU usage is at ' + usage)
        print('CPU usage is at ' + usage)
        battery = psutil.sensors_battery()
        listen.send_bluetooth(f"The battery is at {battery.percent}")
        speech.AlfredSpeak("Battery is at")
        speech.AlfredSpeak(battery.percent)
        print("battery is at:" + str(battery.percent))
       
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass
        

    #joke function
    def jokesAll(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        j = pyjokes.get_joke(language="en", category="all")

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I told you that {j}"
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Something to laugh about...", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response

        
        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

        print(j)
        listen.send_bluetooth(j)
        speech.AlfredSpeak(j)



##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass

    def jokesTwister(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        j = pyjokes.get_joke(language="en", category="twister")

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I told you that {j}"
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Twister Jokes", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response

        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

       
        print(j)
        listen.send_bluetooth(j)
        speech.AlfredSpeak(j)
        
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass




    def jokesNeutral(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        j = pyjokes.get_joke(language="en", category="neutral")

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I told you that {j}"
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Neutral  Jokes", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response

        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")


##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "
        
        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

        print(j)
        listen.send_bluetooth(j)
        speech.AlfredSpeak(j)
        

##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass




    def jokesChuck(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        j = pyjokes.get_joke(language="en", category="chuck")

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I told you that {j}"
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Chuck Norris Jokes", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response

        
        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "
        
        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

    
        print(j)
        listen.send_bluetooth(j)
        speech.AlfredSpeak(j)



##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
                print('')
        except:
            pass


    def JokesAllPlenty(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue

        speech.AlfredSpeak("Here is 15 cool jokes ,")
        All_Jokes_Repeat = ""

        print("Generating jokes...")

        jokes = []
        for i in range(15):
            Single_joke =  pyjokes.get_joke(language="en", category="all")
            jokes.append(Single_joke)
            print(f"{i + 1}: {Single_joke}")
            
        print(f" jokes : {jokes}")
        All_Jokes_Repeat = str(jokes)
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I told you that {All_Jokes_Repeat}"
        print(f"Alfred_Repeat_Previous_Response: {All_Jokes_Repeat}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "All cool Jokes", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
 
        print(f"All_Jokes_Repeat Last : {All_Jokes_Repeat}")

        
        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        print("Finished with jokes...")


        listen.send_bluetooth(All_Jokes_Repeat)
        speech.AlfredSpeak(All_Jokes_Repeat)


    def JokesAllSad(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        
        speech.AlfredSpeak("Here is some cool jokes ,")

        jokes = []
        
        for i in range(15):
            Single_joke =  pyjokes.get_joke(language="en", category="sad")
            jokes.append(Single_joke)
            print(f"{i + 1}: {Single_joke}")
            
        print(f" jokes : {jokes}")
        All_Jokes_Repeat = str(jokes)

        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I told you that {All_Jokes_Repeat}"
        print(f"Alfred_Repeat_Previous_Response: {All_Jokes_Repeat}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "All Sad Jokes", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
 
        print(f"All_Jokes_Repeat Last : {All_Jokes_Repeat}")

        listen.send_bluetooth(All_Jokes_Repeat)
        
        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        speech.AlfredSpeak(All_Jokes_Repeat)

        print("Finished with jokes...")


    def facts(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        
        speech.AlfredSpeak("Here is 15 interresting facts ,")

        facts = []
        
        for i in range(15):
            Single_Fact =  randfacts.get_fact(False)
            facts.append(Single_Fact)
            print(f"{i + 1}: {Single_Fact}")
            
        print(f" facts : {facts}")
        All_Facts_Repeat = str(facts)
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
                print('')
        except:
            pass

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I told you that {All_Facts_Repeat}"
        print(f"Alfred_Repeat_Previous_Response: {All_Facts_Repeat}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "15 Random Facts...", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response

        
        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")
 
        print(f"All_Facts_Repeat Last : {All_Facts_Repeat}")

        listen.send_bluetooth(All_Facts_Repeat)
        speech.AlfredSpeak(All_Facts_Repeat)
        
        print("Finished with Facts...")


    def fact(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
##        speech.AlfredSpeak("Hmm... Did you you know that...")
        New_Repeat_Facts_Appended = ""

        ft=randfacts.get_fact(False)
        print(ft)
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I told you that {ft}"
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Something interresting", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response

        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


 
        speech.AlfredSpeak(f"Hmmm... Did you you know that...{ft}")
        listen.send_bluetooth(f"Hmmm... Did you you know that...{ft}")
        
##        listen.send_bluetooth(ft)
##        speech.AlfredSpeak(ft)
        
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

    def setup_vosk_model(self):
        vosk_model_path = Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"
        return Model(vosk_model_path)

    def setup_text_to_speech(self):
        engine = pyttsx3.init()
        return engine


            # PDF Bible reader
            
    def listen_to_audio_Bible(NumberPages):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            speech.AlfredSpeak("Welcome to Bible reader sir")
            print("Welcome to Bible reader sir")
            speech.AlfredSpeak(f"from page 2 to page {NumberPages} or the next page,")
            speech.AlfredSpeak("wich page would you like to listen to")
            print("wich page would you like to listen to, sir")        
            print("Bible Listening...")
            speech.AlfredSpeak("Bible Listening...")
            recognizer.adjust_for_ambient_noise(sourc, duration=0.5)
            audio = recognizer.listen(source, timeout=15)
            print("Recognizing...")

        return audio


    def listen_to_audio_News(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            speech.AlfredSpeak("Welcome to News reader sir")
            print("Welcome to News reader sir")
            speech.AlfredSpeak("what would you like to know about the news")
            speech.AlfredSpeak("sport, space, finances, technolegy")      
            print("News listening...")
            speech.AlfredSpeak("News listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=15)
            print("Recognizing...")

        return audio

    def listen_to_audio_Wiki(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            speech.AlfredSpeak("Welcome to wikipedia reader")
            print("Welcome to wikipedia reader sir")
            speech.AlfredSpeak("what would you like to know, sir")
            print("wiki Listening...")
            speech.AlfredSpeak("wiki listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=15)
            print("Recognizing...")

        return audio


    def listen_to_audio_Calculator(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            speech.AlfredSpeak("Welcome to Calculator reader sir")
            print("Welcome to Calculator reader")
            speech.AlfredSpeak("what would you like to calculate, sir")      
            print("Calculator istening...")
            speech.AlfredSpeak("Calculator Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=15)
            print("Recognizing...")

        return audio

    def listen_to_audio_Reminder(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            speech.AlfredSpeak("What would you want the reminder to be about or are you done...?")
            print("What would you want the reminder to be about or are you done...?") 
            speech.AlfredSpeak("Reminder listening...")
            print("Reminder listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=15)

            print("Recognizing...")

        return audio


    def listen_to_audio_Door(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            speech.AlfredSpeak ("Excuse me...?")
            speech.AlfredSpeak ("Who is at the door ...?")
            speech.AlfredSpeak ("Please state your name and surname only")
            speech.AlfredSpeak ("Listening ...?")
            print("Excuse me, Who is at the door. listening ...?") 
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=15)

            print("Recognizing...")

        return audio


    def listen_to_audio_Alfred(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("listening ...?", end = "\r")
            print('\033c', end = '')        
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            print("Recognizing...", end = "\r")
            print('\033c', end = '')        

        return audio

    def recognize_speech(vosk_model, audio):
        recognizer = KaldiRecognizer(vosk_model, audio.sample_rate)
        recognizer.AcceptWaveform(audio.frame_data)
        result = recognizer.FinalResult()
        result=json.loads(result)

        return result["text"]
       

    def pdf_Bible_Chapter_reader(self, AlfredQueryOffline):

        try:

            speech.AlfredSpeak("Welcome to Bible chapter reader sir")
            print("Welcome to Bible chapter reader sir")

            global AddPage
            global PreviousPage
            
            reader = PyPDF2.PdfReader(Alfred_config.DRIVE_LETTER+'Python_Env//New_Virtual_Env//Project_Files//My PDF Books//2024 02 10  17h00  The Bible//NIV-Bible-PDF.pdf')

            listen.send_bluetooth("What Chapter would you like sir")
            speech.AlfredSpeak("What Chapter would you like sir")

            if rec.AcceptWaveform(data):
                result=rec.Result()
                result=json.loads(result)
                AlfredQueryOffline = result['text']
                print('sir: ' + AlfredQueryOffline)

                AlfredQuery = AlfredQueryOffline

                string = AlfredQuery
                print("string : " , string)

                try:
                    
                    pattern = string
                    print("pattern : " , pattern)

                    object = PyPDF2.PdfReader(reader)
                    numPages = len(reader.pages)

                    for i in range(0, numPages):
                        pageObj = object.getPage(i)
                        text = pageObj.extractText()
                       
                        for match in re.finditer(pattern, text):
                            print(f'Page no: {i} | Match: {match}')
                
                except Exception as e:
                    print("Error with the request; {0}".format(e))
                    return pdf_Bible_Chapter_reader()

                if ("cool" in string or "thank you" in string or "amen" in string
                      or "good bye" in string):
                    
                    speech.AlfredSpeak("Bible session ended. Have an awesome day and carry the word of God")
       
##                    listen.send_bluetooth("Listening...")
        ##            speech.AlfredSpeak("Listening...")

                    main.main(assistant, gui)        

        except Exception as e:
            print("Error with the request; {0}".format(e))
            return pdf_Bible_Chapter_reader()


    def pdf_Bible_Page_reader(self, AlfredQueryOffline):

        duration = 5.0
        global AddPage
        global PreviousPage
        
        reader = PyPDF2.PdfReader(Alfred_config.DRIVE_LETTER+'Python_Env//New_Virtual_Env//Project_Files//My PDF Books//2024 02 10  17h00  The Bible//NIV-Bible-PDF.pdf')
       
        NumberPages =  len(reader.pages)
        print("Total no. of pages : " , NumberPages)

        Serial_Bluetooth.flushInput()
        Bluetooth_RX = Serial_Bluetooth.readline()

        print(Bluetooth_RX)

        command = Bluetooth_RX
        command = str(command, 'utf-8')    
        print("command : " ,command)
        
        AlfredQueryOffline = command   
        AlfredQueryOfflineNextPage = str(AlfredQueryOffline)

        NewPreviousPage = PreviousPage
        print("NewPreviousPage:", NewPreviousPage)

        if ("next" in AlfredQueryOfflineNextPage or "next page" in AlfredQueryOfflineNextPage or "another" in AlfredQueryOffline
              or "another page" in AlfredQueryOfflineNextPage):

            speech.AlfredSpeak("ok next page")

            AddPage = AddPage + 1

            NewPageNumberToRead = NewPreviousPage + AddPage

            NewPageNumberToReadAdded = str(NewPageNumberToRead)
            
            print("PreviousPage : ", str(NewPreviousPage))
            print("PageNumberToRead : ", str(NewPageNumberToRead)) 
            
            ReadableText = reader.pages[int(NewPageNumberToRead)].extract_text()
            modified_string = ''.join(filter(lambda z: not z.isdigit(), ReadableText))
            re.sub(r"\s+", " ", modified_string)

            modified_string = modified_string.replace("\r","").replace("\n","")
            modified_string =  ''.join(modified_string)

            modified_string.split(".")

            print(modified_string)    
            listen.send_bluetooth(modified_string)
            speech.AlfredSpeak(modified_string)

            return pdf_Bible_Page_reader()


        elif ("cool" in AlfredQueryOfflineNextPage or "thank you" in AlfredQueryOfflineNextPage or "amen" in AlfredQueryOffline
              or "good bye" in AlfredQueryOfflineNextPage):
            speech.AlfredSpeak("Bible session ended. Have an awesome day and carry the word of God")

            main.main(assistant, gui)        

        Input = AlfredQueryOffline

        CheckLeftWord = 'page'
        CheckRightWord = 'please'

        index1 = Input.find(CheckLeftWord)
        index2 = Input.find(CheckRightWord)
        
        BibleAlfredQueryOffline = Input [index1 +len(CheckLeftWord) + 1: index2]
        print("BibleAlfredQueryOffline : " + BibleAlfredQueryOffline)

        print("numbers in string : " + str(w2n.word_to_num(BibleAlfredQueryOffline)))
        Number_String = str(w2n.word_to_num(BibleAlfredQueryOffline)) 

        Result2 = str(Number_String)
        print("Result2 : " + Result2)

        if Result2.isdigit(self):

            numbers = Number_String
            resultIsNum = numbers

            print("After extracting numbers from a new_string:", resultIsNum)
          
            PageNumberToRead = resultIsNum 
            PreviousPage = (str(PageNumberToRead))
            NewPreviousPage = PreviousPage
            
            print("Selected Page number is", PageNumberToRead)
            print("Selected PreviousPage number is", PreviousPage)
            print("Selected NewPreviousPage number is", NewPreviousPage)

            ReadableText = reader.pages[int(PageNumberToRead)].extract_text()
            modified_string = ''.join(filter(lambda z: not z.isdigit(), ReadableText))
            re.sub(r"\s+", " ", modified_string)
            New_Modified_string = modified_string
            New_Modified_string = New_Modified_string.replace("\r","").replace("\n","")
            New_Modified_string =  ''.join(New_Modified_string)

            New_Modified_string.split(".")

            print(New_Modified_string)    
            listen.send_bluetooth(New_Modified_string)
            speech.AlfredSpeak(New_Modified_string)

            return pdf_Bible_Page_reader()
                    
        return pdf_Bible_Page_reader()


    def Calculator(self):

        Serial_Bluetooth.flushInput()
        Bluetooth_RX = Serial_Bluetooth.readline()

        print(Bluetooth_RX)

        command = Bluetooth_RX
        command = str(command, 'utf-8')    
        print("command : " ,command)
        
        AlfredQueryOffline = command

        CalculatorQuery = AlfredQueryOffline

        print("You said: " + CalculatorQuery)

        Input = CalculatorQuery
        Input = CalculatorQuery.replace(",", "")

        CheckLeftWord = 'calculate'
        CheckRightWord = 'please'

        indexLeftCalc = Input.find(CheckLeftWord)
        indexRightCalc = Input.find(CheckRightWord)
        
        NewCalculatorQuery = Input[indexLeftCalc +len(CheckLeftWord) + 1: indexRightCalc]
        print("NewCalculatorQuery : " + NewCalculatorQuery) 

        NewCalculatorQueryFiltered = NewCalculatorQuery.replace("million","000000")
        print("NewCalculatorQueryFiltered : " , NewCalculatorQueryFiltered)

        NewCalculatorQueryNoSpace = NewCalculatorQueryFiltered.replace(" ", "")
        print("NewCalculatorQueryNoSpace : " , NewCalculatorQueryNoSpace)

        NewCalculatorQueryNewX = NewCalculatorQueryNoSpace.replace("x", " x ")    
        print("NewCalculatorQueryNewX : " , NewCalculatorQueryNewX)

        def get_operator_fn(oper):
            return {
                '+' : operator.add,
                '-' : operator.sub,
                'x' : operator.mul, 
                '*' : operator.mul,
                'divided' :operator.__truediv__, 
                '/' :operator.__truediv__ ,
                'Mod' : operator.mod,
                'mod' : operator.mod,
                '^' : operator.xor,
                }[oper]

        def eval_binary_expr(op1, oper, op2):
            try:
                op1,op2 = int(op1), int(op2)
                return get_operator_fn(oper)(op1, op2)
            except sr.RequestError as e:
                print("Error with the request; {0}".format(e))
            
            return Calculator()

        try: 
            print(eval_binary_expr(*(NewCalculatorQueryNewX.split())))
            
            MathAnswer = (eval_binary_expr(*(NewCalculatorQueryNewX.split())))
            print("Answer : ",  MathAnswer)
            CalculatorSpeak("Your answer is : ")
            CalculatorSpeak(MathAnswer)

            CalculatorSpeak("Is there any other equation you want me to do sir?")
            CalculatorQuery = Calculator_Speech_to_Text().lower()
            
            if ("yes" in CalculatorQuery or "yip" in CalculatorQuery):
                return Calculator()
            
            elif ("no" in CalculatorQuery or "nope" in CalculatorQuery):
                return main.main(assistant, gui)

        except Exception as e:
            print("Error with the request; {0}".format(e))
            print('''I am sorry I didn't get that, Please say that again.''')

            return Calculator()
        
    def crops(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        resCrops = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Crops.txt")
          
        try :
            if MyToDoListEdited:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
                print('')
        except:
            pass

        global Alfred_Repeat_Previous_Response

        resCrops_Repeat = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Crops.txt")

        print("")
        print("About me...Repeat")
        print("")

        Alfred_Repeat_Previous_Response = (resCrops_Repeat.read())
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Something interresting about the crops...", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
 
        print("")
        print("About me...Repeat...Finished")
        print("")

        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        listen.send_bluetooth(resCrops.read())
        speech.AlfredSpeak(resCrops.read())
        
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")


    def YourAge(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue

        year = int(datetime.datetime.now().year)
        month = int(datetime.datetime.now().month)
        date = int(datetime.datetime.now().day)
        day = int (datetime.datetime.now().weekday())

        if (day == 0):
            day = 'Monday'
        elif (day == 1):
            day = 'Tuesday'
        elif (day == 2):
            day = 'Wednessday'
        elif (day == 3):
            day = 'Thursday'
        elif (day == 4):
            day = 'Friday'
        elif (day == 5):
            day = 'Saterday'
        elif (day == 6):
            day = 'Sunday'

        birthdate = "18-12-2023"
        Day,Month,Year = map(int, birthdate.split("-"))
        today = datetime.date.today()
        age = today.year - Year - ((today.month, today.day) < (Month, Day))

        print("month :",month)
        current_Month = 12 - month

        print("I am ",age,"years and",month,"months old.")

        My_age = (f"I am {age}, years and {month}, months old.")

        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I told you that {My_age}"
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        print("_____")
        print(My_age)
        print("_____")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "What is your age?", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response

        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "

        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
                print('')
        except:
            pass
 
        listen.send_bluetooth(f"{My_age} I was born inDecember 2023")
        speech.AlfredSpeak(My_age)
        speech.AlfredSpeak("I was born in December 2023")
        

    def personal(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        
        resPersonal = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\personal2.txt")

        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
                print('')
        except:
            pass

        global Alfred_Repeat_Previous_Response

        resPersonal_Repeat = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\personal2.txt")

        print("")
        print("About me...Repeat")
        print("")
        
        About_Me = resPersonal.read()

        Alfred_Repeat_Previous_Response = About_Me
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Tell me about Yourself...", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
  
        print("")
        print("About me...Repeat...Finished")
        print("")


        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "

        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "
        
        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")


        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        
        listen.send_bluetooth(About_Me)
        speech.AlfredSpeak(About_Me)
    
    def made(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        ##        global AlfredQueryOffline

        
        global log_queue
        print("How I was Made...")
        
        resMade = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\HowYouMade.txt")
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
                print('')
        except:
            pass

        global Alfred_Repeat_Previous_Response

        resMade_Repeat = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\HowYouMade.txt")

        print("")
        print("How I was Made...Repeat")
        print("")

        Alfred_Repeat_Previous_Response = (resMade_Repeat.read())
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "Who made you...?", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
 
        print("")
        print("How I was Made...Repeat...Finished")
        print("")

        Who_Made_Me = resMade.read()
    
        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "

        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

        
        listen.send_bluetooth(Who_Made_Me)
        speech.AlfredSpeak(Who_Made_Me)
            
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        
    def what(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        resWhat = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\whatRU.txt")

        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
                print('')
        except:
            pass

        global Alfred_Repeat_Previous_Response

        resWhat_Repeat = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\whatRU.txt")

        print("")
        print("How I was Made...Repeat")
        print("")

        Alfred_Repeat_Previous_Response = (resWhat_Repeat.read())
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "What are you...?", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
 
        print("")
        print("How I was Made...Repeat...Finished")
        print("")

        What_I_Am = resWhat.read()

        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "

        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")

        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        listen.send_bluetooth(What_I_Am)
        speech.AlfredSpeak(What_I_Am)
        
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")


    def name(self, AlfredQueryOffline):

        print(f"AlfredQueryOffline : {AlfredQueryOffline}")
        # call the method on this instance (important)
        message, username = self.assistant_extract_text_from_query(AlfredQueryOffline)
        print(f"NEW  message : {message!r}")
        print(f"NEW  username: {username!r}")

        
        global log_queue
        
        resName = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Name.txt")
            
        try :
            if MyToDoListEdited != []:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("")
                print("AlfredQueryOffline New : " + str(AlfredQueryOfflineNew))
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass

        global Alfred_Repeat_Previous_Response

        resName_Repeat = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Name.txt")

        print("")
        print("How I was Made...Repeat")
        print("")

        Alfred_Repeat_Previous_Response = (resName_Repeat.read())
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "What is your Name...?", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
 
        print("")
        print("How I was Made...Repeat...Finished")
        print("")

        What_Is_My_Name = resName.read()

        # GUI log if available
        query_msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "

        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "
        
        print(f"[DEBUG ASSISTANT query_msg] : {query_msg}")
        print(f"[DEBUG ASSISTANT query_resp] : {query_resp}")


        try:
            self.gui.log_message(query_msg)
            self.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

        listen.send_bluetooth(What_Is_My_Name)
        speech.AlfredSpeak(What_Is_My_Name)    
                

##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")
                     
    def Calculator_Speech_to_Text(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 4000

        with sr.Microphone() as source:

            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Adjusting for ambient noise...")
            print("Calculator listening...")
            speech.AlfredSpeak("Calculator listening...")

            audio = recognizer.listen(source,phrase_time_limit=10)

        try:
            print("You said:", recognizer.recognize_google(audio, language='en-us'))
            speech.AlfredSpeak("ok sir...")

            return recognizer.recognize_google(audio, language='en-us')
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return "None"
        except sr.RequestError as e:
            print("Error with the request; {0}".format(e))
            return "None"


    def Alfred_Greeting_Speech_to_Text(self):

        cnt = 1
        r = sr.Recognizer()
        r.energy_threshold = 600.4452854381937  #400
        
        global Alfred_No_Ollama

        print("Alfred_No_Ollama: " , Alfred_No_Ollama)
                
        with sr.Microphone() as audio:

            r.adjust_for_ambient_noise(audio, duration=1)
            print("Adjusting for ambient noise...")
            print("How are you doing...sir")
            speech.AlfredSpeak("How are you doing...sir") 
            audio = r.record(audio, offset=1, duration=5)
        try:

            print("ok sir...")
            AlfredQuery = r.recognize_google(audio,language='en-us')

            print("Transcription: " + AlfredQuery) 
     
        except Exception as e :

            Alfred_No_Ollama = Alfred_No_Ollama + 1
            
            print("Excuse me sir, please say that again...")
            speech.AlfredSpeak("Excuse me sir, please say that again...")
            
            if (Alfred_No_Ollama == 10):
                print("Alfred_No_Ollama: " , Alfred_No_Ollama) 
                facts()

            if (Alfred_No_Ollama == 20):
                print("Alfred_No_Ollama: " , Alfred_No_Ollama) 
                JokesAllPlenty()

            if (Alfred_No_Ollama == 21):
                Alfred_No_Ollama = 0
            
            return "none"

        return AlfredQuery

    ###########################################################################
    ##                  DO HISTORY

    def Previous_Do_History(self, promptDidEdited, Did_History):
        global log_queue

        NewPromptEdited_Previous_Did = promptDidEdited

        NewPromptEdited_DidSpeak = NewPromptEdited_Previous_Did

        New_Did_String = str(Did_History)
        
        print("")
        print(f"New_Did_String : {New_Did_String}")
        print("")
        
        speech.AlfredSpeak (New_Did_String)
       
        listen.send_bluetooth("Listening...")
        GUI.log_message(f"You Asked: What did you do and I replied : {New_Did_String}")
##        speech.AlfredSpeak("Listening...")

        global Alfred_Repeat_Previous_Response

        resName_Repeat = open(Alfred_config.DRIVE_LETTER+"Python_Env\\New_Virtual_Env\\Personal\\Name.txt")

        print("")
        print("How I was Made...Repeat")
        print("")

        Alfred_Repeat_Previous_Response = (Did_History)
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "What else did you do...?", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
   
        print("")
        print("How I was Made...Repeat...Finished")
        print("")
       
    ##########################################################################

    def Do_History_Function(self, promptEdited, Do_History):

        global log_queue

        NewPromptEdited_Doing = promptEdited

        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("something", "")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("please", "")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("tell me about", "I am telling you about")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("tell me", "I am telling you")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("what is", "what was")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("yourself", "my self")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("about you", "about me")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("who are you", "I am telling you who am I")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("what are you", "I am telling you what I am")

        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("tell me", "telling you")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("me", "you")
        NewPromptEdited_Doing = NewPromptEdited_Doing.replace("create", "created")
                               
        print('\n')
        print('\r')
        print("NewPromptEdited_Doing : " , NewPromptEdited_Doing)

        NewPromptEdited_DoingSpeak = NewPromptEdited_Doing

        print("")
        print("About me...Repeat")
        print("")

        Alfred_Repeat_Previous_Response = (NewPromptEdited_DoingSpeak)
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "What are you doing...?", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
 
        print("")
        print("About me...Repeat...Finished")
        print("")

        listen.send_bluetooth(NewPromptEdited_DoingSpeak)
        GUI.log_message(f"You Asked: What are you doing and I replied : {New_Did_String}")
        speech.AlfredSpeak(NewPromptEdited_DoingSpeak)
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        main.main(assistant, gui)
        
    AlfredQueryOffline = ""

    print("assistant end")

assistant = AlfredAssistant(gui)


