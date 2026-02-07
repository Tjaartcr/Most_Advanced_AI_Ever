

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


from dotenv import load_dotenv

load_dotenv()

### Build paths inside the project like this: BASE_DIR / 'subdir'.
##BASE_DIR = Path(__file__).resolve().parent.parent
##
##
### Quick-start development settings - unsuitable for production
### See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/
##
### SECURITY WARNING: keep the secret key used in production secret!
##SECRET_KEY = os.environ.get('SECRET_KEY')

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
EMAIL_PORT = '587'
EMAIL_USE_TLS = True
EMAIL_USE_SSL = False


print("assistant")

class AlfredAssistant:

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


    def handle_send_email(self):

        while True:
            
            speech.AlfredSpeak("To whom would you like to send the email, sir?")
            query_email_to = listen.listen()
            if not query_email_to:
                continue
            
            if "chart" or "boss" in query_email_to:
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

            # check for any affirmative
            affirmatives = {"yes", "correct", "awesome", "fantastic", "great"}
            negatives   = {"no", "nope", "incorrect", "false"}



##                EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
##                EMAIL_HOST = 'smtp.gmail.com'
##                EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
##                EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
##                EMAIL_PORT = '587'
##                EMAIL_USE_TLS = True
##                EMAIL_USE_SSL = False
##                                
##                server = smtplib.SMTP('smtp.gmail.com', 587)
##                server.ehlo()
##                server.starttls()
##                server.login("tjaartcronje@gmail.com", "Tjaart1234")
##                server.sendmail("tjaartcronje@gmail.com", to, content)
##                server.close()
                    

##                EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
##                EMAIL_HOST = 'smtp.gmail.com'
##                EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
##                EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
##                EMAIL_PORT = '587'
##                EMAIL_USE_TLS = True
##                EMAIL_USE_SSL = False


            #sending email function
            def sendEmail(to, content):

                Message_to_Send = (f".  To : {to}. Message : {content}   ")
                print(f"After : {Message_to_Send}")
                
                server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
                server.ehlo()
                server.starttls()
                server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
                server.sendmail("tjaartcronje@gmail.com", to, content)
                server.close()

                        

##            if any(word in reply for word in affirmatives):
            if "yes" or "correct" or "cool" or "awesome" or "fantastic":

                Message_to_Send = (f".  To : {to}. Message : {content}   ")
                print(f"Before : {Message_to_Send}")
                
                speech.AlfredSpeak("Email sent, sir.")
                sendEmail(to, content)
                break
            
##            elif any(word in reply for word in negatives):
            elif  "no" or "incorrect" or "not cool" or "wrong" or "nope":
                speech.AlfredSpeak("Okay, let's start over.")
                return  # exit the email handler
            
            else:
                speech.AlfredSpeak("I didn't catch that. Please say yes or no.")
                # loop again asking confirmation



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

    def tell_time(self, AlfredQueryOffline):
        """Speak the current time."""
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        listen.send_bluetooth(f"The time is {current_time}")
        speech.AlfredSpeak(f"The time is {current_time}")

    def tell_date(self, AlfredQueryOffline):
        """Speak the current date."""
        today = datetime.datetime.today()
        listen.send_bluetooth(f"Today's date is {today.strftime('%B %d, %Y')}")
        speech.AlfredSpeak(f"Today's date is {today.strftime('%B %d, %Y')}")

    def tell_joke(self, AlfredQueryOffline):
        """Tell a random joke."""
        joke = pyjokes.get_joke()
        listen.send_bluetooth(joke)
        speech.AlfredSpeak(joke)
       
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")


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
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
 

##        listen.send_bluetooth('listening ...')
##        speech.AlfredSpeak('listening ...')
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
       
    #time function
    def timeOfDay(self, AlfredQueryOffline):

        ##        global AlfredQueryOffline
        
        
        timeOfDay = datetime.datetime.now().strftime("%H:%M:%S")
        print(timeOfDay)
        
        global log_queue
        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I said the current time is {timeOfDay}"
        print(f"Alfred_Repeat_Previous_Response: {Alfred_Repeat_Previous_Response}")

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "The time of the day....", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
 
        listen.send_bluetooth(f"The current time is {timeOfDay}")
        speech.AlfredSpeak("The current time is")
        speech.AlfredSpeak(timeOfDay)
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
 

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
                print('')
        except:
            pass
      

    def date(self, AlfredQueryOffline):

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

        print(year,month,date,day)
            
        The_date = (year,month,date,day)
        
        global Alfred_Repeat_Previous_Response
        Alfred_Repeat_Previous_Response = f"I said the current date is {The_date}"
        print(Alfred_Repeat_Previous_Response)

        response = Alfred_Repeat_Previous_Response
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        chat_entry = {"date": current_date, "time": current_time, "query": "The current date...?", "response": response}
        
        memory.add_to_memory(chat_entry)  # ✅ Store in memory
        repeat.add_to_repeat(chat_entry)  # ✅ Store last response
        
        listen.send_bluetooth(f"The current date is {The_date}")
        speech.AlfredSpeak(f"The current date is {The_date}")
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
 
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
                print('')
        except:
            pass

        
    def day(self, AlfredQueryOffline):
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
        
        listen.send_bluetooth(f"The current day is {day}")
        speech.AlfredSpeak("The current day is")
        speech.AlfredSpeak(day)
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
 

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
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
        
##            listen.send_bluetooth("Listening...")
##            speech.AlfredSpeak("Listening...")

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
##            speech.AlfredSpeak_Onboard("Good afternoon sir")
            speech.AlfredSpeak("Good afternoon sir")

        elif (hour >= 18 and hour < 24):
##            speech.AlfredSpeak_Onboard("Good evening sir")
            speech.AlfredSpeak("Good evening sir!")

        else:
##            speech.AlfredSpeak_Onboard("Good evening sir")
            speech.AlfredSpeak("Good evening sir!")


##        speech.AlfredSpeak_Onboard("Please start the Home Automation Application on your Android device and connect via bluetooth....?")
##        print("Please start the Home Automation Application on your Android device and connect via bluetooth....?")

##        listen.send_bluetooth("Thank you, We are connected.... ")
##        speech.AlfredSpeak("Thank you, We are connected.... ")

##        print("Thank you, We are connected.... ")

        time.sleep(1.5)
        
        listen.send_bluetooth("Please, check your Query's and Responses on the graphics user interface, and our New, Web User Interface Running on our localhost, on Port, 5000")
        speech.AlfredSpeak("Please, check your Query's and Responses on the graphics user interface, and our New, Web User Interface Running on our localhost, on Port, 5000")

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

        listen.send_bluetooth("Welcome Back")
        speech.AlfredSpeak("Welcome Back")
        
        hour = datetime.datetime.now().hour
        if (hour >= 6 and hour < 12):
            listen.send_bluetooth("Good morning sir!")
            listen.send_bluetooth("Welcome back.")
            listen.send_bluetooth("what a wonderful morning it is")
            listen.send_bluetooth("hope you have a fantastic day")
            listen.send_bluetooth("how can i assist you, sir")

            speech.AlfredSpeak("Good morning sir!")
            speech.AlfredSpeak("Welcome back.")
            speech.AlfredSpeak("what a wonderful morning it is")
            speech.AlfredSpeak("hope you have a fantastic day")
            speech.AlfredSpeak("how can i assist you, sir")

        elif (hour >= 12 and hour < 18):
            listen.send_bluetooth("Good afternoon sir")
            listen.send_bluetooth("Welcome back.")
            listen.send_bluetooth("what a awesome day so far")
            listen.send_bluetooth("hope you have an awesome afternoon")
            listen.send_bluetooth("how can i assist you, sir")
            
            speech.AlfredSpeak("Good afternoon sir")
            speech.AlfredSpeak("Welcome back.")
            speech.AlfredSpeak("what a awesome day so far")
            speech.AlfredSpeak("hope you have an awesome afternoon")
            speech.AlfredSpeak("how can i assist you, sir")

        elif (hour >= 18 and hour < 22):
            listen.send_bluetooth("Good evening sir")
            listen.send_bluetooth("Welcome back.")
            listen.send_bluetooth("what a fantastic evening")
            listen.send_bluetooth("hope you had a fantastic day")
            listen.send_bluetooth("how can i assist you, sir")

            speech.AlfredSpeak("Good evening sir")
            speech.AlfredSpeak("Welcome back.")
            speech.AlfredSpeak("what a fantastic evening")
            speech.AlfredSpeak("hope you had a fantastic day")
            speech.AlfredSpeak("how can i assist you, sir")
                        
        elif (hour >= 22 and hour < 24):
            listen.send_bluetooth("Good evening sir")
            listen.send_bluetooth("Welcome back.")
            listen.send_bluetooth("what a fantastic evening")
            listen.send_bluetooth("it's very late, you should go to bed, sir")

            speech.AlfredSpeak("Good evening sir")
            speech.AlfredSpeak("Welcome back.")
            speech.AlfredSpeak("what a fantastic evening")
            speech.AlfredSpeak("it's very late, you should go to bed, sir")
                        
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


##    #sending email function

##    def sendEmail(to, content):
##        server = smtplib.SMTP('smtp.gmail.com', 587)
##        server.ehlo()
##        server.starttls()
##        server.login("tjaartcronje@gmail.com", "Tjaart1234")
##        server.sendmail("tjaartcronje@gmail.com", to, content)
##        server.close()


    def sendEmail(self, to, content):

        email_username = os.environ.get('EMAIL_HOST_USER')
        email_password = os.environ.get('EMAIL_HOST_PASSWORD')
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(email_username, email_password)
        server.sendmail("tjaartcronje@gmail.com", to, content)
        server.close()




##EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
##EMAIL_HOST = 'smtp.gmail.com'
##EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
##EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
##EMAIL_PORT = '587'
##EMAIL_USE_TLS = True
##EMAIL_USE_SSL = False



            
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

        print(j)
        listen.send_bluetooth(j)
        speech.AlfredSpeak(j)
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
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
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass


    def jokesTwister(self, AlfredQueryOffline):

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
       
        print(j)
        listen.send_bluetooth(j)
        speech.AlfredSpeak(j)
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
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
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass


    def jokesNeutral(self, AlfredQueryOffline):

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
            
        print(j)
        listen.send_bluetooth(j)
        speech.AlfredSpeak(j)
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
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
            ##    print(AlfredQueryOfflineNew)
                print('')
        except:
            pass


    def jokesChuck(self, AlfredQueryOffline):

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
    
        print(j)
        listen.send_bluetooth(j)
        speech.AlfredSpeak(j)
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
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


    def JokesAllPlenty(self, AlfredQueryOffline):

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

        listen.send_bluetooth(All_Jokes_Repeat)
        speech.AlfredSpeak(All_Jokes_Repeat)
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
 
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        print("Finished with jokes...")


    def JokesAllSad(self, AlfredQueryOffline):

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
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
 
        speech.AlfredSpeak(All_Jokes_Repeat)

        print("Finished with jokes...")


    def facts(self, AlfredQueryOffline):

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
 
        print(f"All_Facts_Repeat Last : {All_Facts_Repeat}")

        listen.send_bluetooth(All_Facts_Repeat)
        speech.AlfredSpeak(All_Facts_Repeat)
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
 
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        print("Finished with Facts...")


    def fact(self, AlfredQueryOffline):

        global log_queue
        speech.AlfredSpeak("Here is an interresting fact, for you ,")
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
 
        listen.send_bluetooth(ft)
        speech.AlfredSpeak(ft)
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
 
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

        listen.send_bluetooth(resCrops.read())
        speech.AlfredSpeak(resCrops.read())
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
 
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")


    def YourAge(self, AlfredQueryOffline):

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
 
        listen.send_bluetooth(f"{My_age} I was born inDecember 2023")
        speech.AlfredSpeak(My_age)
        speech.AlfredSpeak("I was born in December 2023")
        
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
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


    def personal(self, AlfredQueryOffline):

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

        
        listen.send_bluetooth(About_Me)
        speech.AlfredSpeak(About_Me)
    
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
        
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")


    def made(self, AlfredQueryOffline):

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
        
        listen.send_bluetooth(Who_Made_Me)
        speech.AlfredSpeak(Who_Made_Me)
                
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
        
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        
    def what(self, AlfredQueryOffline):

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

        listen.send_bluetooth(What_I_Am)
        speech.AlfredSpeak(What_I_Am)
                
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
        
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")


    def name(self, AlfredQueryOffline):

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

        listen.send_bluetooth(What_Is_My_Name)
        speech.AlfredSpeak(What_Is_My_Name)    
                
        query_msg = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
        
        try:
            assistant.gui.log_message(query_msg)
        except NameError:
            print("GUI instance not available for logging message.")
        
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
