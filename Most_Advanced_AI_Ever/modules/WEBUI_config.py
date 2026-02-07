



import requests
import json
import os
##from get_ip_addresses import ArduinoCommunicationIPModule

# ==========================
# CONFIGURATION PARAMETERS
# ==========================

KEYWORD = "I_AM_HOME_AUTOMATION_001"
TARGET_URL = "/json"  # Path on the ESP
TARGET_URL_HTML = "/html"
COM_PORT = "COM3"
SERIAL_PORT_BLUETOOTH = "COM5"
BAUDRATE_BLUETOOTH = 38400
SERIAL_PORT_ARDUINO = COM_PORT
BAUDRATE_ARDUINO = 9600
SERIAL_PORT_ARDUINO_WEBCAMS = COM_PORT
BAUDRATE_ARDUINO_WEBCAM_IP = 9600
DRIVE_LETTER = "D://"

##IP_MEMORY_FILE = "previous_ip.json"
##HOME_AUTOMATION_IP = None
##
##
##
##from dotenv import load_dotenv
##
##load_dotenv()
##
##
##
##EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
##EMAIL_HOST = 'smtp.gmail.com'
##EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
##EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
##EMAIL_PORT = '587'
##EMAIL_USE_TLS = True
##EMAIL_USE_SSL = False
##
##print(f"Host User is : {EMAIL_HOST_USER}")
##print(f"Password is : {EMAIL_HOST_PASSWORD}")


YOLO_MODEL_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
FACE_RECOGNITION_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
VOSK_MODEL_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"



