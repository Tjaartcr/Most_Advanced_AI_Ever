import os
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

app = Flask(__name__, static_folder=r"C:/Python_Env/New_Virtual_Env/sketch_may09bAlfred_Offline_New_GUI\2025_06_15a_GUI_Serial_WEBUI_Laptop\New_V2_Home_Head_Movement_Smoothing/modules/webui-src/build", static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*")
