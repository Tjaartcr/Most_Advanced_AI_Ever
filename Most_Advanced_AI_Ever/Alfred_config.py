



import requests
import json
import os
from get_ip_addresses import ArduinoCommunicationIPModule

# ==========================
# CONFIGURATION PARAMETERS
# ==========================

KEYWORD = "I_AM_HOME_AUTOMATION_001"
TARGET_URL = "/json"  # Path on the ESP
TARGET_URL_HTML = "/html"
COM_PORT = "COM6"
SERIAL_PORT_BLUETOOTH = "COM5"
BAUDRATE_BLUETOOTH = 38400
SERIAL_PORT_ARDUINO = COM_PORT
BAUDRATE_ARDUINO = 9600
SERIAL_PORT_ARDUINO_WEBCAMS = COM_PORT
BAUDRATE_ARDUINO_WEBCAM_IP = 9600
DRIVE_LETTER = "D://"

IP_MEMORY_FILE = "previous_ip.json"
HOME_AUTOMATION_IP = None



from dotenv import load_dotenv

load_dotenv()



EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
EMAIL_PORT = '587'
EMAIL_USE_TLS = True
EMAIL_USE_SSL = False

print(f"\n Host User is : {EMAIL_HOST_USER}")
print(f"\n Password is : {EMAIL_HOST_PASSWORD}")

current_directory = os.getcwd()
print(f"\n current_directory is : {current_directory} \n")


# ==========================
# FUNCTION DEFINITIONS
# ==========================

def save_ip(ip):
    with open(IP_MEMORY_FILE, 'w') as f:
        json.dump({"last_known_ip": ip}, f)

def load_ip():
    if os.path.exists(IP_MEMORY_FILE):
        with open(IP_MEMORY_FILE, 'r') as f:
            data = json.load(f)
            return data.get("last_known_ip")
    return None

def test_ip(ip):
    url = f"http://{ip}/json"
    try:
        print(f"➡️  Checking {url}...", end=" ")
        response = requests.get(url, timeout=2.5)
        print(f" Response : {response}")

        if response.ok:
            data = response.json()
            if data.get("device") == KEYWORD:
                print(f"✅ FOUND device '{KEYWORD}' at {ip}")
                return True
            else:
                print(f"❌ Wrong device: {data.get('device')}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
    except requests.exceptions.Timeout:
        print("⏱️ Timeout")
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    return False

# ==========================
# IP DETECTION LOGIC
# ==========================

print("📁 Trying previously saved IP...")
last_ip = load_ip()
if last_ip and test_ip(last_ip):
    HOME_AUTOMATION_IP = last_ip
else:
    print("🔍 Scanning network for HOME_AUTOMATION_IP...")
    for i in range(0, 256):
        ip = f"192.168.{i}.146"
        if test_ip(ip):
            HOME_AUTOMATION_IP = ip
            save_ip(ip)
            break

# Use fallback if nothing was found
if HOME_AUTOMATION_IP is None:
    HOME_AUTOMATION_IP = "192.168.138.146"  # Fallback IP
    print("⚠️ Failed to auto-detect. Using fallback IP:", HOME_AUTOMATION_IP)
else:
    print("🎯 Success! Device found at IP:", HOME_AUTOMATION_IP)

# ==========================
# GET CAMERA INPUT IPs
# ==========================

ip_module = ArduinoCommunicationIPModule(port=COM_PORT, baudrate=BAUDRATE_ARDUINO)
ips = ip_module.get_ip_addresses()

LEFT_EYE_CAMERA_INPUT_NEW = ips.get("left") if ips else None
RIGHT_EYE_CAMERA_INPUT_NEW = ips.get("right") if ips else None

print('\n')
print("HOME_AUTOMATION_IP:", HOME_AUTOMATION_IP)
print("LEFT_EYE_CAMERA_INPUT_NEW:", LEFT_EYE_CAMERA_INPUT_NEW)
print("RIGHT_EYE_CAMERA_INPUT_NEW:", RIGHT_EYE_CAMERA_INPUT_NEW)
print('\n')

# ==========================
# ASSIGN CAMERA INPUTS
# ==========================

##LEFT_EYE_CAMERA_INPUT = LEFT_EYE_CAMERA_INPUT_NEW
##CHEST_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##RIGHT_EYE_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW

LEFT_EYE_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
CHEST_CAMERA_INPUT = LEFT_EYE_CAMERA_INPUT_NEW
RIGHT_EYE_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW

# ==========================
# PATHS FOR MODELS
# ==========================

YOLO_MODEL_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
FACE_RECOGNITION_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
VOSK_MODEL_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"



    # NEW AUTO CHEC IP'S LEFT AND RIGHT

    

##import requests
##import json
##import os
##from get_ip_addresses import ArduinoCommunicationIPModule
##
### ==========================
### CONFIGURATION PARAMETERS
### ==========================
##KEYWORD_HOME      = "I_AM_HOME_AUTOMATION_001"
##KEYWORD_LEFT_EYE  = "I_AM_LEFT_EYE_001"
##KEYWORD_RIGHT_EYE = "I_AM_RIGHT_EYE_001"
##
##TARGET_URL = "/json"  # Path on the ESP
##
### Serial / Arduino settings (unchanged)
##COM_PORT = "COM6"
##SERIAL_PORT_BLUETOOTH = "COM5"
##BAUDRATE_BLUETOOTH = 38400
##SERIAL_PORT_ARDUINO = COM_PORT
##BAUDRATE_ARDUINO = 9600
##SERIAL_PORT_ARDUINO_WEBCAMS = COM_PORT
##BAUDRATE_ARDUINO_WEBCAM_IP = 9600
##DRIVE_LETTER = "D://"
##
### Memory files
##HOME_IP_FILE       = "previous_ip.json"
##LEFT_EYE_IP_FILE   = "previous_ip_left_eye.json"
##RIGHT_EYE_IP_FILE  = "previous_ip_right_eye.json"
##
### Fallbacks
##FALLBACK_HOME_IP       = "192.168.138.146"
##FALLBACK_LEFT_EYE_IP   = "192.168.138.147"
##FALLBACK_RIGHT_EYE_IP  = "192.168.138.148"
##
##from dotenv import load_dotenv
##load_dotenv()
##
##EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
##EMAIL_HOST = 'smtp.gmail.com'
##EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
##EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
##EMAIL_PORT = '587'
##EMAIL_USE_TLS = True
##EMAIL_USE_SSL = False
##
##print(f"\n Host User is : {EMAIL_HOST_USER}")
##print(f"\n Password is : {EMAIL_HOST_PASSWORD}")
##print(f"\n current_directory is : {os.getcwd()} \n")
##
### ==========================
### UTILITY FUNCTIONS
### ==========================
##def save_ip(ip, filename):
##    with open(filename, 'w') as f:
##        json.dump({"last_known_ip": ip}, f)
##
##def load_ip(filename):
##    if os.path.exists(filename):
##        with open(filename, 'r') as f:
##            data = json.load(f)
##            return data.get("last_known_ip")
##    return None
##
##def test_ip(ip, expected_device):
##    url = f"http://{ip}{TARGET_URL}"
##    try:
##        print(f"➡️  Checking {url}...", end=" ")
##        resp = requests.get(url, timeout=2.5)
##        print(f"Response: {resp.status_code}", end=" ")
##        if resp.ok:
##            data = resp.json()
##            if data.get("device") == expected_device:
##                print(f"✅ Found '{expected_device}'")
##                return True
##            else:
##                print(f"❌ Device mismatch: {data.get('device')}")
##        else:
##            print("❌ HTTP error")
##    except requests.exceptions.Timeout:
##        print("⏱️ Timeout")
##    except requests.exceptions.ConnectionError:
##        print("❌ Connection Error")
##    except Exception as e:
##        print(f"⚠️ {e}")
##    return False
##
##def discover_ip(keyword, filename, fallback_ip):
##    # 1) Try saved
##    last = load_ip(filename)
##    if last and test_ip(last, keyword):
##        return last
##
##    # 2) Scan network
##    print(f"🔍 Scanning for {keyword}...")
##    for i in range(50, 256):
##        candidate = f"192.168.{i}.146"
##        if test_ip(candidate, keyword):
##            save_ip(candidate, filename)
##            return candidate
##
##    # 3) Fallback
##    print(f"⚠️ {keyword} not found; using fallback {fallback_ip}")
##    return fallback_ip
##
### ==========================
### HOME AUTOMATION IP
### ==========================
##print("📁 Trying previously saved HOME_AUTOMATION_IP...")
##HOME_AUTOMATION_IP = discover_ip(
##    KEYWORD_HOME,
##    HOME_IP_FILE,
##    FALLBACK_HOME_IP
##)
##print("🎯 HOME_AUTOMATION_IP is", HOME_AUTOMATION_IP)
##
### ==========================
### LEFT EYE CAMERA IP
### ==========================
##print("\n📁 Trying previously saved LEFT_EYE_CAMERA_IP...")
##LEFT_EYE_CAMERA_INPUT = "http://" + discover_ip(
##    KEYWORD_LEFT_EYE,
##    LEFT_EYE_IP_FILE,
##    FALLBACK_LEFT_EYE_IP
##) + ":81/stream"
##print("🎯 LEFT_EYE_CAMERA_INPUT is", LEFT_EYE_CAMERA_INPUT)
##
### ==========================
### RIGHT EYE CAMERA IP
### ==========================
##print("\n📁 Trying previously saved RIGHT_EYE_CAMERA_IP...")
##RIGHT_EYE_CAMERA_INPUT = "http://" + discover_ip(
##    KEYWORD_RIGHT_EYE,
##    RIGHT_EYE_IP_FILE,
##    FALLBACK_RIGHT_EYE_IP
##) + ":81/stream"
##print("🎯 RIGHT_EYE_CAMERA_INPUT is", RIGHT_EYE_CAMERA_INPUT)
##
### ==========================
### SERIAL‑BASED IPs (unchanged)
### ==========================
##ip_module = ArduinoCommunicationIPModule(port=COM_PORT,
##                                         baudrate=BAUDRATE_ARDUINO)
##ips = ip_module.get_ip_addresses()
##
### If your serial module still populates these, you can override:
##LEFT_EYE_SERIAL_IP  = ips.get("left") if ips else None
##RIGHT_EYE_SERIAL_IP = ips.get("right") if ips else None
##CHEST_CAMERA_INPUT = RIGHT_EYE_SERIAL_IP
##
##print("\nSERIAL‑DETECTED IPs:")
##print(" LEFT (serial):", LEFT_EYE_SERIAL_IP)
##print(" RIGHT(serial):", RIGHT_EYE_SERIAL_IP)
##print(" CHEST_CAMERA_INPUT:", CHEST_CAMERA_INPUT)
##
##
### ==========================
### PATHS FOR MODELS
### ==========================
##
##YOLO_MODEL_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
##FACE_RECOGNITION_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
##VOSK_MODEL_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"
##










##import requests
##from get_ip_addresses import ArduinoCommunicationIPModule
##
##KEYWORD = "I_AM_HOME_AUTOMATION_001"
##TARGET_URL = "/json"  # This is the correct path on the ESP
##TARGET_URL_HTML = "/html"  # This is the correct path on the ESP
##
### Configuration parameters
##COM_PORT = "COM6"
##
### Configuration parameters
##SERIAL_PORT_BLUETOOTH = "COM10"
##BAUDRATE_BLUETOOTH = 38400
##SERIAL_PORT_ARDUINO = COM_PORT
##BAUDRATE_ARDUINO = 9600
##SERIAL_PORT_ARDUINO_WEBCAMS = COM_PORT
##BAUDRATE_ARDUINO_WEBCAM_IP = 9600
##DRIVE_LETTER = "C://"
##
##HOME_AUTOMATION_IP = None
##
##
##
##import requests
##
##print("🔍 Scanning network for HOME_AUTOMATION_IP...")
##
##HOME_AUTOMATION_IP = None
##KEYWORD = "I_AM_HOME_AUTOMATION_001"
##
##
####def scan_network_ip():
####    print("🔍 Scanning network for HOME_AUTOMATION_IP...")
####
####    found = False
####    for i in range(0, 255):  # Scans 192.168.0.0 to 192.168.254.255
####        ip = f"192.168.{i}.146"
####        url = f"http://{ip}/json"
####        print(f"➡️  Checking {url}...", end=" ")
####
####        try:
####            response = requests.get(url, timeout=2)
####            if response.status_code == 200:
####                print(f"✅ Found!")
####                print(f"📡 IP Address: {ip}")
####                print("🔁 Response:", response.json())
####                found = True
####                break
####            else:
####                print(f"❌ Status {response.status_code}")
####        except requests.exceptions.RequestException as e:
####            print(f"⏱️ Timeout or Error: {e.__class__.__name__}")
####            # Optional: sleep(0.1)  # Delay between requests
####
####    if not found:
####        print("❌ Could not find device on any expected IP.")
####
####scan_network_ip()
##
##print("🔍 Scanning network for HOME_AUTOMATION_IP...")
##
####for i in range(0, 254):
##for i in range(100, 256):
##    ip = f"192.168.{i}.146"
##    url = f"http://{ip}/json"
##    try:
##        print(f"➡️  Checking {url}...", end=" ")
##        response = requests.get(url, timeout=2.5)
##        print(f" Response : {response}")
##
##        if response.ok:
##            data = response.json()
##            device = data.get("device")
##            if device == KEYWORD:
##                HOME_AUTOMATION_IP = ip
##                print(f"✅ FOUND device '{device}' at {ip}")
##                break
##            else:
##                print(f"❌ JSON returned, but wrong device: {device}")
##        else:
##            print(f"❌ HTTP error: {response.status_code}")
##    except requests.exceptions.Timeout:
##        print("⏱️ Timeout")
##    except requests.exceptions.ConnectionError:
##        print("❌ Connection Error")
##    except Exception as e:
##        print(f"⚠️ Error: {e}")
##
##if HOME_AUTOMATION_IP is None:
##    HOME_AUTOMATION_IP = "192.168.138.146"  # Fallback IP
##    print("⚠️ Failed to auto-detect. Using fallback IP:", HOME_AUTOMATION_IP)
##else:
##    print("🎯 Success! Device found at IP:", HOME_AUTOMATION_IP)
##
##
####print("Scanning network for HOME_AUTOMATION_IP...")
####
####for i in range(0, 256):
####    ip = f"192.168.{i}.146"
####    try:
####        url = f"http://{ip}/json"
####        response = requests.get(url, timeout=0.5)
####        if response.ok:
####            data = response.json()
####            if data.get("device") == "I_AM_HOME_AUTOMATION_001":
####                HOME_AUTOMATION_IP = ip
####                print("✅ Found device at:", HOME_AUTOMATION_IP)
####                break
####    except Exception:
####        continue
####
####if HOME_AUTOMATION_IP is None:
####    HOME_AUTOMATION_IP = "192.168.138.146"  # fallback
####    print("⚠️ Failed to auto-detect. Using fallback IP:", HOME_AUTOMATION_IP)
##
##
####print("Scanning network for HOME_AUTOMATION_IP...")
####
####for i in range(0, 256):
####    ip = f"http://192.168.{i}.146{TARGET_URL_HTML}"
####    print(f"ip : {ip}")
####    try:
####        response = requests.get(f"http://{ip}{TARGET_URL_HTML}", timeout=0.5)
####        print(f"Scanning IP {http://{ip}{TARGET_URL_HTML}}")
####        print(f"Response Text {response.text}")
####        if response.ok and KEYWORD in response.text:
####            HOME_AUTOMATION_IP = ip
####            print("✅ Success: Found", HOME_AUTOMATION_IP)
####            break
####    except requests.RequestException:
####        continue
####
####if HOME_AUTOMATION_IP is None:
####    HOME_AUTOMATION_IP = "192.168.138.146"  # fallback
####    print("⚠️ Failed to auto-detect. Using fallback IP:", HOME_AUTOMATION_IP)
##
### Get IPs from ArduinoCommunicationIPModule
##ip_module = ArduinoCommunicationIPModule(port=COM_PORT, baudrate=BAUDRATE_ARDUINO)
##ips = ip_module.get_ip_addresses()
##
##LEFT_EYE_CAMERA_INPUT_NEW = ips.get("left") if ips else None
##RIGHT_EYE_CAMERA_INPUT_NEW = ips.get("right") if ips else None
##
##print('\n')
##print("HOME_AUTOMATION_IP:", HOME_AUTOMATION_IP)
##print("LEFT_EYE_CAMERA_INPUT_NEW:", LEFT_EYE_CAMERA_INPUT_NEW)
##print("RIGHT_EYE_CAMERA_INPUT_NEW:", RIGHT_EYE_CAMERA_INPUT_NEW)
##print('\n')
##
### Assign to camera inputs
##LEFT_EYE_CAMERA_INPUT = LEFT_EYE_CAMERA_INPUT_NEW
##CHEST_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##RIGHT_EYE_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##
### Paths for models
##YOLO_MODEL_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
##FACE_RECOGNITION_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
##VOSK_MODEL_PATH = DRIVE_LETTER + "Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"




##from get_ip_addresses import ArduinoCommunicationIPModule
##import requests
##from ipaddress import ip_address, ip_network
##
##COM_PORT = "COM6"
##
### Configuration parameters
##SERIAL_PORT_BLUETOOTH = "COM10"
##BAUDRATE_BLUETOOTH = 38400
##SERIAL_PORT_ARDUINO = COM_PORT
##BAUDRATE_ARDUINO = 9600
##SERIAL_PORT_ARDUINO_WEBCAMS = COM_PORT
##BAUDRATE_ARDUINO_WEBCAM_IP = 9600
##DRIVE_LETTER = "C://"
##
### Scan for HOME_AUTOMATION_IP
##HOME_AUTOMATION_IP = None
##KEYWORD = "I_AM_HOME_AUTOMATION_001"
##
##print("Scanning network for HOME_AUTOMATION_IP...")
##
##for i in range(0, 256):
##    try:
##        ip = f"192.168.{i}.146"
##        # ✅ Adjusted this line to check the /json endpoint
####        response = requests.get(f"http://{ip}/json", timeout=0.5)
##        response = requests.get(f"http://{ip}/html", timeout=0.5)
##        if KEYWORD in response.text:
##            HOME_AUTOMATION_IP = ip
##            print("✅ Success: Found", HOME_AUTOMATION_IP)
##            break
##    except Exception:
##        continue
##
##if HOME_AUTOMATION_IP is None:
##    HOME_AUTOMATION_IP = "192.168.138.146"  # fallback
##    print("⚠️ Failed to auto-detect. Using fallback IP:", HOME_AUTOMATION_IP)
##
##ip_module = ArduinoCommunicationIPModule(port=COM_PORT, baudrate=9600)
##ips = ip_module.get_ip_addresses()
##
##if ips:
##    LEFT_EYE_CAMERA_INPUT_NEW = ips.get("left")
##    RIGHT_EYE_CAMERA_INPUT_NEW = ips.get("right")
##else:
##    LEFT_EYE_CAMERA_INPUT_NEW = None
##    RIGHT_EYE_CAMERA_INPUT_NEW = None
##
##print('\n')
##print("HOME_AUTOATION_IP:", HOME_AUTOMATION_IP)
##print("LEFT_EYE_CAMERA_INPUT_NEW:", LEFT_EYE_CAMERA_INPUT_NEW)
##print("RIGHT_EYE_CAMERA_INPUT_NEW:", RIGHT_EYE_CAMERA_INPUT_NEW)
##print('\n')
##
##LEFT_EYE_CAMERA_INPUT = LEFT_EYE_CAMERA_INPUT_NEW
##CHEST_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##RIGHT_EYE_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##
### Other configuration parameters...
##YOLO_MODEL_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
##FACE_RECOGNITION_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
##VOSK_MODEL_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"




##from get_ip_addresses import ArduinoCommunicationIPModule
##import requests
##from ipaddress import ip_address, ip_network
##
##COM_PORT = "COM6"
##
### Configuration parameters
##SERIAL_PORT_BLUETOOTH = "COM10"
##BAUDRATE_BLUETOOTH = 38400
##SERIAL_PORT_ARDUINO = COM_PORT
##BAUDRATE_ARDUINO = 9600
##SERIAL_PORT_ARDUINO_WEBCAMS = COM_PORT
##BAUDRATE_ARDUINO_WEBCAM_IP = 9600
##DRIVE_LETTER = "C://"
##
### Scan for HOME_AUTOMATION_IP
##HOME_AUTOMATION_IP = None
##KEYWORD = "I_AM_HOME_AUTOMATION_001"
##
##print("Scanning network for HOME_AUTOMATION_IP...")
##
##for i in range(0, 256):
##    try:
##        ip = f"192.168.{i}.146"
##        response = requests.get(f"http://{ip}", timeout=0.5)
##        if KEYWORD in response.text:
##            HOME_AUTOMATION_IP = ip
##            print("Success: Found", HOME_AUTOMATION_IP)
##            break
##    except Exception:
##        continue
##
##if HOME_AUTOMATION_IP is None:
##    HOME_AUTOMATION_IP = "192.168.155.146"  # fallback
##    print("Failed to auto-detect. Using fallback IP:", HOME_AUTOMATION_IP)
##
##ip_module = ArduinoCommunicationIPModule(port=COM_PORT, baudrate=9600)
##ips = ip_module.get_ip_addresses()
##
##if ips:
##    LEFT_EYE_CAMERA_INPUT_NEW = ips.get("left")
##    RIGHT_EYE_CAMERA_INPUT_NEW = ips.get("right")
##else:
##    LEFT_EYE_CAMERA_INPUT_NEW = None
##    RIGHT_EYE_CAMERA_INPUT_NEW = None
##
##print('\n')
##print("HOME_AUTOATION_IP:", HOME_AUTOMATION_IP)
##print("LEFT_EYE_CAMERA_INPUT_NEW:", LEFT_EYE_CAMERA_INPUT_NEW)
##print("RIGHT_EYE_CAMERA_INPUT_NEW:", RIGHT_EYE_CAMERA_INPUT_NEW)
##print('\n')
##
##LEFT_EYE_CAMERA_INPUT = LEFT_EYE_CAMERA_INPUT_NEW
##CHEST_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##RIGHT_EYE_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##
### Other configuration parameters...
##YOLO_MODEL_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
##FACE_RECOGNITION_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
##VOSK_MODEL_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"
##




##import socket
##from get_ip_addresses import ArduinoCommunicationIPModule
##
##COM_PORT = "COM6"
##
### Configuration parameters
##SERIAL_PORT_BLUETOOTH = "COM10"
##BAUDRATE_BLUETOOTH = 38400
##SERIAL_PORT_ARDUINO = COM_PORT
##BAUDRATE_ARDUINO = 9600
##SERIAL_PORT_ARDUINO_WEBCAMS = COM_PORT
##BAUDRATE_ARDUINO_WEBCAM_IP = 9600
##DRIVE_LETTER = "C://"
##
####def scan_for_home_automation_identifier():
####
####    print("Scanning for IP....")
####    target_port = 80  # Or use a known port your home automation device responds on
####    keyword = "I_AM_HOME_AUTOMATION_001"
####    timeout = 0.3
####
####    for i in range(0, 256):
####        ip = f"192.168.{i}.146"
####        try:
####            with socket.create_connection((ip, target_port), timeout=timeout) as conn:
####                conn.sendall(b"GET / HTTP/1.1\r\nHost: "+ip.encode()+b"\r\n\r\n")
####                response = conn.recv(1024)
####                if keyword.encode() in response:
####                    print('\n')
####                    print(f'ip found with pass: {ip}')
####                    print('\n')
####                    return ip
####        except Exception:
####            continue
####    return None
##
##def scan_for_home_automation_identifier():
##    print("Start Scanning for IP....")
##    target_port = 80  # Or use a known port your home automation device responds on
##    keyword = "I_AM_HOME_AUTOMATION_001"
##    timeout = 0.3
##
##    for i in range(0, 256):
##        ip = f"192.168.{i}.146"
##        try:
##            with socket.create_connection((ip, target_port), timeout=timeout) as conn:
##                conn.sendall(b"GET / HTTP/1.1\r\nHost: "+ip.encode()+b"\r\n\r\n")
##                response = conn.recv(1024)
##                if keyword.encode() in response:
##                    print("Success: Found HOME_AUTOMATION_IP at", ip)
##                    return ip
##        except Exception:
##            continue
##    return None
##
### Automatically scan for HOME_AUTOMATION_IP
##HOME_AUTOMATION_IP = scan_for_home_automation_identifier()
##if not HOME_AUTOMATION_IP:
##    HOME_AUTOMATION_IP = "192.168.155.146"  # fallback value
##
##ip_module = ArduinoCommunicationIPModule(port=COM_PORT, baudrate=9600)
##ips = ip_module.get_ip_addresses()
##
##if ips:
##    LEFT_EYE_CAMERA_INPUT_NEW = ips.get("left")
##    RIGHT_EYE_CAMERA_INPUT_NEW = ips.get("right")
##else:
##    LEFT_EYE_CAMERA_INPUT_NEW = None
##    RIGHT_EYE_CAMERA_INPUT_NEW = None
##
##print('\n')
##print("HOME_AUTOATION_IP:", HOME_AUTOMATION_IP)
##print("LEFT_EYE_CAMERA_INPUT_NEW:", LEFT_EYE_CAMERA_INPUT_NEW)
##print("RIGHT_EYE_CAMERA_INPUT_NEW:", RIGHT_EYE_CAMERA_INPUT_NEW)
##print('\n')
##
##LEFT_EYE_CAMERA_INPUT = LEFT_EYE_CAMERA_INPUT_NEW
##CHEST_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##RIGHT_EYE_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##
### Other configuration parameters...
##YOLO_MODEL_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
##FACE_RECOGNITION_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
##VOSK_MODEL_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"



##from get_ip_addresses import ArduinoCommunicationIPModule
##
##COM_PORT = "COM6"
##
### Configuration parameters
##SERIAL_PORT_BLUETOOTH = "COM10"
##BAUDRATE_BLUETOOTH = 38400
##SERIAL_PORT_ARDUINO = COM_PORT
##BAUDRATE_ARDUINO = 9600
##SERIAL_PORT_ARDUINO_WEBCAMS = COM_PORT
##BAUDRATE_ARDUINO_WEBCAM_IP = 9600
##DRIVE_LETTER = "C://"
##
####HOME_AUTOMATION_IP = "192.168.89.80"
##HOME_AUTOMATION_IP = "192.168.155.146"
####HOME_AUTOMATION_IP = "192.168.155.80"
##
##ip_module = ArduinoCommunicationIPModule(port=COM_PORT, baudrate=9600)
##ips = ip_module.get_ip_addresses()
##
##if ips:
##    LEFT_EYE_CAMERA_INPUT_NEW = ips.get("left")
##    RIGHT_EYE_CAMERA_INPUT_NEW = ips.get("right")
##else:
##    LEFT_EYE_CAMERA_INPUT_NEW = None
##    RIGHT_EYE_CAMERA_INPUT_NEW = None
##
##print('\n')
##print("HOME_AUTOATION_IP:", HOME_AUTOMATION_IP)
##print("LEFT_EYE_CAMERA_INPUT_NEW:", LEFT_EYE_CAMERA_INPUT_NEW)
##print("RIGHT_EYE_CAMERA_INPUT_NEW:", RIGHT_EYE_CAMERA_INPUT_NEW)
##print('\n')
##
##LEFT_EYE_CAMERA_INPUT = LEFT_EYE_CAMERA_INPUT_NEW
##CHEST_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##RIGHT_EYE_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##
### Other configuration parameters...
##YOLO_MODEL_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
##FACE_RECOGNITION_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
##VOSK_MODEL_PATH = DRIVE_LETTER+"Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"


##from get_ip_addresses import ArduinoCommunicationIPModule
##
### Configuration parameters
##SERIAL_PORT_BLUETOOTH = "COM10"
##BAUDRATE_BLUETOOTH = 38400
##SERIAL_PORT_ARDUINO = "COM6"
##BAUDRATE_ARDUINO = 115200
##SERIAL_PORT_ARDUINO_WEBCAMS = "COM6"
##BAUDRATE_ARDUINO_WEBCAM_IP = 19200
##
##
##ip_module = ArduinoCommunicationIPModule(port="COM6", baudrate=19200)
##ips = ip_module.get_ip_addresses()
##
##if ips:
##    LEFT_EYE_CAMERA_INPUT_NEW = ips.get("left")
##    RIGHT_EYE_CAMERA_INPUT_NEW = ips.get("right")
##else:
##    LEFT_EYE_CAMERA_INPUT_NEW = None
##    RIGHT_EYE_CAMERA_INPUT_NEW = None
##
##print("LEFT_EYE_CAMERA_INPUT_NEW:", LEFT_EYE_CAMERA_INPUT_NEW)
##print("RIGHT_EYE_CAMERA_INPUT_NEW:", RIGHT_EYE_CAMERA_INPUT_NEW)
##
##
##LEFT_EYE_CAMERA_INPUT = LEFT_EYE_CAMERA_INPUT_NEW
##
####CAMERA_INPUT_CHANNEL = 1
##CHEST_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
####CAMERA_INPUT_CHANNEL = 'http://192.168.180.84:81/stream'
##
##RIGHT_EYE_CAMERA_INPUT = RIGHT_EYE_CAMERA_INPUT_NEW
##
##
### Other configuration parameters...
##YOLO_MODEL_PATH = "C://Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
##FACE_RECOGNITION_PATH = "C://Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
##VOSK_MODEL_PATH = "C://Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"
##
##


####from arduino_com import arduino_com
##from get_ip_addresses import ArduinoCommunicationIPModule
##
### config.py
##SERIAL_PORT_BLUETOOTH = "COM10"
##BAUDRATE_BLUETOOTH = 38400
##SERIAL_PORT_ARDUINO = "COM7"
##BAUDRATE_ARDUINO = 115200
##SERIAL_PORT_ARDUINO_WEBCAMS = "COM7"
##BAUDRATE_ARDUINO_WEBCAM_IP = 19200
##
##
##LEFT_EYE_CAMERA_INPUT_NEW = ArduinoCommunicationIPModule.receive_esp32cam_ip_left()
##print(f"LEFT_EYE_CAMERA_INPUT_NEW : {LEFT_EYE_CAMERA_INPUT_NEW}")
##
##RIGHT_EYE_CAMERA_INPUT_NEW = ArduinoCommunicationIPModule.receive_esp32cam_ip_right()
##print(f"RIGHT_EYE_CAMERA_INPUT_NEW : {RIGHT_EYE_CAMERA_INPUT_NEW}")
##
##
##
##
######LEFT_EYE_CAMERA_INPUT_NEW = arduino_com.receive_arduino()
######print(f"LEFT_EYE_CAMERA_INPUT_NEW : {LEFT_EYE_CAMERA_INPUT_NEW}")
######
######
######LEFT_EYE_CAMERA_INPUT = 'http://192.168.235.81:81/stream'
######
########CAMERA_INPUT_CHANNEL = 1
######CHEST_CAMERA_INPUT = 'http://192.168.235.81:81/stream'
########CAMERA_INPUT_CHANNEL = 'http://192.168.180.84:81/stream'
######
######RIGHT_EYE_CAMERA_INPUT = 'http://192.168.235.81:81/stream'
######
##
##YOLO_MODEL_PATH = "C://Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
##FACE_RECOGNITION_PATH = "C://Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images"
##
##VOSK_MODEL_PATH = "C://Python_Env//New_Virtual_Env//Project_Files//vosk//vosk_models//vosk-model-en-us-daanzu-20200905-lgraph//vosk-model-en-us-daanzu-20200905-lgraph"
##
