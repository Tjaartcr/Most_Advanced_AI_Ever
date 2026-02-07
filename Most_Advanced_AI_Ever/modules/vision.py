




import cv2
##from Face_Recognition_Image_Trainer_Best_Great import SimpleFacerec
import Alfred_config
##from communication import comm
import time
import datetime
import queue

from ultralytics import YOLO

from speech import speech
from GUI import gui
from face_tracking import face_tracking
from listen import listen
from assistant import assistant
from arduino_com import arduino

print("vision")

############################################
###         CAMERA CHANNEL SELECT

Camera_Input_Channel = 1

############################################
#       FACE RECOGNITION SOFTWARE

What_I_See_Front = []
##What_Is_In_Front_Speak = []
What_Is_In_Front_Speak = ""

Who_Is_In_Front = []
Who_Is_In_Front_Speak = []

POI_Who_Is_In_Front = []
POI_String_New = 0

Names_and_POI_Together_List = []
Names_and_POI_Together = 0

Name_Only_For_Where = 0
Name_Only_For_Look_AT = 0

print("vision start")

############################################
#       OBJECT DETECTION SOFTWARE

from ultralytics import YOLO

Model_File = Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8m.pt"  #//yolov8n.pt"
print("Loading the Model YoloV8n.....")

print('\n')
print('____________________________________________________________________')
print('\n')

print("visio start 1")

# YOLO Model (legacy instance for some functions)
Obsticle_Detection_Vision_Model = YOLO(Model_File)
with open(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
    classNames = [line.rstrip('\n') for line in f]

from collections import Counter

print("vision start 2")


##LEFT_EYE_CAMERA_INPUT
##RIGHT_EYE__CAMERA_INPUT



class VisionModule:
    
    def __init__(self):
        ## Uncomment if you want to load face recognition features
        # self.face_recognition = SimpleFacerec()
        # self.face_recognition.load_encoding_images(Alfred_config.FACE_RECOGNITION_PATH)


        # Initialize the object detector using the YOLO model path from Alfred_config.
        self.object_detector = YOLO(Alfred_config.YOLO_MODEL_PATH)

        # Open the camera using the channel defined in Alfred_config.
        self.camera = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
        
        self.camera.set(3, 640)  # Width
        self.camera.set(4, 480)  # Height

        self.Coord_Top_Left_X = 0
        self.Coord_Top_Left_Y = 480

        self.Coord_Middel_Left_X = 0  #   200
        self.Coord_Middel_Left_Y = 240  #   110

        self.Coord_Bottom_Left_X = 0
        self.Coord_Bottom_Left_Y = 0

        self.Coord_Top_Middel_X = 320
        self.Coord_Top_Middel_Y = 480

        self.Coord_Middel_Middel_X = 320
        self.Coord_Middel_Middel_Y = 240

        self.Coord_Bottom_Middel_X = 320
        self.Coord_Bottom_Middel_Y = 0

        self.Coord_Top_Right_X = 640
        self.Coord_Top_Right_Y = 480

        self.Coord_Middel_Right_X = 640
        self.Coord_Middel_Right_Y = 240

        self.Coord_Bottom_Right_X = 640
        self.Coord_Bottom_Right_Y = 0
        
        self.gui = gui   # ✅ Ensure GUI is correctly assigned
        
        if not gui:
            raise ValueError("GUI instance must be provided to Vision module!")
        self.gui = gui

        self.response_queue = queue.Queue()

        # Load object class names (using your original file path)
        try:
            with open(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
                self.classNames = [line.rstrip('\n') for line in f]
            print("Loaded object class names in VISION.py.")
        except Exception as e:
            self.classNames = []
            print("Warning: Could not load object class names:", e)

    def detect_faces(self):
        """Detects faces and returns recognized names."""
        ret, frame = self.camera.read()
        if not ret:
            return []
        names = self.face_recognition.detect_known_faces(frame)
        return names

    def detect_objects(self):
        """Detects objects and returns list of recognized items."""
        ret, frame = self.camera.read()
        if not ret:
            return []
        results = self.object_detector(frame)
        detected_objects = [result['name'] for result in results]
        return detected_objects

    # ---------------------------
    # New method for GUI integration:
    def get_frame(self):
        """
        Capture a frame from the camera, run object detection,
        draw bounding boxes and labels, and return the processed frame.
        """
        ret, frame = self.camera.read()
        if not ret:
            return None

        # Run the YOLO object detection on the frame.
        results = self.object_detector(frame, stream=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates.
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw stylized bounding box lines.
                cv2.line(frame, (x1, y1), (x1 + 15, y1), (0, 255, 0), 2)
                cv2.line(frame, (x1, y1), (x1, y1 + 15), (0, 255, 0), 2)
                cv2.line(frame, (x2, y1), (x2 - 15, y1), (0, 255, 0), 2)
                cv2.line(frame, (x1, y2), (x1 + 15, y2), (0, 255, 0), 2)
                cv2.line(frame, (x1, y2), (x1, y2 - 15), (0, 255, 0), 2)
                cv2.line(frame, (x2, y2), (x2 - 15, y2), (0, 255, 0), 2)
                cv2.line(frame, (x2, y1), (x2, y1 + 15), (0, 255, 0), 2)
                cv2.line(frame, (x2, y2), (x2, y2 - 15), (0, 255, 0), 2)
                
                # Compute confidence and get class name.
                confidence = round(float(box.conf[0]) * 100, 2)
                class_index = int(box.cls[0])
                if self.classNames and class_index < len(self.classNames):
                    class_name = self.classNames[class_index]
                else:
                    class_name = "Unknown"
                text = f"{class_name}: {confidence}%"
                # Draw the label text.
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame

    def release(self):
        """Release the camera resource."""
        self.camera.release()


        
    def Vision_assistant_extract_text_from_query(self, resp):
        """
        Parse resp and return (timestamp, message, username).
        - timestamp: "YYYY-MM-DD HH:MM:SS" or None
        - message: cleaned string (no leading/trailing whitespace, no newlines)
        - username: parsed username or self.current_user fallback
        """
        import re

        username = getattr(self, "current_user", "ITF")
        timestamp = None
        message = ""

        if not resp:
            return None, "", username

        # dict case
        if isinstance(resp, dict):
            date = resp.get("date") or resp.get("timestamp_date") or None
            time = resp.get("time") or resp.get("timestamp_time") or None
            if date and time:
                timestamp = f"{date} {time}"
            elif resp.get("timestamp"):
                timestamp = str(resp.get("timestamp")).strip()
            message = str(resp.get("text") or resp.get("query") or resp.get("response") or "").strip()
            username = str(resp.get("username") or resp.get("user") or username)
            message = re.sub(r'\s*\n+\s*', ' ', message).strip()
            message = re.sub(r'\s{2,}', ' ', message)
            return timestamp, message, username

        # string case
        if isinstance(resp, str):
            s = resp.replace('\r', '').strip()
            s = re.sub(r'\n+\s*$', '', s)  # strip trailing blank lines

            # 1) extract username at end if present
            m_user = re.search(r":\s*'username':\s*(?P<u>[^\n:]+)\s*$", s)
            if m_user:
                username = m_user.group("u").strip()
                s = s[:m_user.start()].rstrip(" :\n\t")

            # Remove optional leading "At " then try to capture date/time with flexible separator
            s_no_at = re.sub(r'^\s*At\s+', '', s, flags=re.IGNORECASE)

            # pattern handles either "YYYY-MM-DD  HH:MM:SS" or "YYYY-MM-DD :  HH:MM:SS"
            m_ts = re.match(
                r"^(?P<date>\d{4}-\d{2}-\d{2})\s*(?:[:]\s*|\s+)(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<rest>.*)$",
                s_no_at,
                flags=re.DOTALL
            )
            if m_ts:
                timestamp = f"{m_ts.group('date')} {m_ts.group('time')}"
                rest = m_ts.group("rest")
            else:
                # no timestamp; treat entire string (after removing username) as rest
                rest = s

            # strip common labels and collapse newlines/spaces
            rest = re.sub(r"^\s*(I replied:|I said:)\s*", "", rest, flags=re.IGNORECASE)
            message = re.sub(r'\s*\n+\s*', ' ', rest).strip()
            message = re.sub(r'\s{2,}', ' ', message)

            return timestamp, message, username

        # fallback
        message = str(resp).strip()
        message = re.sub(r'\s*\n+\s*', ' ', message).strip()
        message = re.sub(r'\s{2,}', ' ', message)
        return None, message, username



    # ---------------------------
    # ---------------------------


    def Vision_Look_Left(self, AlfredQueryOffline):

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")

        data_left = f"L{640}Z"
        print(f"data_left : {data_left}")

        while True:

            for i in range(10):
                i = i + 1
                arduino.send_arduino(data_left)
                print(f" i : {i}")

            break

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        Alfred_Repeat_Previous_Response = "I am looking towards the Left..."

        # GUI log if available
        msg = (
            f"At {current_date} :  {current_time} : You Asked: {message} "
        )

        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "
        
        try:
            assistant.gui.log_message(msg)
            assistant.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

        time.sleep(3)
        
        speech.AlfredSpeak(Alfred_Repeat_Previous_Response)
        listen.send_bluetooth(Alfred_Repeat_Previous_Response)


    def Vision_Look_InFront(self, AlfredQueryOffline):
        
        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")

        data_front_UD = f"D{240}Z"
        data_front_LR = f"F{320}Z"
                
        print(f"data_front_UD : {data_front_UD}")
        print(f"data_front_LR : {data_front_LR}")

        while True:

            for i in range(10):
                i = i + 1
                arduino.send_arduino(data_front_UD)
                print(f" i : {i}")

            for i in range(10):
                i = i + 1
                arduino.send_arduino(data_front_LR)
                print(f" i : {i}")

            break

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        Alfred_Repeat_Previous_Response = "I am looking towards the Front ..."

        # GUI log if available
        msg = (
            f"At {current_date} :  {current_time} : You Asked: {message} "
        )

        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"
##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        
        try:
            assistant.gui.log_message(msg)
            assistant.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")

        time.sleep(3)
        
        speech.AlfredSpeak(Alfred_Repeat_Previous_Response)
        listen.send_bluetooth(Alfred_Repeat_Previous_Response)


    def Vision_Look_Forward(self, AlfredQueryOffline):
        

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")


##        data_front = f"D{320}Z"
##        data_front = f"C{self.Coord_Middel_Middel_Y}D{self.Coord_Middel_Middel_X}Z"
        data_front = f"F{320}Z"
        print(f"data_front : {data_front}")

        while True:

            for i in range(10):
                i = i + 1
                arduino.send_arduino(data_front)
    ##                    time.sleep(0.1)
                print(f" i : {i}")

            break


        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        Alfred_Repeat_Previous_Response = "I am looking towards the front ..."

        # GUI log if available
        msg = (
            f"At {current_date} :  {current_time} : You Asked: {message} "
        )

        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

####        query_resp = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        
        try:
            assistant.gui.log_message(msg)
            assistant.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        time.sleep(3)
        
        speech.AlfredSpeak(Alfred_Repeat_Previous_Response)
        listen.send_bluetooth(Alfred_Repeat_Previous_Response)


    def Vision_Look_Straight(self, AlfredQueryOffline):

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")


        data_front_UD = f"D{240}Z"
        print(f"data_front_UD : {data_front_UD}")

        while True:

            for i in range(10):
                i = i + 1
                arduino.send_arduino(data_front_UD)
    ##                    time.sleep(0.1)
                print(f" i : {i}")

            break

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        Alfred_Repeat_Previous_Response = "I am looking Straight ..."

        # GUI log if available
        msg = (
            f"At {current_date} :  {current_time} : You Asked: {message} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"
####        query_resp = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        
        try:
            assistant.gui.log_message(msg)
            assistant.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        time.sleep(2)
        
        speech.AlfredSpeak(Alfred_Repeat_Previous_Response)
        listen.send_bluetooth(Alfred_Repeat_Previous_Response)



    def Vision_Look_Right(self, AlfredQueryOffline):

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")

##        data_right = f"D{640}Z"
##        data_right = f"C{self.Coord_Middel_Right_Y}D{self.Coord_Middel_Right_X}Z"
        data_right = f"G{10}Z"
        print(f"data_right : {data_right}")

        while True:

            for i in range(10):
                i = i + 1
                arduino.send_arduino(data_right)
    ##                    time.sleep(0.1)
                print(f" i : {i}")

            break


        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        Alfred_Repeat_Previous_Response = "I am looking towards the Right ..."

        # GUI log if available
        msg = (
            f"At {current_date} :  {current_time} : You Asked: {message} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"

####        query_resp = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        
        try:
            assistant.gui.log_message(msg)
            assistant.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        time.sleep(3)
        
        speech.AlfredSpeak(Alfred_Repeat_Previous_Response)
        listen.send_bluetooth(Alfred_Repeat_Previous_Response)


    def Vision_Look_Up(self, AlfredQueryOffline):
        

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")


##        data_up = f"C{self.Coord_Top_Middel_Y}D{self.Coord_Top_Middel_X}Z"
##        data_up = f"G{self.Coord_Top_Middel_Y}H{self.Coord_Top_Middel_X}Z"
##        data_up = f"C{480}Z"
        data_up = f"A{480}Z"

##        data_up = f"C{480}D{320}Z"

        print(f"data_up : {data_up}")

        while True:

            for i in range(10):
                i = i + 1
                arduino.send_arduino(data_up)
    ##                    time.sleep(0.1)
                print(f" i : {i}")

            break

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        Alfred_Repeat_Previous_Response = "I am looking Up ..."

        # GUI log if available
        msg = (
            f"At {current_date} :  {current_time} : You Asked: {message} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"
####        query_resp = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        
        try:
            assistant.gui.log_message(msg)
            assistant.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        time.sleep(3)
        
        speech.AlfredSpeak(Alfred_Repeat_Previous_Response)
        listen.send_bluetooth(Alfred_Repeat_Previous_Response)



    def Vision_Look_Down(self, AlfredQueryOffline):

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")

        data_down = f"J{10}Z"

        print(f"data_down : {data_down}")

        while True:

            for i in range(10):
                i = i + 1
                arduino.send_arduino(data_down)
    ##                    time.sleep(0.1)
                print(f" i : {i}")

            break


        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        Alfred_Repeat_Previous_Response = "I am looking Down ..."

        # GUI log if available
        msg = (
            f"At {current_date} :  {current_time} : You Asked: {message} "
        )


        model = "Alfred"
        query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"
####        query_resp = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
##        query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

        
        try:
            assistant.gui.log_message(msg)
            assistant.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        time.sleep(3)
        
        speech.AlfredSpeak(Alfred_Repeat_Previous_Response)
        listen.send_bluetooth(Alfred_Repeat_Previous_Response)


    # ---------------------------
    # ---------------------------

    
    # The following methods are your original functions.
    def Vision_What_Left(self, AlfredQueryOffline):

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")

        print("Object detection system is running.....")

        ### ESP32-CAM IP ADRESS
        ##cap = cv2.VideoCapture('http://192.168.35.231:4747/video')

        try:
            cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT, cv2.CAP_DSHOW)
            if not cap.isOpened():
                    raise ConnectionError("ESP32 stream unreachable")
        except Exception as e:
            print(f"⚠️ Camera stream error: {e}")

        cap.set(3, 1023)
        cap.set(4, 768)

        Model_File = Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"

        print("Loading the Model YoloV8n.....")

        # YOLO Model
        Obsticle_Detection_Vision_Model = YOLO(Model_File)

        with open(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
            classNames = [line.rstrip('\n') for line in f]

        print(classNames)

        print("Object detection Model is Loaded.....")

        while True:
            success, img = cap.read()
            results = Obsticle_Detection_Vision_Model(img, stream=False)

            print("Object Detection Software is running.....")

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # Draw bounding box lines
                    cv2.line(img, (x1, y1), (x1 + 15, y1), (0, 255, 0), 2)
                    cv2.line(img, (x1, y1), (x1, y1 + 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y1), (x2 - 15, y1), (0, 255, 0), 2)
                    cv2.line(img, (x1, y2), (x1 + 15, y2), (0, 255, 0), 2)
                    cv2.line(img, (x1, y2), (x1, y2 - 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y2), (x2 - 15, y2), (0, 255, 0), 2)
                    cv2.line(img, (x2, y1), (x2, y1 + 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y2), (x2, y2 - 15), (0, 255, 0), 2)
                    # Draw confidence and class name
                    confidence = round(float(box.conf[0]) * 100, 2)
                    class_index = int(box.cls[0])
                    class_name = classNames[class_index]
                    text = f"{class_name}: {confidence}%"
                    org = (x1, y1 - 10)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.75
                    color = (0, 0, 255)
                    thickness = 1
                    cv2.putText(img, text, org, font, font_scale, color, thickness)

            cv2.imshow("Object Detection", img)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                cap.release()
                cv2.destroyAllWindows()

                break


        cap.release()
        cv2.destroyAllWindows()

    def Vision_Where_InFront(self, AlfredQueryOffline):

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")

        What_I_See_Front.clear()

        print("Where are you is running.....")

        ### ESP32-CAM IP ADRESS
        ##cap = cv2.VideoCapture('http://192.168.35.231:4747/video')

        try:
            Cap_Object_Front = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
            if not Cap_Object_Front.isOpened():
                    raise ConnectionError("ESP32 stream unreachable")
        except Exception as e:
            print(f"⚠️ Camera stream error: {e}")

        Cap_Object_Front.set(3, 640)
        Cap_Object_Front.set(4, 480)

        with open(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
            classNames = [line.rstrip('\n') for line in f]

        print("Object detection Model is Loaded.....")

        while True:
            success, img = Cap_Object_Front.read()
            results = Obsticle_Detection_Vision_Model(img, stream=False)

            print("Object Detection Software is running.....")

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # Draw bounding box
                    cv2.line(img, (x1, y1), (x1 + 15, y1), (0, 255, 0), 2)
                    cv2.line(img, (x1, y1), (x1, y1 + 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y1), (x2 - 15, y1), (0, 255, 0), 2)
                    cv2.line(img, (x1, y2), (x1 + 15, y2), (0, 255, 0), 2)
                    cv2.line(img, (x1, y2), (x1, y2 - 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y2), (x2 - 15, y2), (0, 255, 0), 2)
                    cv2.line(img, (x2, y1), (x2, y1 + 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y2), (x2, y2 - 15), (0, 255, 0), 2)
                    # Draw confidence and class name
                    confidence = round(float(box.conf[0]) * 100, 2)
                    class_index = int(box.cls[0])
                    class_name = classNames[class_index]
                    text = f"{class_name}: {confidence}%"
                    org = (x1, y1 - 10)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    color = (0, 0, 255)
                    thickness = 1
                    cv2.putText(img, text, org, font, font_scale, color, thickness)
                    What_I_See_Front.append(class_name)
                    print(What_I_See_Front)

            cv2.imshow("Face Detection System", img)

            My_New_What_Is_InFront_String = ''.join(What_I_See_Front)
            print("My_New_What_Is_InFront_String : " + My_New_What_Is_InFront_String)

            My_New_What_Is_InFront_String_Edited = My_New_What_Is_InFront_String.replace(",", "  ")
            print("My_New_What_Is_InFront_String_Edited : " + My_New_What_Is_InFront_String_Edited)

            My_New_What_Is_InFront_String_Edited_Final = My_New_What_Is_InFront_String_Edited.replace("  ", " , and a ")
            print("My_New_What_Is_InFront_String_Edited_Final : " + My_New_What_Is_InFront_String_Edited_Final)

            my_response_where = My_New_What_Is_InFront_String_Edited_Final

        ########################################################
                
            if "TV Monitor" in my_response_where and ("Sofa" in  my_response_where or "couch" in  my_response_where):
                listen.send_bluetooth("I am in the living room, Sir")
                speech.AlfredSpeak("I am in the living room, Sir")
                time.sleep(2)
                What_I_See_Front.clear()
                Cap_Object_Front.release()
                cv2.destroyAllWindows()

            if "Bed" in my_response_where:
                listen.send_bluetooth("I am in the bedroom, Sir")
                speech.AlfredSpeak("I am in the bedroom, Sir")
                time.sleep(2)
                What_I_See_Front.clear()
                Cap_Object_Front.release()
                cv2.destroyAllWindows()
                 
            if "bath" in my_response_where:
                listen.send_bluetooth("I am in the bathroom, Sir")
                speech.AlfredSpeak("I am in the bathroom, Sir")
                time.sleep(2)
                What_I_See_Front.clear()
                Cap_Object_Front.release()
                cv2.destroyAllWindows()
                 
            if "Toilet" in my_response_where:
                listen.send_bluetooth("I am in the toilet, Sir")
                speech.AlfredSpeak("I am in the toilet, Sir")
                time.sleep(2)
                What_I_See_Front.clear()
                Cap_Object_Front.release()
                cv2.destroyAllWindows()
                 
            if "Dining Table" in my_response_where or "Oven" in my_response_where: 
                listen.send_bluetooth("I am in the kitchen, Sir")
                speech.AlfredSpeak("I am in the kitchen, Sir")
                time.sleep(2)
                What_I_See_Front.clear()
                Cap_Object_Front.release()
                cv2.destroyAllWindows()

        print("Let's go back to Alfred....")
        return                  


    def Vision_What_InFront(self, AlfredQueryOffline):

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")

        global Camera_Input_Channel
        
        What_I_See_Front.clear()

        print("Object detection system is running.....")


        # Use the provided ESP32 URL - update if needed (e.g., add /stream if required)
        stream_url = "http://192.168.69.64:81/stream"

        # Open the video stream from the ESP32 camera
##        Cap_Object_Front = cv2.VideoCapture(stream_url)

        try:
            Cap_Object_Front = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
            if not Cap_Object_Front.isOpened():
                    raise ConnectionError("ESP32 stream unreachable")
        except Exception as e:
            print(f"⚠️ Camera stream error: {e}")


##        Cap_Object_Front = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT, cv2.CAP_DSHOW)
##        Cap_Object_Front = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        Cap_Object_Front.set(3, 640)
        Cap_Object_Front.set(4, 480)

        with open(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
            classNames = [line.rstrip('\n') for line in f]

        print("Object detection Model is Loaded.....")

        while True:
            success, img = Cap_Object_Front.read()
            results = Obsticle_Detection_Vision_Model(img, stream=False)

            print("Object Detection Software is running.....")

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.line(img, (x1, y1), (x1 + 15, y1), (0, 255, 0), 2)
                    cv2.line(img, (x1, y1), (x1, y1 + 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y1), (x2 - 15, y1), (0, 255, 0), 2)
                    cv2.line(img, (x1, y2), (x1 + 15, y2), (0, 255, 0), 2)
                    cv2.line(img, (x1, y2), (x1, y2 - 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y2), (x2 - 15, y2), (0, 255, 0), 2)
                    cv2.line(img, (x2, y1), (x2, y1 + 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y2), (x2, y2 - 15), (0, 255, 0), 2)
                    confidence = round(float(box.conf[0]) * 100, 2)
                    class_index = int(box.cls[0])
                    class_name = classNames[class_index]
                    text = f"{class_name}: {confidence}%"
                    org = (x1, y1 - 10)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    color = (0, 0, 255)
                    thickness = 1
                    cv2.putText(img, text, org, font, font_scale, color, thickness)
                    What_I_See_Front.append(class_name)
                    print(What_I_See_Front)

                cv2.imshow("Object Detection System", img)

                My_New_What_Is_InFront_String = ''.join(What_I_See_Front)
                print("My_New_What_Is_InFront_String : " + My_New_What_Is_InFront_String)

                My_New_What_Is_InFront_String_Edited = My_New_What_Is_InFront_String.replace(",", "  ")
                print("My_New_What_Is_InFront_String_Edited : " + My_New_What_Is_InFront_String_Edited)

                My_New_What_Is_InFront_String_Edited_Final = My_New_What_Is_InFront_String_Edited.replace("  ", " , and a ")
                print("My_New_What_Is_InFront_String_Edited_Final : " + My_New_What_Is_InFront_String_Edited_Final)

                speech.AlfredSpeak("Sir, I see a ")
                speech.AlfredSpeak(My_New_What_Is_InFront_String_Edited_Final)

        ##        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                current_date = datetime.datetime.now().strftime('%Y-%m-%d')
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                
                Alfred_Repeat_Previous_Response = My_New_What_Is_InFront_String_Edited_Final
                                                                    

                    # GUI log if available
                msg = (
                    f"At {current_date} :  {current_time} : You Asked: {message} "
                    f"and I replied:\n\n{Alfred_Repeat_Previous_Response}\n\n"
                )


                model = "Alfred"
                query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"
##        ##        query_resp = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
##                query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

                
                try:
                    assistant.gui.log_message(msg)
                    assistant.gui.log_response(query_resp)
                except NameError:
                    print("GUI instance not available for logging message.")

                             
                What_I_See_Front.clear()
                Cap_Object_Front.release()

            if "Person" in My_New_What_Is_InFront_String_Edited_Final:
                print("and the person or person's are?")
                What_I_See_Front.clear()
                face_tracking.Vision_Who_InFront()


            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                                    
                Cap_Object_Front.release()
                cv2.destroyAllWindows()
##                break



            print("Let's go back to Alfred....")                


    def Vision_What_Right(self, AlfredQueryOffline):

        print(f"GUI my_received_response  : {AlfredQueryOffline}")

        # call the method on this instance (important)
        timestamp, message, username = self.Vision_assistant_extract_text_from_query(AlfredQueryOffline)
        
        print(f"GUI RESPONSE  timestamp : {timestamp!r}")
        print(f"GUI RESPONSE  message : {message!r}")
        print(f"GUI RESPONSE  username: {username!r}")

        print("Object detection system is running.....")

        ### ESP32-CAM IP ADRESS
        ##cap = cv2.VideoCapture('http://192.168.35.231:4747/video')

        try:
            cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT, cv2.CAP_DSHOW)
            if not cap.isOpened():
                    raise ConnectionError("ESP32 stream unreachable")
        except Exception as e:
            print(f"⚠️ Camera stream error: {e}")

        cap.set(3, 1023)
        cap.set(4, 768)

        Model_File = Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"

        print("Loading the Model YoloV8n.....")

        # YOLO Model
        Obsticle_Detection_Vision_Model = YOLO(Model_File)

        with open(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
            classNames = [line.rstrip('\n') for line in f]

        print(classNames)

        print("Object detection Model is Loaded.....")

        while True:
            success, img = cap.read()
            results = Obsticle_Detection_Vision_Model(img, stream=False)

            print("Object Detection Software is running.....")

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.line(img, (x1, y1), (x1 + 15, y1), (0, 255, 0), 2)
                    cv2.line(img, (x1, y1), (x1, y1 + 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y1), (x2 - 15, y1), (0, 255, 0), 2)
                    cv2.line(img, (x1, y2), (x1 + 15, y2), (0, 255, 0), 2)
                    cv2.line(img, (x1, y2), (x1, y2 - 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y2), (x2 - 15, y2), (0, 255, 0), 2)
                    cv2.line(img, (x2, y1), (x2, y1 + 15), (0, 255, 0), 2)
                    cv2.line(img, (x2, y2), (x2, y2 - 15), (0, 255, 0), 2)
                    confidence = round(float(box.conf[0]) * 100, 2)
                    class_index = int(box.cls[0])
                    class_name = classNames[class_index]
                    text = f"{class_name}: {confidence}%"
                    org = (x1, y1 - 10)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.75
                    color = (0, 0, 255)
                    thickness = 1
                    cv2.putText(img, text, org, font, font_scale, color, thickness)
            cv2.imshow("Object Detection", img)
            
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                                    
                cap.release()
                cv2.destroyAllWindows()
                
##                break

    ##        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            
            Alfred_Repeat_Previous_Response = My_New_What_Is_InFront_String_Edited_Final

                # GUI log if available
            msg = (
                f"At {current_date} :  {current_time} : You Asked: {message} "
                f"and I replied:\n\n{Alfred_Repeat_Previous_Response}\n\n"
            )


            model = "Alfred"
            query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {username}"
##    ##        query_resp = f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} and I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
##            query_resp = f"At {current_date} :  {current_time} : {Alfred_Repeat_Previous_Response} : 'username':{username} "

            
            try:
                assistant.gui.log_message(msg)
                assistant.gui.log_response(query_resp)
            except NameError:
                print("GUI instance not available for logging message.")

        cap.release()
        cv2.destroyAllWindows()

print("vision end")

vision = VisionModule()






