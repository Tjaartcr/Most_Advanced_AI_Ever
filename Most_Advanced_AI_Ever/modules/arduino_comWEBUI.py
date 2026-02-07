
import serial
import WEBUI_config



print("arduino_com")

class WEBUIArduinoCommunicationModule:
    def __init__(self):
        self.arduinoWEBUI = None
        try:
            self.arduinoWEBUI = serial.Serial(WEBUI_config.SERIAL_PORT_ARDUINO, WEBUI_config.BAUDRATE_ARDUINO, timeout=1)
            print("✅ Connected to Arduino.")
        except serial.SerialException as e:
            print("❌ Failed to connect to Arduino:", e)

    def send_arduino(self, message):
        """Send message to Arduino."""
##        print(f"arduino : {self.arduino}")
##        print(f"Before sending message to Arduino: {message}")
        if self.arduinoWEBUI:
            print(f"Sending message to Arduino: {message}")
            self.arduinoWEBUI.write(message.encode('utf-8'))

    def receive_arduino(self):
        """Receive message from Arduino."""
        if self.arduino:
            data = self.arduinoWEBUI.readline().decode()
            print(f"Received from Arduino: {data}")
            return data
        return None

arduinoWEBUI = WEBUIArduinoCommunicationModule()



### communication.py
##import serial
##import Alfred_config
##
##print("arduino_com")
##
##class ArduinoCommunicationModule:
##    
##    def __init__(self):
##        
##        self.arduino = None
##
##        try:
##            self.arduino = serial.Serial(Alfred_config.SERIAL_PORT_ARDUINO, Alfred_config.BAUDRATE_ARDUINO, timeout=1)
##            print("✅ Connected to Arduino.")
##        except serial.SerialException:
##            print("Failed to connect to Arduino.")
##
##    def send_arduino(self, message):
##        """Send message to Arduino."""
##        print(f"arduino : {self.arduino}")
##        print(f"Before Sending this message to Arduino {message}")
##        if self.arduino:
##            print(f"Sending this message to Arduino {message}")
##            self.arduino.write(message.encode('utf-8'))
##
##    def receive_arduino(self):
##        """Receive message from Arduino."""
##        if self.arduino:
##            return self.arduino.readline().decode()
##            print(f"receive_arduino : {receive_arduino}")
##            
##        return None
##
##
##arduino = ArduinoCommunicationModule()

##arduino_com = ArduinoLEFT_EYE_CAMERA_INPUT_NEW = arduino_com.receive_arduino()
##print(f"LEFT_EYE_CAMERA_INPUT_NEW : {LEFT_EYE_CAMERA_INPUT_NEW}")
##
##
##LEFT_EYE_CAMERA_INPUT = 'http://192.168.235.81:81/stream'
##
####CAMERA_INPUT_CHANNEL = 1
##CHEST_CAMERA_INPUT = 'http://192.168.235.81:81/stream'
####CAMERA_INPUT_CHANNEL = 'http://192.168.180.84:81/stream'
##
##RIGHT_EYE_CAMERA_INPUT = 'http://192.168.235.81:81/stream'

arduinoWEBUI = WEBUIArduinoCommunicationModule
