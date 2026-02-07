# arduino_com.py
import time
import serial
import Alfred_config

class ArduinoCommunicationModule:
    def __init__(self, port=None, baud=None, auto_open=True):
        self.arduino = None
        self.port = port or Alfred_config.SERIAL_PORT_ARDUINO
        self.baud = baud or Alfred_config.BAUDRATE_ARDUINO
        if auto_open:
            self.open()

    def open(self):
        if self.arduino and getattr(self.arduino, "is_open", False):
            return
        try:
            # Non-blocking reads (timeout=0)
            self.arduino = serial.Serial(self.port, self.baud, timeout=0)
            # Allow Arduino to reset and boot
            time.sleep(2.0)
            # Clear buffers
            try:
                self.arduino.reset_input_buffer()
                self.arduino.reset_output_buffer()
            except Exception:
                pass
            print(f"✅ Connected to Arduino on {self.port} @ {self.baud}")
        except serial.SerialException as e:
            self.arduino = None
            print(f"❌ Failed to connect to Arduino: {e}")

    def close(self):
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
        except Exception:
            pass
        self.arduino = None

    def send_arduino(self, message: str, add_newline: bool = False):
        """Send message to Arduino. Optionally add newline (safe)."""
        if not self.arduino or not self.arduino.is_open:
            return
        out = message + ("\n" if add_newline else "")
        try:
            self.arduino.write(out.encode('utf-8'))
            self.arduino.flush()
        except Exception:
            # avoid raising on write errors in production; log if needed
            pass

    def receive_arduino(self) -> str:
        """Non-blocking receive. Returns whatever bytes are available (decoded)."""
        if not self.arduino or not getattr(self.arduino, "is_open", False):
            return ""
        try:
            n = self.arduino.in_waiting
            if n:
                raw = self.arduino.read(n)
                return raw.decode('utf-8', errors='ignore')
        except Exception:
            return ""
        return ""

# Create a module-level instance so 'from arduino_com import arduino' works.
# If connection fails, arduino will be an instance whose `.arduino` attribute is None.
arduino = ArduinoCommunicationModule(auto_open=True)

# Example standalone test (only runs when executed directly)
if __name__ == "__main__":
    # Use the module-level instance
    if not arduino.arduino:
        print("No Arduino connected; exiting test.")
    else:
        try:
            while True:
                arduino.send_arduino('T', add_newline=False)
                t0 = time.time()
                elapsed = 0
                response = ""
                while elapsed < 0.1:  # wait up to 100ms for an ack
                    response = arduino.receive_arduino()
                    if response:
                        break
                    time.sleep(0.001)
                    elapsed = time.time() - t0
                print(f"roundtrip {elapsed*1000:.1f} ms, response: {repr(response)}")
                time.sleep(0.25)
        except KeyboardInterrupt:
            arduino.close()
            print("Test stopped.")










########    # LASTEST WORKING 2025_11_04__22h00
########
########import serial
########import Alfred_config
########
########
########
########print("arduino_com")
########
########class ArduinoCommunicationModule:
########    def __init__(self):
########        self.arduino = None
########        try:
########            self.arduino = serial.Serial(Alfred_config.SERIAL_PORT_ARDUINO, Alfred_config.BAUDRATE_ARDUINO, timeout=1)
########            print("✅ Connected to Arduino.")
########        except serial.SerialException as e:
########            print("❌ Failed to connect to Arduino:", e)
########
########    def send_arduino(self, message):
########        """Send message to Arduino."""
##########        print(f"arduino : {self.arduino}")
##########        print(f"Before sending message to Arduino: {message}")
########        if self.arduino:
##########            print(f"Sending message to Arduino: {message}")
########            self.arduino.write(message.encode('utf-8'))
########
########    def receive_arduino(self):
########        """Receive message from Arduino."""
########        if self.arduino:
########            data = self.arduino.readline().decode()
##########            print(f"Received from Arduino: {data}")
########            return data
########        return None
########
########arduino = ArduinoCommunicationModule()
########
########

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

arduino_com = ArduinoCommunicationModule
