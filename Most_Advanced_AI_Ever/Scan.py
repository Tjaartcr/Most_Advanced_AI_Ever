import serial
import time

class ArduinoController:
    def __init__(self, port="COM6", baudrate=9600, timeout=1):
        try:
            self.arduino = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(5)  # Wait for Arduino reset
            print(f"✅ Connected to Arduino on {port}")
        except Exception as e:
            print(f"❌ Error connecting to Arduino: {e}")
            self.arduino = None

    def send_arduino(self, data):
        if self.arduino and self.arduino.is_open:
            self.arduino.write(data.encode())
            print(f"➡ Sent: {data}")
        else:
            print("⚠ Arduino not connected")

    def vision_look_left(self):
        data_left = f"L{640}Z"
        print(f"data_left : {data_left}")

        for i in range(10):
            self.send_arduino(data_left)
            print(f" i : {i+1}")
            time.sleep(0.1)

    def vision_look_right(self):
        data_right = f"G{10}Z"
        print(f"data_right : {data_right}")

        for i in range(10):
            self.send_arduino(data_right)
            print(f" i : {i+1}")
            time.sleep(0.1)

    def sweep(self, cycles=5):
        """Sweep left and right for given number of cycles"""
        for c in range(cycles):
            print(f"\n🔄 Sweep cycle {c+1}")
            self.vision_look_left()
            time.sleep(10)
            self.vision_look_right()
            time.sleep(10)


if __name__ == "__main__":
    arduino = ArduinoController(port="COM6", baudrate=9600)
    if arduino.arduino:
        arduino.sweep(cycles=5)  # Run 5 sweeps
