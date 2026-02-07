
import serial
import re
import time

class ArduinoCommunicationIPModule:
    def __init__(self, port="COM6", baudrate=9600, timeout=3):
        """
        Initializes the serial connection with the Arduino.
        """
        self.serial_conn = None
        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=timeout)
            print(f"✅ Connected to Arduino on port {port}")
        except serial.SerialException as e:
            print(f"❌ Failed to connect to Arduino: {e}")

    def _read_line(self):
        """
        Reads a line from the serial port and decodes it using latin-1.
        """
        if self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode("latin-1", errors="ignore").strip()
                return line
            except Exception as e:
                print("Error reading from serial:", e)
        return None

    def _extract_ips(self, data):
        """
        Extracts left and right IP addresses from a given string.
        """
        ips = {}

        # Try to extract using the specific "left eye" and "right eye" pattern.
        left_match = re.search(r"left\s+eye\s+'http://(\d{1,3}(?:\.\d{1,3}){3})'", data, re.IGNORECASE)
        if left_match:
            ips["left"] = left_match.group(1)

        right_match = re.search(r"right\s+eye\s+'http://(\d{1,3}(?:\.\d{1,3}){3})'", data, re.IGNORECASE)
        if right_match:
            ips["right"] = right_match.group(1)

        # Fallback to a generic extraction based on keywords.
        if not ips.get("left") and "left" in data.lower():
            generic_left = re.findall(r'(\d{1,3}(?:\.\d{1,3}){3})', data)
            if generic_left:
                ips["left"] = generic_left[0]
        if not ips.get("right") and "right" in data.lower():
            generic_right = re.findall(r'(\d{1,3}(?:\.\d{1,3}){3})', data)
            if generic_right:
                ips["right"] = generic_right[-1]  # take the last occurrence

        return ips

    def _format_ip(self, ip):
        """
        Formats a plain IP address into the desired URL pattern.
        """
        return f"http://{ip}:81/stream"
##        return f"http://{ip}:80/stream"

    def get_ip_addresses(self):
        """
        Continuously reads from the serial port until both left and right IP addresses are found.
        """
        ip_addresses = {"left": None, "right": None}

        print("⏳ Waiting for both IP addresses...")
        while True:
            line = self._read_line()
            if line:
                print(f"📡 Received: {line}")
                extracted_ips = self._extract_ips(line)
                
                if "left" in extracted_ips:
                    ip_addresses["left"] = self._format_ip(extracted_ips["left"])
                    print(f"✅ Left IP set to: {ip_addresses['left']}")
                if "right" in extracted_ips:
                    ip_addresses["right"] = self._format_ip(extracted_ips["right"])
                    print(f"✅ Right IP set to: {ip_addresses['right']}")
                
                # Return once both IPs are set
                if ip_addresses["left"] and ip_addresses["right"]:
                    print(f"🎯 Both IPs received: {ip_addresses}")
                    if self.serial_conn and self.serial_conn.is_open:
                        self.serial_conn.close()
                    return ip_addresses

            time.sleep(0.5)  # Delay to prevent overwhelming the serial port

if __name__ == "__main__":
    ip_module = ArduinoCommunicationIPModule()
    ips = ip_module.get_ip_addresses()
    print("🎯 Final IP Addresses:", ips)

