


import datetime
import random
import pyjokes
import socket
import serial
import Alfred_config




from speech import speech
##from communication import comm
from memory import memory
from Repeat_Last import repeat
##from assistant import assistant
from arduino_com import arduino_com
from listen import listen
from GUI import gui


##########################


##IP_ADRESS = "132.124.246.123"
##IP_ADRESS = "192.168.194.80"
##IP_ADRESS = "192.168.197.59"
##IP_ADRESS = "192.168.202.197"
IP_ADRESS = Alfred_config.HOME_AUTOMATION_IP
TIMEOUT = 5
port = 80
ip = IP_ADRESS

class HomeAutomation:

    def __init__(self):
        
        self.gui = gui   # ✅ Ensure GUI is correctly assigned
        
        if not gui:
            raise ValueError("GUI instance must be provided to AI_Assistant!")


    def Arduino_Home_Automation_Lights_On(self, AlfredQueryOffline):
        try:
            with socket.socket() as my_out_socket:
                my_out_socket.settimeout(TIMEOUT)
                my_out_socket.connect((ip, port))
                msg = "LED_ON\n"
                print("Message Sent:", msg)
                my_out_socket.send(msg.encode())


                current_date = datetime.datetime.now().strftime('%Y-%m-%d')
                current_time = datetime.datetime.now().strftime('%H:%M:%S')

                Alfred_Repeat_Previous_Response = "I am switching the lights on..."

                # GUI log if available
                msg_gui = (
                    f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
                    )

                query_resp = f"At {current_date} :  {current_time} : I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
                
                try:
                    assistant.gui.log_message(msg_gui)
                    assistant.gui.log_response(query_resp)
                except NameError:
                    print("GUI instance not available for logging message.")


            listen.send_bluetooth("Lights on")
            speech.AlfredSpeak("Lights on")

        except socket.error as e:
            print(f"Socket Error (Lights On): {e}")
            listen.send_bluetooth("Failed to turn lights on")
            speech.AlfredSpeak("Failed to turn lights on")
            return

##        handle_todo()
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")
     

    def Arduino_Home_Automation_Lights_Off(self, AlfredQueryOffline):
        try:
            with socket.socket() as my_out_socket:
                my_out_socket.settimeout(TIMEOUT)
                my_out_socket.connect((ip, port))
                msg = "LED_OFF\n"
                print("Message Sent:", msg)
                my_out_socket.send(msg.encode())


                current_date = datetime.datetime.now().strftime('%Y-%m-%d')
                current_time = datetime.datetime.now().strftime('%H:%M:%S')

                Alfred_Repeat_Previous_Response = "I am switching the lights off..."

                # GUI log if available
                msg_gui = (
                    f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
                )

                query_resp = f"At {current_date} :  {current_time} : I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
                
                try:
                    assistant.gui.log_message(msg_gui)
                    assistant.gui.log_response(query_resp)
                except NameError:
                    print("GUI instance not available for logging message.")


            listen.send_bluetooth("Lights off")
            speech.AlfredSpeak("Lights off")

        except socket.error as e:
            print(f"Socket Error (Lights Off): {e}")
            listen.send_bluetooth("Failed to turn lights off")
            speech.AlfredSpeak("Failed to turn lights off")
            return

##        handle_todo()
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")
             

        ##########################################################

    def Arduino_Home_Automation_Lights_Brightness(self, AlfredQueryOffline):
        import serial.tools.list_ports

        try:
            with serial.Serial("COM8", baudrate=9600, timeout=2) as serialInst:
                serialInst.flush()
                Serial_Bluetooth.flushInput()
                Bluetooth_RX = Serial_Bluetooth.readline()
                print("Bluetooth_RX:", Bluetooth_RX)

                command = Bluetooth_RX.decode('utf-8').strip()
                print("command:", command)

                listen.send_bluetooth("What would you want the brightness to be, dimmer or brighter sir?")
                speech.AlfredSpeak("What would you want the brightness to be, dimmer or brighter sir?")

                CheckLeftWord = 'be'
                CheckRightWord = 'please'

                indexLeftCalc = command.find(CheckLeftWord)
                indexRightCalc = command.find(CheckRightWord)

                NewBrighnessQuery = command[indexLeftCalc + len(CheckLeftWord) + 1:indexRightCalc].strip()
                print("NewBrighnessQuery:", NewBrighnessQuery)

                if "more dimmer" in NewBrighnessQuery:
                    serialInst.write(b'dimmer2')
                    reply = "Setting it more dimmer sir"

                elif "dimmer" in NewBrighnessQuery:
                    serialInst.write(b'dimmer')
                    reply = "Setting it dimmer sir"

                elif "more brighter" in NewBrighnessQuery:
                    serialInst.write(b'brighter2')
                    reply = "Setting it more brighter sir"

                elif "brighter" in NewBrighnessQuery:
                    serialInst.write(b'brighter')
                    reply = "Setting it brighter sir"

                elif any(word in command for word in ["thanks", "thank you", "done"]):
                    reply = "Brightness session ended"
                    listen.send_bluetooth(reply)
                    speech.AlfredSpeak(reply)
                    main.main()
                    return

                else:
                    reply = "Sorry, I didn’t understand the brightness request."

                listen.send_bluetooth(reply)
                speech.AlfredSpeak(reply)

        except Exception as e:
            print(f"Serial Error (Brightness): {e}")
            listen.send_bluetooth("Brightness adjustment failed")
            speech.AlfredSpeak("Brightness adjustment failed")

##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

            

    def Arduino_Home_Automation_Send_Door_Open(self, AlfredQueryOffline):
        my_out_socket = socket.socket()
        my_out_socket.settimeout(5)  # Avoid infinite hang

        port = 80
        ip = IP_ADRESS  # Make sure this is defined correctly

        try:
            my_out_socket.connect((ip, port))
            msg = "OPEN_DOOR\n"
            print("Message Sent:", msg)
            my_out_socket.send(msg.encode())


            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.datetime.now().strftime('%H:%M:%S')

            Alfred_Repeat_Previous_Response = "I am opening the door ..."

            # GUI log if available
            msg_gui = (
                f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
            )

            query_resp = f"At {current_date} :  {current_time} : I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
            
            try:
                assistant.gui.log_message(msg_gui)
                assistant.gui.log_response(query_resp)
            except NameError:
                print("GUI instance not available for logging message.")


            listen.send_bluetooth("Opening the door sir")
            speech.AlfredSpeak("Opening the door sir")
        except socket.error as e:
            print(f"Socket connection failed: {e}")
            listen.send_bluetooth("Failed to open the door")
            speech.AlfredSpeak("Failed to open the door")
            return
        finally:
            my_out_socket.close()

##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        try:
            if MyToDoListEdited:
                MyToDoListEdited.pop(0)
                AlfredQueryOfflineNew = MyToDoListEdited
                print("\nAlfredQueryOffline New:", AlfredQueryOfflineNew, "\n")
        except:
            pass

        ##########################################################
            
    def Arduino_Home_Automation_Send_Door_Close(self, AlfredQueryOffline):
        try:
            with socket.socket() as my_out_socket:
                my_out_socket.settimeout(TIMEOUT)
                my_out_socket.connect((ip, port))
                msg = "CLOSE_DOOR\n"
                print("Message Sent:", msg)
                my_out_socket.send(msg.encode())

            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.datetime.now().strftime('%H:%M:%S')

            Alfred_Repeat_Previous_Response = "I am closing the door ..."

            # GUI log if available
            msg_gui = (
                f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
            )

            query_resp = f"At {current_date} :  {current_time} : I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
            
            try:
                assistant.gui.log_message(msg_gui)
                assistant.gui.log_response(query_resp)
            except NameError:
                print("GUI instance not available for logging message.")


            listen.send_bluetooth("Closing the door, sir")
            speech.AlfredSpeak("Closing the door, sir")

        except socket.error as e:
            print(f"Socket Error (Close Door): {e}")
            listen.send_bluetooth("Failed to close the door")
            speech.AlfredSpeak("Failed to close the door")
            return

##        handle_todo()
##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

        ##########################################################
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    def Arduino_Home_Automation_Analog_Read(self, AlfredQueryOffline):
        try:
            with socket.socket() as my_in_socket:
                my_in_socket.settimeout(TIMEOUT)
                my_in_socket.connect((IP_ADRESS, 80))

                msg = my_in_socket.recv(1024).decode()
                print("Message Received:", msg)

            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.datetime.now().strftime('%H:%M:%S')

            Alfred_Repeat_Previous_Response = "I am reading the value ..."

            # GUI log if available
            msg_gui = (
                f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
            )

            query_resp = f"At {current_date} :  {current_time} : I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "
            
            try:
                assistant.gui.log_message(msg_gui)
                assistant.gui.log_response(query_resp)
            except NameError:
                print("GUI instance not available for logging message.")


        except socket.error as e:
            print(f"Socket Error (Analog Read): {e}")
            listen.send_bluetooth("Analog read failed")
            speech.AlfredSpeak("Analog read failed")


        ##########################################################
    def Arduino_Home_Automation_Read(self, AlfredQueryOffline):
        print("Let's wait for response...")

        try:
            with socket.socket() as my_in_socket:
                my_in_socket.settimeout(TIMEOUT)
                my_in_socket.connect((IP_ADRESS, 80))

                msgReceive = my_in_socket.recv(1024).decode().strip()
                print("Message Received:", msgReceive)

                if msgReceive == 'DOOR_IS_OPEN':
                    listen.send_bluetooth("Door is Open")
                    speech.AlfredSpeak("Door is Open")

                elif msgReceive == 'DOOR_IS_CLOSED':
                    listen.send_bluetooth("Door is Closed")
                    speech.AlfredSpeak("Door is Closed")

                elif msgReceive == 'Stop_Return':
                    listen.send_bluetooth("Door is stopped returning")
                    speech.AlfredSpeak("Door is stopped returning")

                elif msgReceive == 'C':
                    listen.send_bluetooth("Good Bye from Arduino protocol")
                    speech.AlfredSpeak("Good Bye from Arduino protocol")

                elif msgReceive == 'D':
                    listen.send_bluetooth("OK sir")
                    speech.AlfredSpeak("OK sir")
                    return self.Arduino_Home_Automation_Read(AlfredQueryOffline)

                else:
                    listen.send_bluetooth("Unknown message received")
                    speech.AlfredSpeak("Unknown message received")

        except socket.error as e:
            print(f"Socket Error (General Read): {e}")
            listen.send_bluetooth("Failed to get a response")
            speech.AlfredSpeak("Failed to get a response")

##        listen.send_bluetooth("Listening...")
##        speech.AlfredSpeak("Listening...")

    #////////////////////////////////////////////////////////////////////////////
       



    AlfredQueryOffline = ""


    print("assistant end")


home_auto = HomeAutomation()





