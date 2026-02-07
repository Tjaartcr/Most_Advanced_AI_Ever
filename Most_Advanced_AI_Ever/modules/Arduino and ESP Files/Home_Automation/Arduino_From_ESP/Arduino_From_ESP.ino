
#include <SoftwareSerial.h>
SoftwareSerial mySerial(2, 3); // RX, TX on Arduino

////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 20, 4);

int On_Off_Switch=8;
int OpenLimitSwitch=11;
int CloseLimitSwitch=10;    //15
int Enable_Mosfet_Pin=5;
int Open_Close_Mosfet_Pin=4;

int Enable_Mosfet_PinState=LOW;
int Open_Close_Mosfet_PinState=LOW;

int OpenLimitState=0;
int CloseLimitState=0;
int On_Off_Switch_State=0;

volatile bool OpenDoor = false;
volatile bool Door_Is_Open = false;

volatile bool CloseDoor = false;
volatile bool Door_Is_Closed = false;

volatile bool CloseDoorRunning = false;
volatile bool OpenDoorRunning = false;

volatile bool Door_is_Opened = false;
volatile bool Door_is_Closed = false;

//#define LED_pin 2
//#define LED_error_pin 15

//int sensorPin = A0; // the potentiometer is connected to analog pin 0
int ledPin = 9; // the LED is connected to digital pin 13
//int sensorValue; // an integer variable to store the potentiometer reading


// Variables will change:
int ledState = LOW;             // ledState used to set the LED

//***********************************************
//            SERIAL PRINT TIMER

unsigned long SerialPrintTime = 10;
unsigned long SerialPrintStartTime;
unsigned long SerialPrintProgress;
unsigned long SerialPrintResetTime = 1000;

int SerialPrintState;
//********************************************

//***********************************************
//           SERIAL PRINT DELAY TIMER

void serialprintTimer() {

  SerialPrintProgress = millis() - SerialPrintStartTime;     // Servo Head Progress

  if (SerialPrintProgress <= SerialPrintTime) {
    SerialPrintState = HIGH;
  }

  if (SerialPrintProgress >= SerialPrintTime) {
    SerialPrintState = LOW;
  }

  if (SerialPrintProgress >= SerialPrintResetTime) {
    SerialPrintStartTime = millis();
  }
}

//***********************************************

//////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////

void setup() {
 lcd.init();
 lcd.backlight();
 
 Serial.begin(9600);
 mySerial.begin(9600);

////////////////////////////////////////////////////////////////////////

Serial.println("Service started...");

   lcd.setCursor(0,0);
  lcd.print("Service running");
  
pinMode(ledPin, OUTPUT);
digitalWrite(ledPin, HIGH);


  SerialPrintStartTime = millis();

  pinMode(LED_BUILTIN, OUTPUT);

  pinMode(OpenLimitSwitch, INPUT_PULLUP);
  pinMode(CloseLimitSwitch, INPUT_PULLUP);
  pinMode(On_Off_Switch, INPUT_PULLUP);


  pinMode(Enable_Mosfet_Pin, OUTPUT);
  pinMode(Open_Close_Mosfet_Pin, OUTPUT);

  digitalWrite(Enable_Mosfet_Pin,LOW);
  digitalWrite(Open_Close_Mosfet_Pin,LOW);
  Enable_Mosfet_PinState=LOW;
  Open_Close_Mosfet_PinState=LOW;  

////////////////////////////////////////////////////////////////////////
volatile bool Door_is_Opened = false;
volatile bool Door_is_Closed = false;

delay (2000);
lcd.clear();

}

////////////////////////////////////////////////////////////////////////



void Open_Door(){
  serialprint();

 OpenLimitState=digitalRead(OpenLimitSwitch);
   
    if ((OpenDoor == true) && (OpenLimitState == HIGH)){    

      digitalWrite(Enable_Mosfet_Pin,HIGH);
      digitalWrite(Open_Close_Mosfet_Pin,LOW);
      Enable_Mosfet_PinState=HIGH;
      Open_Close_Mosfet_PinState=LOW;  
    Door_Is_Open = false;
         
    CloseDoorRunning = false;
    OpenDoorRunning = true;
   delay(10);
    }

    if (OpenLimitState == LOW){
     delay(10);
      Stop_Motors();
  }
  
}

void Close_Door(){
  serialprint();

 CloseLimitState=digitalRead(CloseLimitSwitch);

      if ((CloseDoor == true) &&  (CloseLimitState == HIGH)){
      digitalWrite(Enable_Mosfet_Pin,HIGH);
      digitalWrite(Open_Close_Mosfet_Pin,HIGH);
      Enable_Mosfet_PinState=HIGH;
      Open_Close_Mosfet_PinState=HIGH;      

    Door_Is_Closed = false;
        
    CloseDoorRunning = true;
    OpenDoorRunning = false;
   delay(10);


    }
    
    if (CloseLimitState == LOW){
      delay(10);
       Stop_Motors();

  }  
}

/////////////////////////////////////////////////////////////////////////////////////////////////

void Stop_Motors(){
  serialprint();

 OpenLimitState=digitalRead(OpenLimitSwitch);
 CloseLimitState=digitalRead(CloseLimitSwitch);

    if ((OpenDoor == true) && (OpenLimitState == LOW) && (CloseDoorRunning == true)){
Door_is_Opened = true;
    }


    if ((CloseDoor == true) && (CloseDoorRunning == true) && (CloseLimitState == LOW)){
Door_is_Closed = true;
    }
    
      Serial.println("");
      Serial.println("Stopping Motor !!!");
      Serial.println("");

      digitalWrite(Enable_Mosfet_Pin,LOW);
      digitalWrite(Open_Close_Mosfet_Pin,LOW);
      Enable_Mosfet_PinState=LOW;
      Open_Close_Mosfet_PinState=LOW;  

delay(3000);

      OpenDoor = false;
      CloseDoor = false;

        
      CloseDoorRunning = false;
      OpenDoorRunning = false;


}



///////////////////////////////////////////////////////////////////////////////////////

void serialprint() {

  serialprintTimer();

  if (SerialPrintState == HIGH) {
    
Serial.println("");
Serial.print("Open Limit Switch Status : ");
Serial.print(OpenLimitState);

Serial.println("");
Serial.print("Open Door Flag : ");
Serial.print(OpenDoor);

Serial.println("");
Serial.print("Door Open Status : ");
Serial.print(Door_Is_Open);


Serial.println("");
Serial.print("Close Limit Switch Status : ");
Serial.print(CloseLimitState);

Serial.println("");
Serial.print("Close Door Flag : ");
Serial.print(CloseDoor);

Serial.println("");
Serial.print("Door Close Status : ");
Serial.print(Door_Is_Closed);

//Serial.println("");
//Serial.print("Enable_Mosfet_PinState : ");
//Serial.print(Enable_Mosfet_PinState);
//
//Serial.println("");
//Serial.print("Open_Close_Mosfet_PinState : ");
//Serial.print(Open_Close_Mosfet_PinState);

Serial.println("");

}

digitalWrite(LED_BUILTIN, SerialPrintState);

}

///////////////////////////////////////////////////////////////////////


void loop() {

/////////////////////////////////////////////////////////////////////////
  
 OpenLimitState=digitalRead(OpenLimitSwitch);
 CloseLimitState=digitalRead(CloseLimitSwitch);

 On_Off_Switch_State=digitalRead(On_Off_Switch);

//  serialprintTimer();
  serialprint();

if ((OpenLimitState == HIGH) && (OpenDoor == true)){
OpenDoor = true;
Open_Door();
 }

if ((CloseLimitState == HIGH) && (CloseDoor == true)){
CloseDoor = true;
Close_Door();
 }

if (On_Off_Switch_State == LOW){

 mySerial.println("Stop_Door");
// Serial.println("");
//Serial.println("Door is stopped");
//Serial.println("");
}

    if ((OpenDoor == true) && (OpenLimitState == LOW)){
//Serial.println("");
//Serial.println("Open End Stop Reached !!!");
//Serial.println("");
Door_is_Opened = true;
//Door_is_Closed = true;
Stop_Motors();

  }
  
    if ((CloseDoor == true) &&  (CloseLimitState == LOW)){
//Serial.println("");
//Serial.println("Closed End Stop Reached !!!");
//Serial.println("");
Door_is_Closed = true;
Stop_Motors();      
  }  

///////////////////////////////////////////////////////////////////////////

 if (mySerial.available()) {
  String msg = mySerial.readString();
  Serial.println("");
  Serial.print("Message received: ");
  Serial.print(msg);
   lcd.clear();
   lcd.setCursor(0,0);
   lcd.print("Message received:");
  
   lcd.setCursor(0,1);
  lcd.print(msg);
  lcd.print("  ");

int value = LOW;

//**********************************************************************************

if (msg.indexOf("LED_ON") != -1) {
Serial.println("lights is on");  
mySerial.println("LED_IS_ON");
lcd.setCursor(0,1);
//  lcd.print("msg: ");
lcd.print("Lights is on");
digitalWrite(ledPin, HIGH);
value = HIGH;
} 

if (msg.indexOf("LED_OFF") != -1){
Serial.println("lights is off");  
mySerial.println("LED_IS_OFF");
lcd.setCursor(0,1);
//  lcd.print("msg: ");
lcd.print("Lights is off");
digitalWrite(ledPin, LOW);
value = LOW;
}

if (msg.indexOf("OPEN_DOOR") != -1) {
Serial.println("Opening the door");  
CloseDoor = false;
OpenDoor = true;
} 
    
if (msg.indexOf("CLOSE_DOOR") != -1){
Serial.println("Closing the door");  
OpenDoor = false;
CloseDoor = true;
}

//**********************************************************************************

  Serial.flush();
  mySerial.flush();
 
 }

if (Door_is_Opened == true){

mySerial.println("Door_is_Opened");

lcd.setCursor(0,0);
lcd.print("Door is Opened");

lcd.setCursor(0,1);
lcd.print("msg: ");

delay(100);

Door_is_Opened = false; 

delay(100);

}

if (Door_is_Closed == true){

lcd.setCursor(0,0);
lcd.print("Door is Closed");

lcd.setCursor(0,1);
lcd.print("msg: ");

delay(100);

Door_is_Closed = false;

delay(100);

}

}
