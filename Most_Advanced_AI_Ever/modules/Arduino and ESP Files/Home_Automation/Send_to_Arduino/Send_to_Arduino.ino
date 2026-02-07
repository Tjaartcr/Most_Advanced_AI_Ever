

#include <SoftwareSerial.h>

#include <ESP8266WiFi.h>
SoftwareSerial mySerial(4, 5); // RX, TX on Arduino

const char* ssid = "Beast";//type your ssid
const char* password = "TjaartCronje1234";//type your password

int ledPin = 2; // GPIO2 of ESP8266
WiFiServer server(80);//Service Port

// Set your Static IP address
//IPAddress local_IP(192, 168, 202, 197);
IPAddress local_IP(192, 168, 176, 59);
// Set your Gateway IP address
IPAddress gateway(192, 168, 176, 1);

IPAddress subnet(255, 255, 0, 0);
IPAddress primaryDNS(8, 8, 8, 8);   //optional
IPAddress secondaryDNS(8, 8, 4, 4); //optional


//***********************************************
//            SERIAL PRINT TIMER

unsigned long SerialPrintTime = 2000;
unsigned long SerialPrintStartTime;
unsigned long SerialPrintProgress;
unsigned long SerialPrintResetTime = 3000;

int SerialPrintState;
//********************************************


//////////////////////////////////////////////////////////////////////////////////////////////



void setup() {

  SerialPrintStartTime = millis();
  pinMode(ledPin, OUTPUT);

  
 Serial.begin(9600);
 mySerial.begin(9600);
delay(10);

     
// Connect to WiFi network
Serial.println();
Serial.println();
Serial.print("Connecting to ");
Serial.println(ssid);

//WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS);

WiFi.begin(ssid, password);

while (WiFi.status() != WL_CONNECTED) {
delay(5);
Serial.print(".");
}
Serial.println("");
Serial.println("WiFi connected");
Serial.println("");

// Start the server
server.begin();
Serial.println("Server started");


//// Print the IP address
//Serial.print("Use this URL to connect: ");
//Serial.print("http://");

Serial.println(WiFi.localIP());
//mySerial.println("Hallo I am ESP");
mySerial.println(WiFi.localIP());
mySerial.print("   ");


WiFiClient client = server.available();

//client.println("I_AM_HOME_AUTOMATION_001");

//delay(2000);

//Serial.println("/");

}

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

////////////////////////////////////////////////////////////////////////////////////////
void serialprint() {

  serialprintTimer();

  if (SerialPrintState == HIGH) {
    
Serial.println("");
Serial.println("Testing");
digitalWrite(ledPin, HIGH);

Serial.println("");

}
else
{
  digitalWrite(ledPin, LOW);
}

}

///////////////////////////////////////////////////////////////////////

void loop() {
  // put your main code here, to run repeatedly:
  serialprintTimer();
digitalWrite(ledPin, SerialPrintState);

/////////////////////////////////////////////////////////////////////////////////////
  
// Check if a client has connected
WiFiClient client = server.available();

String myReceived_Message = mySerial.readStringUntil('\r');

if (myReceived_Message.indexOf("Stop_Door") != -1) {

Serial.println(""); 
Serial.print("myReceived_Message : "); 
Serial.print(myReceived_Message); 
Serial.println(""); 

client.println("Stop_Return");  
Serial.println("Door is stopped"); 
} 
    
if (myReceived_Message.indexOf('C') != -1) {

Serial.println(""); 
Serial.print("myReceived_Message : "); 
Serial.print(myReceived_Message); 
Serial.println(""); 

client.println("Stop_Return");  
Serial.println("Door is stopped"); 
} 
    
if (myReceived_Message.indexOf("Door_is_Opened") != -1) {
client.println("DOOR_IS_OPEN");  
Serial.println("the door is open"); 
} 
    
if (myReceived_Message.indexOf("Door_is_Closed") != -1){

Serial.println(""); 
Serial.print("myReceived_Message : "); 
Serial.print(myReceived_Message); 
Serial.println(""); 

client.println("DOOR_IS_CLOSED");  
Serial.println("the door is closed");  
}

if (myReceived_Message.indexOf("LED_IS_ON") != -1) {
client.println("LED_IS_ON");  
Serial.println("the light os on"); 
} 
    
if (myReceived_Message.indexOf("LED_IS_OFF") != -1){

Serial.println(""); 
Serial.print("myReceived_Message : "); 
Serial.print(myReceived_Message); 
Serial.println(""); 

client.println("LED_IS_OFF");  
Serial.println("the light is off");  
}


if (!client) {
return;
}

// Wait until the client sends some data
//Serial.println("Message Received");

while(!client.available()){
delay(1);
}

// Read the first line of the request
String request = client.readStringUntil('\r');
Serial.println(request);
mySerial.println(request);
  
//client.println("I_AM_HOME_AUTOMATION_001");

///////////////////////////////////////////

  if (request.indexOf("/json") >= 0) {
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: application/json");
    client.println("Connection: close");
    client.println();
    client.println("{");
    client.println("\"device\": \"I_AM_HOME_AUTOMATION_001\",");
    client.print("\"ip\": \"");
    client.print(WiFi.localIP());
    client.println("\"");
    client.println("}");
  } else {
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/html");
    client.println("Connection: close");
    client.println();
    client.println("<!DOCTYPE html><html><head>");
    client.println("<meta name='viewport' content='width=device-width, initial-scale=1'>");
    client.println("<style>");
    client.println("body { font-family: Arial; background: #1e1e2f; color: white; text-align: center; padding: 30px; }");
    client.println("h1 { color: #4CAF50; }");
    client.println(".card { background: #2c2c3e; margin: auto; padding: 20px; max-width: 600px; border-radius: 15px; box-shadow: 0 0 15px rgba(0,0,0,0.4); }");
    client.println(".btn { background: #4CAF50; color: white; padding: 15px 25px; font-size: 16px; border: none; border-radius: 8px; cursor: pointer; margin: 10px; }");
    client.println(".btn:hover { background: #45a049; }");
    client.println("</style>");
    client.println("<title>My ESP8266 Home Automation 01 Dashboard</title></head><body>");
    client.println("<div class='card'>");
    client.println("<h1>I_AM_HOME_AUTOMATION_001</h1>");
    client.println("<p>Welcome to the ESP8266 Home Automation Node!</p>");
    client.print("<p>IP Address: ");
    client.print(WiFi.localIP());
    client.println("</p>");
//    client.println("<a href='/toggle_led'><button class='btn'>Toggle LED</button></a>");
    client.println("</div></body></html>");
  }


//if (request.indexOf("GET /json") >= 0) {
//  // Respond with JSON
//  client.println("HTTP/1.1 200 OK");
//  client.println("Content-Type: application/json");
//  client.println("Connection: close");
//  client.println();
//  client.println("{");
//  client.println("\"device\": \"I_AM_HOME_AUTOMATION_001\",");
//  client.print("\"ip\": \"");
//  client.print(WiFi.localIP());
//  client.println("\"");
//  client.println("}");
//} else {
//  // Respond with HTML
//  client.println("HTTP/1.1 200 OK");
//  client.println("Content-Type: text/html");
//  client.println("Connection: close");
//  client.println();
//  client.println("<!DOCTYPE HTML>");
//  client.println("<html>");
//  client.println("<head><title>Home Automation Node</title></head>");
//  client.println("<body style='font-family:Arial;text-align:center;'>");
//  client.println("<h1>I_AM_HOME_AUTOMATION_001</h1>");
//  client.println("<p>Welcome to the ESP8266 Home Automation Node!</p>");
//  client.println("<p>IP Address: " + WiFi.localIP().toString() + "</p>");
//  client.println("</body>");
//  client.println("</html>");
//}


///////////////////////////////////////////

delay(10);

Serial.flush();
mySerial.flush();
client.flush();


delay(1);  

digitalWrite(ledPin, LOW);

}
