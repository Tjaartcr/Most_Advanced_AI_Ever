#include "esp_camera.h"
#include <WiFi.h>
#include <ArduinoOTA.h>
#include <WebServer.h>


//#include <SoftwareSerial.h>
//
//
//SoftwareSerial mySerial(17, 16); // RX, TX


//
// WARNING!!! PSRAM IC required for UXGA resolution and high JPEG quality
//            Ensure ESP32 Wrover Module or other board with PSRAM is selected
//            Partial images will be transmitted if image exceeds buffer size
//
//            You must select partition scheme from the board menu that has at least 3MB APP space.
//            Face Recognition is DISABLED for ESP32 and ESP32-S2, because it takes up from 15
//            seconds to process single frame. Face Detection is ENABLED if PSRAM is enabled as well

// ===================
// Select camera model
// ===================
//#define CAMERA_MODEL_WROVER_KIT // Has PSRAM
//#define CAMERA_MODEL_ESP_EYE  // Has PSRAM
//#define CAMERA_MODEL_ESP32S3_EYE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_PSRAM // Has PSRAM
//#define CAMERA_MODEL_M5STACK_V2_PSRAM // M5Camera version B Has PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_ESP32CAM // No PSRAM
//#define CAMERA_MODEL_M5STACK_UNITCAM // No PSRAM
//#define CAMERA_MODEL_M5STACK_CAMS3_UNIT  // Has PSRAM
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
//#define CAMERA_MODEL_TTGO_T_JOURNAL // No PSRAM
//#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM
// ** Espressif Internal Boards **
//#define CAMERA_MODEL_ESP32_CAM_BOARD
//#define CAMERA_MODEL_ESP32S2_CAM_BOARD
//#define CAMERA_MODEL_ESP32S3_CAM_LCD
//#define CAMERA_MODEL_DFRobot_FireBeetle2_ESP32S3 // Has PSRAM
//#define CAMERA_MODEL_DFRobot_Romeo_ESP32S3 // Has PSRAM
#include "camera_pins.h"

// ===========================
// Enter your WiFi credentials
// ===========================
const char *ssid = "*********";  // !!! INSERT YOUR OWN CREDENTIALS HERE
const char *password = "*********";  // !!! INSERT YOUR OWN CREDENTIALS HERE
const char* ota_password = "*******";  // !!! INSERT YOUR OWN CREDENTIALS HERE


#define LED_ONBOARD 4

String My_IP_Right = "";
String My_Previous_IP_Right = "";

int Send_Counter = 0;
volatile bool IP_OK = false;
volatile bool Send_Finished = false;

WebServer server(80);


//IPAddress localIP(192, 168, 128, 123); // ESP32 static IP
//IPAddress gateway(192, 168, 128, 1);    // IP Address of your network gateway (router)
//IPAddress subnet(255, 255, 255, 0);   // Subnet mask
//IPAddress primaryDNS(192, 168, 1, 1); // Primary DNS (optional)
//IPAddress secondaryDNS(0, 0, 0, 0);   // Secondary DNS (optional)

void startCameraServer();
void setupLedFlash(int pin);

void setup() {
//  mySerial.begin(115200);
  Serial.begin(9600);
//  Serial.setDebugOutput(true);
  Serial.println("");

  pinMode(LED_ONBOARD, OUTPUT);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
//  config.xclk_freq_hz = 20000000;
//  config.frame_size = FRAMESIZE_UXGA;
  config.xclk_freq_hz = 5000000;
  config.frame_size = FRAMESIZE_VGA;
  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    // Best option for face detection/recognition
//    config.frame_size = FRAMESIZE_240X240;
    config.frame_size = FRAMESIZE_VGA;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }
  // drop down frame size for higher initial frame rate
  if (config.pixel_format == PIXFORMAT_JPEG) {
//    s->set_framesize(s, FRAMESIZE_QVGA);
    s->set_framesize(s, FRAMESIZE_VGA);
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

// Setup LED FLash if LED pin is defined in camera_pins.h
#if defined(LED_GPIO_NUM)
  setupLedFlash(LED_GPIO_NUM);
#endif

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Right Not CONNECTED to WIFI !!!!");
  }
  Serial.println("");
  Serial.println("Right EYE WiFi is connected and running...");

  startCameraServer();

  Serial.println("Right EYE is Ready!");
  Serial.print("Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.print("' to connect to the 'Right EYE' WEB Server...");
  Serial.println("");
//  Serial.println("");

//
//  My_IP_Right = WiFi.localIP().toString();
//  mySerial.println("");
//  mySerial.print("My_IP_Right :  ");
//  mySerial.println(My_IP_Right);
//  String My_IP_Right_Send = My_IP_Right;    
//  mySerial.println(My_IP_Right_Send);    




  // OTA
  ArduinoOTA.setHostname("ESP32-CAM-RightEye");
  ArduinoOTA.setPassword(ota_password);

  ArduinoOTA.onStart([]() {
    Serial.println("OTA Start");
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\nOTA End");
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("OTA Progress: %u%%\r\n", (progress * 100) / total);
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("OTA Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
    else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
    else if (error == OTA_END_ERROR) Serial.println("End Failed");
  });
  ArduinoOTA.begin();
  Serial.println("OTA Ready");

//  // Web server endpoints
//  server.on("/", handleRoot);
//  server.on("/capture", HTTP_GET, handleCapture);
//  server.begin();
  digitalWrite(LED_ONBOARD, LOW);

  Serial.println("HTTP server started.");

}

void loop() {
  
  // Do nothing. Everything is done in another task by the web server


  ArduinoOTA.handle();

  // Handle web server client
  server.handleClient();


if ((Send_Counter >= 10)){
Send_Finished = true;
}

if (My_IP_Right != My_Previous_IP_Right){
  IP_OK = false;
  Send_Counter = false;
  Send_Finished = false;
}

if (My_IP_Right == My_Previous_IP_Right){
  IP_OK = true;
}


if ((Send_Finished == false) && (Send_Counter <= 10)){
//if (Send_Counter <= 10){  
  
  Serial.println("right is Ready!");
  Serial.println("");
  Serial.println("");
  Serial.print("right eye 'http://");
  Serial.print(WiFi.localIP());
  Serial.print("'right running...");
  Serial.println("");
  Send_Counter = Send_Counter + 1;

}
  
  delay(2000);
  
  My_IP_Right = WiFi.localIP().toString();
  My_Previous_IP_Right = My_IP_Right;

}
  
//  Serial.println("");
//  Serial.print("My_Previous_IP_Right : ");
//  Serial.print(My_Previous_IP_Right);
//  Serial.println("");
//
//  Serial.println("");
//  Serial.print("Send_Counter R : ");
//  Serial.print(Send_Counter);
//
//  Serial.println("");
//  Serial.print("IP_OK R : ");
//  Serial.print(IP_OK);  
   
//  Serial.println("Right is Ready!");
//  Serial.println("");
//  Serial.println("");
//  Serial.print("right eye 'http://");
//  Serial.print(WiFi.localIP());
//  Serial.print("'right running...");
//  Serial.println(""); 
