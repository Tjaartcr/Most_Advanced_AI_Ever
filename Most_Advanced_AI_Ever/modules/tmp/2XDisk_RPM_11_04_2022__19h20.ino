


const int ledPin = 13; // 2 Generating interrupts using CLK signal

//*****************************************************************
//                 ENCODER 1 (LEFT)

volatile boolean TurnDetected1;  // need volatile for Interrupts
volatile boolean rotationdirection1; // CW or CCW rotation

const int PinCLK1 = 18; // 2 Generating interrupts using CLK signal

int RotaryPosition1 = 0; // To store Stepper Motor Position
int PrevPosition1;   // Previous Rotary Position Value to check accuracy
int RPMavg1 = 0;
int RPM1 = 0;   // Revolutions Per Minute
int WheelCounter1 = 20; //Wheel has 20 gaps
int TimeStart1 = 0;
int TimeStart1a = 1000;
int Reset1 = 2000;
int TimeRunner1 = 0;
int RPMCounter1 = 0;
int RPMCounter1a = 0;

//Interrupt routine runs if CLK goes from HIGH to LOW

void isr1 () {

//  delay(2); // delay for Debouncing  (4)
  if (digitalRead(PinCLK1))

    rotationdirection1 = digitalRead(PinCLK1);
  TurnDetected1 = true;

}

//*****************************************************************
//                 ENCODER 2 (RIGHT)

volatile boolean TurnDetected2;  // need volatile for Interrupts
volatile boolean rotationdirection2; // CW or CCW rotation

const int PinCLK2 = 19; // 2 Generating interrupts using CLK signal

int RotaryPosition2 = 0; // To store Stepper Motor Position
int PrevPosition2;   // Previous Rotary Position Value to check accuracy
int RPMavg2 = 0;   // Revolutions Per Minute
int RPM2 = 0;   // Revolutions Per Minute
int WheelCounter2 = 20; //Wheel has 20 gaps
int TimeStart2 = 0;
int TimeStart2a = 1000;
int Reset2 = 2000;
int TimeRunner2 = 0;
int RPMCounter2 = 0;
int RPMCounter2a = 0;

//Interrupt routine runs if CLK goes from HIGH to LOW

void isr2 () {

//  delay(2); // delay for Debouncing  (4)
  if (digitalRead(PinCLK2))

    rotationdirection2 = digitalRead(PinCLK2);
  TurnDetected2 = true;

}

void setup() {

  Serial.begin(9600);

  pinMode (ledPin, OUTPUT);

  pinMode (PinCLK1, INPUT);
  attachInterrupt (5, isr1, RISING);  //interrupt 0 always connected to pin 2 on Arduino Uno

  pinMode (PinCLK2, INPUT);
  attachInterrupt (4, isr2, RISING);  //interrupt 0 always connected to pin 2 on Arduino Uno

  TimeRunner1 = millis();
  TimeRunner2 = millis();

}

void loop() {

  //************************************************************
  //                 ENCODER 1

  // Runs if rotation was detected

  if (TurnDetected1) {
    PrevPosition1 = RotaryPosition1;  // Save previous position in variable
    RotaryPosition1 = RotaryPosition1 + 1;  // Increase Position by 1
    RPMCounter1 = RPMCounter1 + 1;
    TurnDetected1 = false; // do NOT Repeat IF loop until new rotation detected

  }

  TimeStart1 = millis() - TimeRunner1;

  if (TimeStart1 <= TimeStart1a) {
    RPM1 = 0;
    RPMCounter1 = 0;   
    digitalWrite(ledPin, HIGH);
  }

  if (TimeStart1 >= TimeStart1a) {
    RPMCounter1a = RPMCounter1;
    digitalWrite(ledPin, LOW);

    }

  if (TimeStart1 >= Reset1) {
    RPM1 = (RPMCounter1a / 20) * 60;
    RPMavg1 = (RPM1 + RPM1) / 2;
    TimeRunner1 = millis();
  Serial.print("RPM1: ");
  Serial.println(RPM1);     

  Serial.print("RPMavg1: ");
  Serial.println(RPMavg1);   

  Serial.print("RPMCounter1a: ");
  Serial.println(RPMCounter1a);   
  
  }

//  //************************************************************
//  //                 ENCODER 2
//
//  // Runs if rotation was detected
//
//  if (TurnDetected2) {
//    PrevPosition2 = RotaryPosition2;  // Save previous position in variable
//    RotaryPosition2 = RotaryPosition2 + 2;  // Increase Position by 1
//    RPMCounter2 = RPMCounter2 + 1;
//    TurnDetected2 = false; // do NOT Repeat IF loop until new rotation detected
//
//  }
//
//  TimeStart2 = millis() - TimeRunner2;
//
//  if (TimeStart2 <= TimeStart2a) {
//    RPMCounter2 = RotaryPosition2;
//    digitalWrite(ledPin, HIGH);
//  }
//
//  if (TimeStart2 >= TimeStart2a) {
//    RPMCounter2a = RPMCounter2;
//    RPM2 = (RPMCounter2a / 20) * 60;
//    digitalWrite(ledPin, LOW);
//  Serial.print("RPM2: ");
//  Serial.println(RPM2);
//    }
//
//  if (TimeStart2 >= Reset2) {
//    RPM2 = 0;
//    TimeRunner2 = millis();
//  }
  
//  Serial.print("RPMCounter1a: ");
//  Serial.println(RPMCounter1a);

//  Serial.print("RPM1: ");
//  Serial.println(RPM1);

//  Serial.print("TimeStart1: ");
//  Serial.println(TimeStart1);  
}
