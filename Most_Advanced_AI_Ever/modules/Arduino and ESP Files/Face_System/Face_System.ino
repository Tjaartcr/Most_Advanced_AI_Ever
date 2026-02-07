#include <SoftwareSerial.h> 
#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

SoftwareSerial serialOne(2, 3); // Software Serial ONE
SoftwareSerial serialTwo(8, 9); // Software Serial TWO
SoftwareSerial serialCheck(4, 5); // Software Serial CHECK

// ----------------- small configuration tweaks added -----------------
bool ARD_INVERT_LR = false;   // If left/right feeling swapped, toggle this
bool ARD_INVERT_UD = false;   // If up/down feeling swapped, toggle this

int SMALL_STEP = 15;         // degrees for "little" moves (adjustable via M<int> from Python)
int EXTREME_LEFT_ANGLE = 30;   // min
int EXTREME_RIGHT_ANGLE = 150; // max
int EXTREME_UP_ANGLE = 30;     // min
int EXTREME_DOWN_ANGLE = 150;  // max

// --------------------------------------------------------------------

unsigned long lastMoveTimeUp = 0;
const int moveIntervalUp = 50;  // ms between updates

unsigned long lastMoveTimeDown = 0;
const int moveIntervalDown = 50;  // ms between updates

unsigned long lastMoveTimeLeft = 0;
const int moveIntervalLeft = 1;  // ms between updates

unsigned long lastMoveTimeRight = 0;
const int moveIntervalRight = 1;  // ms between updates

unsigned long lastMoveTimeForward = 0;
const int moveIntervalForward = 50;  // ms between updates

unsigned long lastMoveTimeStraight = 0;
const int moveIntervalStraight = 50;  // ms between updates

// movement state
int currentPosUp;       // not heavily used, kept for compatibility
bool MoveUp = false;
bool MoveDown = false;
bool MoveLeft = false;
bool MoveRight = false;
bool MoveForward = false;
bool MoveStraight = false;

// ==== CONFIG (per-direction) ====
int slowdownZoneUp = 20;   // degrees before target where servo slows down
int fastStepUp     = 4;    // step size when far from target
int slowStepUp     = 1;    // step size when in slowdown zone

int slowdownZoneDown = 20;
int fastStepDown     = 4;
int slowStepDown     = 1;

int slowdownZoneLeft = 20;
int fastStepLeft     = 4;
int slowStepLeft     = 1;

int slowdownZoneRight = 20;
int fastStepRight     = 4;
int slowStepRight     = 1;

int slowdownZoneForward = 20;
int fastStepForward     = 4;
int slowStepForward     = 1;

int slowdownZoneStraight = 20;
int fastStepStraight     = 4;
int slowStepStraight     = 1;

// Serial / input strings
String inByteRight = "";
String inByteLeft = "";

String Left_Center_Read_Serial = "";
String Serial_Check_Read;

// many variables preserved from original sketch
int Serial_Check_Read_Serial;

int Up_Up_Down_Serial_Read;
int Up_Up_Down_Serial_Read_Use;
int Up_Up_Down_Serial_Read_Mapped;
int Head_Up_Up_Down_Previous_Possision;

int Down_Up_Down_Serial_Read;
int Down_Up_Down_Serial_Read_Use;
int Down_Up_Down_Serial_Read_Mapped;
int Head_Down_Up_Down_Previous_Possision;

int Forward_Left_Right_Serial_Read;
int Forward_Left_Right_Serial_Read_Use;
int Forward_Left_Right_Serial_Read_Mapped;
int Head_Forward_Left_Right_Previous_Possision;

int Straight_Up_Down_Serial_Read;
int Straight_Up_Down_Serial_Read_Use;
int Straight_Up_Down_Serial_Read_Mapped;
int Head_Straight_Up_Down_Previous_Possision;

int InFront_Up_Down_Serial_Read;
int InFront_Up_Down_Serial_Read_Use;
int InFront_Up_Down_Serial_Read_Mapped;
int Head_InFront_Up_Down_Previous_Possision;

int InFront_Left_Right_Serial_Read;
int InFront_Left_Right_Serial_Read_Use;
int InFront_Left_Right_Serial_Read_Mapped;
int Head_InFront_Left_Right_Previous_Possision;

int Left_Left_Right_Serial_Read;
int Left_Left_Right_Serial_Read_Use;
int Left_Left_Right_Serial_Read_Mapped;
int Head_Left_Left_Right_Previous_Possision;

int Right_Left_Right_Serial_Read;
int Right_Left_Right_Serial_Read_Use;
int Right_Left_Right_Serial_Read_Mapped;
int Head_Right_Left_Right_Previous_Possision;

int Left_Top_Up_Down_Read_Serial;
int Left_Top_Left_Right_Read_Serial;
int Left_Middel_Up_Down_Read_Serial;
int Left_Middel_Left_Right_Read_Serial;
int Left_Bottom_Up_Down_Read_Serial;
int Left_Bottom_Left_Right_Read_Serial;

int Right_Top_Up_Down_Read_Serial;
int Right_Top_Left_Right_Read_Serial;
int Right_Middel_Up_Down_Read_Serial;
int Right_Middel_Left_Right_Read_Serial;
int Right_Bottom_Up_Down_Read_Serial;
int Right_Bottom_Left_Right_Read_Serial;

int Middel_Top_Up_Down_Read_Serial;
int Middel_Top_Left_Right_Read_Serial;
int Middel_Middel_Up_Down_Read_Serial;
int Middel_Middel_Left_Right_Read_Serial;
int Middel_Bottom_Up_Down_Read_Serial;
int Middel_Bottom_Left_Right_Read_Serial;

int Left_Top_Up_Down_Read;
int Left_Top_Left_Right_Read;
int Left_Middel_Up_Down_Read;
int Left_Middel_Left_Right_Read;
int Left_Bottom_Up_Down_Read;
int Left_Bottom_Left_Right_Read;

int Right_Top_Up_Down_Read;
int Right_Top_Left_Right_Read;
int Right_Middel_Up_Down_Read;
int Right_Middel_Left_Right_Read;
int Right_Bottom_Up_Down_Read;
int Right_Bottom_Left_Right_Read;

int Middel_Top_Up_Down_Read;
int Middel_Top_Left_Right_Read;
int Middel_Middel_Up_Down_Read;
int Middel_Middel_Left_Right_Read;
int Middel_Bottom_Up_Down_Read;
int Middel_Bottom_Left_Right_Read;

Servo Head_Up_Down_Servo;
Servo Head_Left_Right_Servo;

LiquidCrystal_I2C lcd(0x27, 20, 4);

int Head_Left_Right_Read_Serial;
int Head_Up_Down_Read_Serial;

int Head_Left_Right_Read_Serial_Mapped;
int Head_Up_Down_Read_Serial_Mapped;

int Head_Left_Right_Min = 30;
int Head_Left_Right_Max = 150;
int New_Head_Left_Right;

int Head_Up_Down_Min = 30;
int Head_Up_Down_Max = 150;
int New_Head_Up_Down;

float Head_Left_Right_Previous_Possision;
float Head_Up_Down_Previous_Possision;

int Head_Left_Right_Now_Read_Received;
int Head_Left_Right_Now_Read;
int Head_Left_Right_Now_Read_Received_Previous;

int Head_Up_Down_Now_Read_Received;
int Head_Up_Down_Now_Read;
int Head_Up_Down_Now_Read_Received_Previous;

int Angle_Left_Right_Mapped;
int Angle_Up_Down_Mapped;

String LeftEyeRead = "";
String RightEyeRead = "";

int Top_Possision_Right_Up_Down = 30;
int Top_Possision_Right_Left_Right = 30;

int Home_Possision_Right_Up_Down = 90;
int Home_Possision_Right_Left_Right = 30;

int Bottom_Possision_Right_Up_Down = 150;
int Bottom_Possision_Right_Left_Right = 30;

int Top_Possision_Center_Up_Down = 30;
int Top_Possision_Center_Left_Right = 90;

int Home_Possision_Center_Up_Down = 90;
int Home_Possision_Center_Left_Right = 90;

int Bottom_Possision_Center_Up_Down = 150;
int Bottom_Possision_Center_Left_Right = 90;

int Top_Possision_Left_Up_Down = 90;
int Top_Possision_Left_Left_Right = 150;

int Home_Possision_Left_Up_Down = 90;
int Home_Possision_Left_Left_Right = 150;

int Bottom_Possision_Left_Up_Down = 90;
int Bottom_Possision_Left_Left_Right = 150;

//**************************************************************

byte LCD_Speech_On_Byte[] = {
  0b11111,
  0b11111,
  0b11111,
  0b11111,
  0b11111,
  0b11111,
  0b11111,
  0b11111,
};

//**************************************************************
///////////////////////////////////////////////////////////////////////////////////

const int Eye_Power_Relay_Pin = 4;
const int Left_Eye_Reset_Relay_Pin = 9;
const int Right_Eye_Reset_Relay_Pin = 10;

//***********************************************
//            SERIAL PRINT TIMER

unsigned long Speech_Talk_Time = 100;
unsigned long Speech_Start_Time;
unsigned long Speech_Progress;
unsigned long Speech_Reset_Time = 110;

volatile bool Speech_State = false;

volatile bool Start_Speech = false;
volatile bool Stop_Speech = false;

///////////////////////////////////////////////////////////////////////////////////

////***********************************************************************************************************
void Speech_Timer() {
  Speech_Progress = millis() - Speech_Start_Time;     // Servo Head Progress

  if (Speech_Progress <= Speech_Talk_Time) {
    Speech_State = false;
  }

  if (Speech_Progress >= Speech_Talk_Time) {
    Speech_State = true;
  }

  if (Speech_Progress >= Speech_Reset_Time) {
    Speech_Start_Time = millis();
  }
}
////***********************************************************************************************************

void LCD_Speech_On() {
  Speech_Timer();
  if (Speech_State == true) {
    // write custom char across LCD (kept from original)
    for (int r = 0; r < 4; ++r) {
      lcd.setCursor(0, r);
      for (int c = 0; c < 20; ++c) {
        lcd.write((byte)0);
      }
    }
  }
}
////***********************************************************************************************************

void LCD_Speech_Off() {
  lcd.setCursor(0, 0);
  lcd.print("                    ");
  lcd.setCursor(0, 1);
  lcd.print("                    ");
  lcd.setCursor(0, 2);
  lcd.print("                    ");
  lcd.setCursor(0, 3);
  lcd.print("                    ");
}
    
////***********************************************************************************************************

void LCD_Display() {
  lcd.setCursor(10, 0);
  lcd.print("RX:");
  lcd.setCursor(10, 2);
  lcd.print("RY:");
  lcd.setCursor(10, 1);
  lcd.print("MX:");
  lcd.setCursor(10, 3);
  lcd.print("MY:");

  lcd.setCursor(0, 0);
  lcd.print("LR:");
  lcd.print(Head_Left_Right_Now_Read);

  lcd.setCursor(0, 1);
  lcd.print("UD:");
  lcd.print(Head_Up_Down_Now_Read);

  lcd.setCursor(0, 2);
  lcd.print("LR:");
  lcd.print(Head_Left_Right_Previous_Possision);
  lcd.print(" ");

  lcd.setCursor(0, 3);
  lcd.print("UD:");
  lcd.print(Head_Up_Down_Previous_Possision );
  lcd.print(" ");
}
////***********************************************************************************************************


void Head_Left_Right_Function() {
  if (Head_Left_Right_Previous_Possision >= Head_Left_Right_Max) {
    Head_Left_Right_Previous_Possision = Head_Left_Right_Max;
  }
  if (Head_Left_Right_Previous_Possision <= Head_Left_Right_Min) {
    Head_Left_Right_Previous_Possision = Head_Left_Right_Min;
  }

  if ((Head_Left_Right_Previous_Possision <= Head_Left_Right_Max) || (Head_Left_Right_Previous_Possision >= Head_Left_Right_Min)) {

    if ((Head_Left_Right_Read_Serial >= 600)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision - 5;
    }
    if ((Head_Left_Right_Read_Serial >= 500)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision - 2;
    }
    if ((Head_Left_Right_Read_Serial >= 400)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision - 1;
    }
    if ((Head_Left_Right_Read_Serial >= 340)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision - 0.5;
    }
    if ((Head_Left_Right_Read_Serial >= 335)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision - 0.05;
    }
    if ((Head_Left_Right_Read_Serial <= 335) && (Head_Left_Right_Read_Serial >= 305)) {
      Head_Left_Right_Now_Read_Received_Previous = Head_Left_Right_Previous_Possision;
    }
    if ((Head_Left_Right_Read_Serial <= 305)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision + 0.05;
    }
    if ((Head_Left_Right_Read_Serial <= 300)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision + 0.5;
    }
    if ((Head_Left_Right_Read_Serial <= 200)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision + 1;
    }
    if ((Head_Left_Right_Read_Serial <= 100)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision + 2;
    }
    if ((Head_Left_Right_Read_Serial <= 50)) {
      Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision + 5;
    }
  }

  Head_Left_Right_Servo.write((int)Head_Left_Right_Previous_Possision);
  // report position
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
  return;
}
//***************************************************************************************

void Head_Up_Down_Function() {
  if (Head_Up_Down_Previous_Possision >= Head_Up_Down_Max) {
    Head_Up_Down_Previous_Possision = Head_Up_Down_Max;
  }
  if (Head_Up_Down_Previous_Possision <= Head_Up_Down_Min) {
    Head_Up_Down_Previous_Possision = Head_Up_Down_Min;
  }

  if ((Head_Up_Down_Previous_Possision <= Head_Up_Down_Max) || (Head_Up_Down_Previous_Possision >= Head_Up_Down_Min)) {

    if ((Head_Up_Down_Read_Serial >= 480)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision - 5;
    }
    if ((Head_Up_Down_Read_Serial >= 400)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision - 2;
    }
    if ((Head_Up_Down_Read_Serial >= 350)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision - 1;
    }
    if ((Head_Up_Down_Read_Serial >= 300)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision - 1;
    }
    if ((Head_Up_Down_Read_Serial >= 260)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision - 0.05;
    }
    if ((Head_Up_Down_Read_Serial <= 255) && (Head_Up_Down_Read_Serial >= 225)) {
      Head_Up_Down_Now_Read_Received_Previous = Head_Up_Down_Previous_Possision;
    }
    if ((Head_Up_Down_Read_Serial <= 220)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision + 0.05;
    }
    if ((Head_Up_Down_Read_Serial <= 200)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision + 1;
    }
    if ((Head_Up_Down_Read_Serial <= 250)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision + 1;
    }
    if ((Head_Up_Down_Read_Serial <= 100)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision + 2;
    }
    if ((Head_Up_Down_Read_Serial <= 50)) {
      Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision + 5;
    }
  }

  Head_Up_Down_Servo.write((int)Head_Up_Down_Previous_Possision);
  // report position
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
  return;
}

void handleLCD()
{
  lcd.setCursor(10, 0);
  lcd.print("RX:");
  lcd.setCursor(10, 2);
  lcd.print("RY:");
  lcd.setCursor(10, 1);
  lcd.print("MX:");
  lcd.setCursor(10, 3);
  lcd.print("MY:");

  lcd.setCursor(0, 0);
  lcd.print("LR:");
  lcd.print(Head_Left_Right_Now_Read);
  lcd.print(" ");

  lcd.setCursor(0, 1);
  lcd.print("UD:");
  lcd.print(Head_Up_Down_Now_Read);
  lcd.print(" ");

  lcd.setCursor(0, 2);
  lcd.print("LR:");
  lcd.print(Head_Left_Right_Previous_Possision);
  lcd.print(" ");

  lcd.setCursor(0, 3);
  lcd.print("UD:");
  lcd.print(Head_Up_Down_Previous_Possision );
  lcd.print(" ");
}

// ==== FUNCTION ====
void Moving_to_Left() {
  if (Serial.available() == 0)
  {
    lcd.setCursor(13, 0);
    lcd.print(Head_Left_Right_Now_Read);

    // Map serial value (0–640) to servo angle (30–150)
    int rawVal = Head_Left_Right_Now_Read;
    if (ARD_INVERT_LR) rawVal = 640 - rawVal; // invert pixel input if needed
    int targetPosLeft = map(rawVal, 0, 640, Head_Left_Right_Min, Head_Left_Right_Max);

    lcd.setCursor(13, 2);
    lcd.print(targetPosLeft);

    unsigned long now = millis();
    if (now - lastMoveTimeLeft >= moveIntervalLeft) {
      lastMoveTimeLeft = now;

      float diffLeft = (float)targetPosLeft - Head_Left_Right_Previous_Possision;

      if (diffLeft != 0.0f) {
        // compute step with linear interpolation for smooth slowdown
        float absDiff = fabs(diffLeft);
        float ratio = 1.0;
        if (slowdownZoneLeft > 0) ratio = min(1.0f, absDiff / (float)slowdownZoneLeft);
        float step = slowStepLeft + (fastStepLeft - slowStepLeft) * ratio;
        if (absDiff < step) step = absDiff; // clamp final micro-step

        if (diffLeft > 0) Head_Left_Right_Previous_Possision += step;
        else              Head_Left_Right_Previous_Possision -= step;

        // clamp to target
        if ((diffLeft > 0 && Head_Left_Right_Previous_Possision > targetPosLeft) ||
            (diffLeft < 0 && Head_Left_Right_Previous_Possision < targetPosLeft)) {
          Head_Left_Right_Previous_Possision = targetPosLeft;
        }

        lcd.setCursor(13, 3);
        lcd.print(Head_Left_Right_Previous_Possision);

        Head_Left_Right_Servo.write((int)Head_Left_Right_Previous_Possision);
        // report HEAD_POS
        Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
        Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
      }
    }

    handleLCD();
  }
}

// ==== FUNCTION ====
void Moving_to_Right() {
  if (Serial.available() == 0)
  {
    lcd.setCursor(13, 0);
    lcd.print(Head_Left_Right_Now_Read);

    int rawVal = Head_Left_Right_Now_Read;
    if (ARD_INVERT_LR) rawVal = 640 - rawVal;
    int targetPosRight = map(rawVal, 0, 640, Head_Left_Right_Min, Head_Left_Right_Max);

    lcd.setCursor(13, 2);
    lcd.print(targetPosRight);

    unsigned long now = millis();
    if (now - lastMoveTimeRight >= moveIntervalRight) {
      lastMoveTimeRight = now;

      float diffRight = (float)targetPosRight - Head_Left_Right_Previous_Possision;

      if (diffRight != 0.0f) {
        float absDiff = fabs(diffRight);
        float ratio = 1.0;
        if (slowdownZoneRight > 0) ratio = min(1.0f, absDiff / (float)slowdownZoneRight);
        float step = slowStepRight + (fastStepRight - slowStepRight) * ratio;
        if (absDiff < step) step = absDiff;

        if (diffRight > 0) Head_Left_Right_Previous_Possision += step;
        else               Head_Left_Right_Previous_Possision -= step;

        if ((diffRight > 0 && Head_Left_Right_Previous_Possision > targetPosRight) ||
            (diffRight < 0 && Head_Left_Right_Previous_Possision < targetPosRight)) {
          Head_Left_Right_Previous_Possision = targetPosRight;
        }

        lcd.setCursor(13, 3);
        lcd.print(Head_Left_Right_Previous_Possision);

        Head_Left_Right_Servo.write((int)Head_Left_Right_Previous_Possision);
        Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
        Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
      }
    }

    handleLCD();
  }
}

// ==== FUNCTION ====
void Moving_to_Up() {
  if (Serial.available() == 0)
  {
    lcd.setCursor(13, 0);
    lcd.print(Head_Up_Down_Now_Read);

    int rawVal = Head_Up_Down_Now_Read;
    if (ARD_INVERT_UD) rawVal = 480 - rawVal;
    int targetPosUp = map(rawVal, 0, 480, Head_Up_Down_Min, Head_Up_Down_Max);

    lcd.setCursor(13, 2);
    lcd.print(targetPosUp);

    unsigned long now = millis();
    if (now - lastMoveTimeUp >= moveIntervalUp) {
      lastMoveTimeUp = now;

      float diffUp = (float)targetPosUp - Head_Up_Down_Previous_Possision;

      if (diffUp != 0.0f) {
        float absDiff = fabs(diffUp);
        float ratio = 1.0;
        if (slowdownZoneUp > 0) ratio = min(1.0f, absDiff / (float)slowdownZoneUp);
        float step = slowStepUp + (fastStepUp - slowStepUp) * ratio;
        if (absDiff < step) step = absDiff;

        if (diffUp > 0) Head_Up_Down_Previous_Possision += step;
        else               Head_Up_Down_Previous_Possision -= step;

        if ((diffUp > 0 && Head_Up_Down_Previous_Possision > targetPosUp) ||
            (diffUp < 0 && Head_Up_Down_Previous_Possision < targetPosUp)) {
          Head_Up_Down_Previous_Possision = targetPosUp;
        }

        lcd.setCursor(13, 3);
        lcd.print(Head_Up_Down_Previous_Possision);

        Head_Up_Down_Servo.write((int)Head_Up_Down_Previous_Possision);
        Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
        Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
      }
    }

    handleLCD();
  }
}

// ==== FUNCTION ====
void Moving_to_Down() {
  lcd.setCursor(13, 0);
  lcd.print(Head_Up_Down_Now_Read);

  int rawVal = Head_Up_Down_Now_Read;
  if (ARD_INVERT_UD) rawVal = 480 - rawVal;
  int targetPosDown = map(rawVal, 0, 480, Head_Up_Down_Min, Head_Up_Down_Max);

  lcd.setCursor(13, 2);
  lcd.print(targetPosDown);

  unsigned long now = millis();
  if (now - lastMoveTimeDown >= moveIntervalDown) {
    lastMoveTimeDown = now;

    float diffDown = (float)targetPosDown - Head_Up_Down_Previous_Possision;

    if (diffDown != 0.0f) {
      float absDiff = fabs(diffDown);
      float ratio = 1.0;
      if (slowdownZoneDown > 0) ratio = min(1.0f, absDiff / (float)slowdownZoneDown);
      float step = slowStepDown + (fastStepDown - slowStepDown) * ratio;
      if (absDiff < step) step = absDiff;

      if (diffDown > 0) Head_Up_Down_Previous_Possision += step;
      else               Head_Up_Down_Previous_Possision -= step;

      if ((diffDown > 0 && Head_Up_Down_Previous_Possision > targetPosDown) ||
          (diffDown < 0 && Head_Up_Down_Previous_Possision < targetPosDown)) {
        Head_Up_Down_Previous_Possision = targetPosDown;
      }

      lcd.setCursor(13, 3);
      lcd.print(Head_Up_Down_Previous_Possision);

      Head_Up_Down_Servo.write((int)Head_Up_Down_Previous_Possision);
      Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
      Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
    }
  }

  handleLCD();
}

// ==== FUNCTION ====
void Moving_to_Forward() {
  if (Serial.available() == 0)
  {
    lcd.setCursor(13, 0);
    lcd.print(Head_Left_Right_Now_Read);

    int rawVal = Head_Left_Right_Now_Read;
    if (ARD_INVERT_LR) rawVal = 640 - rawVal;
    int targetPosForward = map(rawVal, 0, 640, Head_Left_Right_Min, Head_Left_Right_Max);

    lcd.setCursor(13, 2);
    lcd.print(targetPosForward);

    unsigned long now = millis();
    if (now - lastMoveTimeForward >= moveIntervalForward) {
      lastMoveTimeForward = now;

      float diffForward = (float)targetPosForward - Head_Left_Right_Previous_Possision;

      if (diffForward != 0.0f) {
        float absDiff = fabs(diffForward);
        float ratio = 1.0;
        if (slowdownZoneForward > 0) ratio = min(1.0f, absDiff / (float)slowdownZoneForward);
        float step = slowStepForward + (fastStepForward - slowStepForward) * ratio;
        if (absDiff < step) step = absDiff;

        if (diffForward > 0) Head_Left_Right_Previous_Possision += step;
        else               Head_Left_Right_Previous_Possision -= step;

        if ((diffForward > 0 && Head_Left_Right_Previous_Possision > targetPosForward) ||
            (diffForward < 0 && Head_Left_Right_Previous_Possision < targetPosForward)) {
          Head_Left_Right_Previous_Possision = targetPosForward;
        }

        lcd.setCursor(13, 3);
        lcd.print(Head_Left_Right_Previous_Possision);

        Head_Left_Right_Servo.write((int)Head_Left_Right_Previous_Possision);
        Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
        Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
      }
    }

    handleLCD();
  }
}

// ==== FUNCTION ====
void Moving_to_Straight() {
  if (Serial.available() == 0)
  {
    lcd.setCursor(13, 0);
    lcd.print(Head_Up_Down_Now_Read);

    int rawVal = Head_Up_Down_Now_Read;
    if (ARD_INVERT_UD) rawVal = 480 - rawVal;
    int targetPosStraight = map(rawVal, 0, 480, Head_Up_Down_Min, Head_Up_Down_Max);

    lcd.setCursor(13, 2);
    lcd.print(targetPosStraight);

    unsigned long now = millis();
    if (now - lastMoveTimeStraight >= moveIntervalStraight) {
      lastMoveTimeStraight = now;

      float diffStraight = (float)targetPosStraight - Head_Up_Down_Previous_Possision;

      if (diffStraight != 0.0f) {
        float absDiff = fabs(diffStraight);
        float ratio = 1.0;
        if (slowdownZoneStraight > 0) ratio = min(1.0f, absDiff / (float)slowdownZoneStraight);
        float step = slowStepStraight + (fastStepStraight - slowStepStraight) * ratio;
        if (absDiff < step) step = absDiff;

        if (diffStraight > 0) Head_Up_Down_Previous_Possision += step;
        else               Head_Up_Down_Previous_Possision -= step;

        if ((diffStraight > 0 && Head_Up_Down_Previous_Possision > targetPosStraight) ||
            (diffStraight < 0 && Head_Up_Down_Previous_Possision < targetPosStraight)) {
          Head_Up_Down_Previous_Possision = targetPosStraight;
        }

        lcd.setCursor(13, 3);
        lcd.print(Head_Up_Down_Previous_Possision);

        Head_Up_Down_Servo.write((int)Head_Up_Down_Previous_Possision);
        Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
        Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
      }
    }

    handleLCD();
  }
}

// ===== New helper functions for small/extreme direct moves =====
void directSmallLeft() {
  Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision + SMALL_STEP;
  if (Head_Left_Right_Previous_Possision < Head_Left_Right_Min) Head_Left_Right_Previous_Possision = Head_Left_Right_Min;
  Head_Left_Right_Servo.write((int)Head_Left_Right_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}

void directSmallRight() {
  Head_Left_Right_Previous_Possision = Head_Left_Right_Previous_Possision - SMALL_STEP;
  if (Head_Left_Right_Previous_Possision > Head_Left_Right_Max) Head_Left_Right_Previous_Possision = Head_Left_Right_Max;
  Head_Left_Right_Servo.write((int)Head_Left_Right_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}

void directSmallUp() {
  Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision + SMALL_STEP;
  if (Head_Up_Down_Previous_Possision < Head_Up_Down_Min) Head_Up_Down_Previous_Possision = Head_Up_Down_Min;
  Head_Up_Down_Servo.write((int)Head_Up_Down_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}

void directSmallDown() {
  Head_Up_Down_Previous_Possision = Head_Up_Down_Previous_Possision - SMALL_STEP;
  if (Head_Up_Down_Previous_Possision > Head_Up_Down_Max) Head_Up_Down_Previous_Possision = Head_Up_Down_Max;
  Head_Up_Down_Servo.write((int)Head_Up_Down_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}

void directExtremeLeft() {
  Head_Left_Right_Previous_Possision = Head_Left_Right_Max;
  Head_Left_Right_Servo.write((int)Head_Left_Right_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}

void directExtremeRight() {
  Head_Left_Right_Previous_Possision = Head_Left_Right_Min;
  Head_Left_Right_Servo.write((int)Head_Left_Right_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}

void directExtremeUp() {
  Head_Up_Down_Previous_Possision = Head_Up_Down_Max;
  Head_Up_Down_Servo.write((int)Head_Up_Down_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}

void directExtremeDown() {
  Head_Up_Down_Previous_Possision = Head_Up_Down_Min;
  Head_Up_Down_Servo.write((int)Head_Up_Down_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}


void forward_center() {
  Head_Left_Right_Previous_Possision = Home_Possision_Center_Left_Right;
  Head_Left_Right_Servo.write((int)Head_Left_Right_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}


void straight_center() {
  Head_Up_Down_Previous_Possision = Home_Possision_Center_Up_Down;
  Head_Up_Down_Servo.write((int)Head_Up_Down_Previous_Possision);
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}


//===========================================================================

void setup()
{
  Serial.begin(9600);
  while (!Serial) { // wait till Serial (for native USB boards)
    ; // noop
  }
  Serial.println("Tracking System Is Starting....");
  serialOne.begin(9600);
  serialTwo.begin(9600);
  serialCheck.begin(9600);

  pinMode(Eye_Power_Relay_Pin, OUTPUT);
  pinMode(Left_Eye_Reset_Relay_Pin, OUTPUT);
  pinMode(Right_Eye_Reset_Relay_Pin, OUTPUT);

  lcd.init();
  lcd.backlight();

  Serial.println("");
  Serial.println(" .__________________________.");
  Serial.println(" | Sytem has Started....    |");
  Serial.println(" | Please, make sure that   |");
  Serial.println(" | the WIFI is ON & Running |");
  Serial.println(" |         !!!!!!           |");
  Serial.println(" |__________________________|");

  Speech_Start_Time = millis();

  lcd.createChar(0, LCD_Speech_On_Byte);
  lcd.clear();

  digitalWrite(Eye_Power_Relay_Pin, LOW);
  digitalWrite(Left_Eye_Reset_Relay_Pin, LOW);
  digitalWrite(Right_Eye_Reset_Relay_Pin, LOW);
  delay (2000);
  digitalWrite(Eye_Power_Relay_Pin, HIGH);
  delay (2000);
  digitalWrite(Left_Eye_Reset_Relay_Pin, HIGH);
  delay (2000);
  digitalWrite(Right_Eye_Reset_Relay_Pin, HIGH);

  lcd.setCursor(0, 0);
  lcd.print("Face System Starting");
  lcd.setCursor(0, 1);
  lcd.print("All Good....");
  lcd.setCursor(0, 2);
  lcd.print("Running....");
  delay(2000);
  lcd.clear();

  Head_Up_Down_Servo.attach(7);
  Head_Left_Right_Servo.attach(6);

  // initialize home positions
  Head_Up_Down_Servo.write(Home_Possision_Left_Up_Down);
  Head_Left_Right_Servo.write(Home_Possision_Left_Left_Right);

  Head_Left_Right_Previous_Possision = Home_Possision_Left_Left_Right;
  Head_Up_Down_Previous_Possision = Home_Possision_Left_Up_Down;

  // initial HEAD_POS broadcast
  Serial.print("HEAD_POS X:"); Serial.print((int)Head_Left_Right_Previous_Possision);
  Serial.print(" Y:"); Serial.println((int)Head_Up_Down_Previous_Possision);
}

//===========================================================================

void loop() {
  // read from software serial lines when hardware Serial idle
  if (Serial.available() == 0)
  {
    serialOne.listen(); // listening on Serial One
    while (serialOne.available() > 0) {
      inByteLeft = serialOne.readStringUntil('\n');
      Serial.println(inByteLeft);
    }

    serialTwo.listen(); // listening on Serial Two
    while (serialTwo.available() > 0) {
      inByteRight = serialTwo.readStringUntil('\n');
      Serial.println(inByteRight);
    }
  }
  else if (Serial.available() > 0)
  {
    handleLCD();

    // parse a command char and possible integer value after it
    // we read the char once to avoid consuming multiple bytes accidentally
    char cmd = Serial.read();
    if (cmd == '\n' || cmd == '\r') {
      // skip empty line / whitespace
    } else {
      // handle lowercase direct move commands and M step setter
      if (cmd == 'l') { directSmallLeft(); }
      else if (cmd == 'r') { directSmallRight(); }
      else if (cmd == 'u') { directSmallUp(); }
      else if (cmd == 'd') { directSmallDown(); }
//      else if (cmd == 'q') { directExtremeLeft(); }
//      else if (cmd == 'e') { directExtremeRight(); }
//      else if (cmd == 'w') { directExtremeUp(); }
//      else if (cmd == 's') { directExtremeDown(); }
//      else if (cmd == 'f') { forward_center(); }
      else if (cmd == 's') { straight_center(); }

      else if (cmd == 'q') { //directExtremeLeft(); 
        Left_Left_Right_Serial_Read = 640;
        Head_Left_Right_Now_Read = Left_Left_Right_Serial_Read;
        MoveRight = MoveUp = MoveDown = MoveStraight = MoveForward = false;
        lcd.setCursor(13, 1);
        lcd.print(Head_Left_Right_Now_Read);
        MoveLeft = true;
      }
      else if (cmd == 'e') { //directExtremeRight(); 
        Right_Left_Right_Serial_Read = 0;
        Head_Left_Right_Now_Read = Right_Left_Right_Serial_Read;
        MoveLeft = MoveUp = MoveDown = MoveStraight = MoveForward = false;
        lcd.setCursor(13, 1);
        lcd.print(Head_Left_Right_Now_Read);
        MoveRight = true;
      }

      
      
      else if (cmd == 'w') { //directExtremeUp(); 
        Up_Up_Down_Serial_Read = 480;
        Head_Up_Down_Now_Read = Up_Up_Down_Serial_Read;
        MoveLeft = MoveRight = MoveDown = MoveStraight = MoveForward = false;
        lcd.setCursor(13, 1);
        lcd.print(Head_Up_Down_Now_Read);
        MoveUp = true;
      }
      else if (cmd == 's') { //directExtremeDown(); 
        Down_Up_Down_Serial_Read = 0;
        Head_Up_Down_Now_Read = Down_Up_Down_Serial_Read;
        MoveLeft = MoveRight = MoveUp = MoveStraight = MoveForward = false;
        lcd.setCursor(13, 1);
        lcd.print(Head_Up_Down_Now_Read);
        MoveDown = true;
      }
      else if (cmd == 'f') { //forward_center(); 
          Forward_Left_Right_Serial_Read = 320; 
          Head_Left_Right_Now_Read = Forward_Left_Right_Serial_Read; 
          MoveLeft = MoveRight = MoveUp = MoveDown = MoveStraight = false;
          lcd.setCursor(13, 1);
          lcd.print(Head_Left_Right_Now_Read);
          MoveForward = true;
          

          
      }
      
      else if (cmd == 'p') { //forward_center(); 
          Straight_Up_Down_Serial_Read = 240;
          Head_Up_Down_Now_Read = Straight_Up_Down_Serial_Read;
          MoveLeft = MoveRight = MoveUp = MoveDown = MoveForward = false;
          lcd.setCursor(13, 1);
          lcd.print(Head_Up_Down_Now_Read);
          MoveStraight = true;
      }


      else if (cmd == 'M') {
        int v = Serial.parseInt();
        if (v > 0 && v <= 90) {
          SMALL_STEP = v;
          Serial.print("SMALL_STEP "); Serial.println(SMALL_STEP);
        }
      }
      else {
        switch (cmd) {
          case 'X':
            MoveLeft = MoveRight = MoveUp = MoveDown = MoveStraight = MoveForward = false;
            Head_Left_Right_Read_Serial = Serial.parseInt();
            Head_Left_Right_Now_Read = Head_Left_Right_Read_Serial;
            // no changes to logic here, but invert the input mapping if requested:
            Head_Left_Right_Function();
            break;

          case 'Y':
            MoveLeft = MoveRight = MoveUp = MoveDown = MoveStraight = MoveForward = false;
            Head_Up_Down_Read_Serial = Serial.parseInt();
            Head_Up_Down_Now_Read = Head_Up_Down_Read_Serial;
            Head_Up_Down_Function();
            break;

          case 'L': // Left (legacy)
            Left_Left_Right_Serial_Read = Serial.parseInt();
            Head_Left_Right_Now_Read = Left_Left_Right_Serial_Read;
            MoveRight = MoveUp = MoveDown = MoveStraight = MoveForward = false;
            lcd.setCursor(13, 1);
            lcd.print(Head_Left_Right_Now_Read);
            MoveLeft = true;
            break;

          case 'F': // Forward
            Forward_Left_Right_Serial_Read = Serial.parseInt();
            Head_Left_Right_Now_Read = Forward_Left_Right_Serial_Read;
            MoveLeft = MoveRight = MoveUp = MoveDown = MoveStraight = false;
            lcd.setCursor(13, 1);
            lcd.print(Head_Left_Right_Now_Read);
            MoveForward = true;
            break;

          case 'G': // Right (legacy)
            Right_Left_Right_Serial_Read = Serial.parseInt();
            Head_Left_Right_Now_Read = Right_Left_Right_Serial_Read;
            MoveLeft = MoveUp = MoveDown = MoveStraight = MoveForward = false;
            lcd.setCursor(13, 1);
            lcd.print(Head_Left_Right_Now_Read);
            MoveRight = true;
            break;

          case 'D': // Straight
            Straight_Up_Down_Serial_Read = Serial.parseInt();
            Head_Up_Down_Now_Read = Straight_Up_Down_Serial_Read;
            MoveLeft = MoveRight = MoveUp = MoveDown = MoveForward = false;
            lcd.setCursor(13, 1);
            lcd.print(Head_Up_Down_Now_Read);
            MoveStraight = true;
            break;

          case 'A': // Up (legacy)
            Up_Up_Down_Serial_Read = Serial.parseInt();
            Head_Up_Down_Now_Read = Up_Up_Down_Serial_Read;
            MoveLeft = MoveRight = MoveDown = MoveStraight = MoveForward = false;
            lcd.setCursor(13, 1);
            lcd.print(Head_Up_Down_Now_Read);
            MoveUp = true;
            break;

          case 'J': // Down (legacy)
            Down_Up_Down_Serial_Read = Serial.parseInt();
            Head_Up_Down_Now_Read = Down_Up_Down_Serial_Read;
            MoveLeft = MoveRight = MoveUp = MoveStraight = MoveForward = false;
            lcd.setCursor(13, 1);
            lcd.print(Head_Up_Down_Now_Read);
            MoveDown = true;
            break;

          default:
        break;
        }
      }
    }

    handleLCD();
  }

  // perform movements according to flags (non-blocking)
  if (MoveLeft) Moving_to_Left();
  if (MoveRight) Moving_to_Right();
  if (MoveStraight) Moving_to_Straight();
  if (MoveForward) Moving_to_Forward();
  if (MoveUp) Moving_to_Up();
  if (MoveDown) Moving_to_Down();
}
