#include <LiquidCrystal_I2C.h>
#include<Servo.h>
#include<Wire.h>

LiquidCrystal_I2C lcd(0x3F,16,2);
Servo myServo;

int angle;
void setup() {

Serial.begin(9600);
myServo.attach(9);
lcd.init();
lcd.backlight();
}

void loop() {
   if (Serial.available() > 0) { // Check if there is data available to read
    String data = Serial.readStringUntil('\n'); // Read the data until a newline character is received
    Serial.println(data); // Print the data to the serial monitor

      if (data.indexOf("cardboard") != -1) {
      angle = 30; 
      }
      else if (data.indexOf("plastic") != -1) {
      angle = 60;
         } 
       else if (data.indexOf("metal") != -1) {
      angle = 90;
  } 
  else {
    angle = 0;
  }
      if (angle > 180) angle = 0; 
      myServo.write(angle);
      Serial.println("Servo angle set to " + String(angle) + " degrees.");
    }
  }
