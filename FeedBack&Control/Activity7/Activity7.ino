#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(9600);
  myServo.attach(9);
  myServo.write(0);

  Serial.println("Arduino is ready to receive angles (0-180).");
}

void loop() {
  if (Serial.available() > 0) {

    int angle = Serial.parseInt();

    if (angle >= 0 && angle <= 180) {
      myServo.write(angle);

      Serial.print("Servo moved to: ");
      Serial.println(angle);
    }
  }
}
