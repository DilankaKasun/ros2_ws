#include <Wire.h>
#include <Adafruit_VL53L0X.h>

#define I2C_SDA 21
#define I2C_SCL 22

#define RIGHT_XSHUT 13
#define LEFT_XSHUT 14
#define RIGHT_ADDR 0x30
#define LEFT_ADDR 0x31

Adafruit_VL53L0X rightSensor;
Adafruit_VL53L0X leftSensor;

int distRight = -1, distLeft = -1;
bool oorRight = false, oorLeft = false;
bool okRight = false, okLeft = false;
unsigned long lastRead = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Wire.begin(I2C_SDA, I2C_SCL);

  pinMode(RIGHT_XSHUT, OUTPUT);
  pinMode(LEFT_XSHUT, OUTPUT);
  digitalWrite(RIGHT_XSHUT, LOW);
  digitalWrite(LEFT_XSHUT, LOW);
  delay(10);

  pinMode(RIGHT_XSHUT, INPUT);
  delay(10);
  if (!rightSensor.begin(0x29)) {
    Serial.println("Right sensor not found!");
  } else {
    rightSensor.setAddress(RIGHT_ADDR);
    okRight = true;
    Serial.println("Right sensor ready");
  }

  pinMode(LEFT_XSHUT, INPUT);
  delay(10);
  if (!leftSensor.begin(0x29)) {
    Serial.println("Left sensor not found!");
  } else {
    leftSensor.setAddress(LEFT_ADDR);
    okLeft = true;
    Serial.println("Left sensor ready");
  }

  Serial.println("ESP32 ready, sending to Jetson...");
}

void readSensors() {
  if (okRight) {
    VL53L0X_RangingMeasurementData_t m;
    rightSensor.rangingTest(&m, false);
    if (m.RangeStatus != 4) {
      distRight = m.RangeMilliMeter;
      oorRight = false;
    } else {
      distRight = -1;
      oorRight = true;
    }
  }
  if (okLeft) {
    VL53L0X_RangingMeasurementData_t m;
    leftSensor.rangingTest(&m, false);
    if (m.RangeStatus != 4) {
      distLeft = m.RangeMilliMeter;
      oorLeft = false;
    } else {
      distLeft = -1;
      oorLeft = true;
    }
  }
}

void loop() {
  unsigned long now = millis();
  if (now - lastRead >= 200) {
    lastRead = now;
    readSensors();

    String data = "{\"right\":" + String(distRight) +
                  ",\"left\":" + String(distLeft) +
                  ",\"oorRight\":" + String(oorRight ? "true" : "false") +
                  ",\"oorLeft\":" + String(oorLeft ? "true" : "false") +
                  ",\"okRight\":" + String(okRight ? "true" : "false") +
                  ",\"okLeft\":" + String(okLeft ? "true" : "false") + "}";

    Serial.println("");               // dummy line - primes the TX
    delayMicroseconds(500);           // let TX line stabilize
    Serial.println(data);             // real data - clean signal
  }
}
