#include <Wire.h>
#include <Adafruit_VL53L0X.h>

#define I2C_SDA 21
#define I2C_SCL 22

#define SENSOR1_XSHUT 13
#define SENSOR2_XSHUT 14
#define SENSOR1_ADDR 0x30
#define SENSOR2_ADDR 0x31

Adafruit_VL53L0X sensor1;
Adafruit_VL53L0X sensor2;

int dist1 = -1, dist2 = -1;
bool oor1 = false, oor2 = false;
bool ok1 = false, ok2 = false;
unsigned long lastRead = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Wire.begin(I2C_SDA, I2C_SCL);

  pinMode(SENSOR1_XSHUT, OUTPUT);
  pinMode(SENSOR2_XSHUT, OUTPUT);
  digitalWrite(SENSOR1_XSHUT, LOW);
  digitalWrite(SENSOR2_XSHUT, LOW);
  delay(10);

  pinMode(SENSOR1_XSHUT, INPUT);
  delay(10);
  if (!sensor1.begin(0x29)) {
    Serial.println("Sensor 1 not found!");
  } else {
    sensor1.setAddress(SENSOR1_ADDR);
    ok1 = true;
    Serial.println("Sensor 1 ready");
  }

  pinMode(SENSOR2_XSHUT, INPUT);
  delay(10);
  if (!sensor2.begin(0x29)) {
    Serial.println("Sensor 2 not found!");
  } else {
    sensor2.setAddress(SENSOR2_ADDR);
    ok2 = true;
    Serial.println("Sensor 2 ready");
  }

  Serial.println("ESP32 ready, sending to Jetson...");
}

void readSensors() {
  if (ok1) {
    VL53L0X_RangingMeasurementData_t m;
    sensor1.rangingTest(&m, false);
    if (m.RangeStatus != 4) {
      dist1 = m.RangeMilliMeter;
      oor1 = false;
    } else {
      dist1 = -1;
      oor1 = true;
    }
  }
  if (ok2) {
    VL53L0X_RangingMeasurementData_t m;
    sensor2.rangingTest(&m, false);
    if (m.RangeStatus != 4) {
      dist2 = m.RangeMilliMeter;
      oor2 = false;
    } else {
      dist2 = -1;
      oor2 = true;
    }
  }
}

void loop() {
  unsigned long now = millis();
  if (now - lastRead >= 200) {
    lastRead = now;
    readSensors();

    String data = "{\"sensor1\":" + String(dist1) +
                  ",\"sensor2\":" + String(dist2) +
                  ",\"oor1\":" + String(oor1 ? "true" : "false") +
                  ",\"oor2\":" + String(oor2 ? "true" : "false") +
                  ",\"ok1\":" + String(ok1 ? "true" : "false") +
                  ",\"ok2\":" + String(ok2 ? "true" : "false") + "}";

    Serial.println("");               // dummy line - primes the TX
    delayMicroseconds(500);           // let TX line stabilize
    Serial.println(data);             // real data - clean signal
  }
}
