#include "TLE5012Sensor.h"
#include "TLx493D_inc.hpp"
#include "config.h"
#include <SimpleFOC.h>
#include <PID_v1.h>

#define PIN_SPI1_SS0 94
#define PIN_SPI1_MOSI 69
#define PIN_SPI1_MISO 95
#define PIN_SPI1_SCK 68

tle5012::SPIClass3W tle5012::SPI3W1(2);
TLE5012Sensor tle5012Sensor(&SPI3W1, PIN_SPI1_SS0, PIN_SPI1_MISO, PIN_SPI1_MOSI, PIN_SPI1_SCK);

BLDCMotor motor = BLDCMotor(7, 0.24, 360, 0.000133);
const int U = 11, V = 10, W = 9, EN_U = 6, EN_V = 5, EN_W = 3;
BLDCDriver3PWM driver = BLDCDriver3PWM(U, V, W, EN_U, EN_V, EN_W);

using namespace ifx::tlx493d;
TLx493D_A2B6 dut(Wire1, TLx493D_IIC_ADDR_A0_e);
const int CALIBRATION_SAMPLES = 20;
double xOffset = 0, yOffset = 0, zOffset = 0;

float target_voltage = 0;
float target_final = 0;
bool gripping = false;
int grip_count = 0;
bool pid_active = false;

const double Kp = 0.5, Ki = 0.5, Kd = 0.01;
double pid_input = 0, pid_output = 0, pid_setpoint = 0;
double z_before = 0, z_after = 0;
PID pidController(&pid_input, &pid_output, &pid_setpoint, Kp, Ki, Kd, DIRECT);

String serial_command = "";
String material = "unknown";
float THRESHOLD_Z = 0.5;
float THRESHOLD_DZ = 0.5;

void setup() {
  Serial.begin(115200);
  SimpleFOCDebug::enable(&Serial);

  tle5012Sensor.init();
  motor.linkSensor(&tle5012Sensor);

  driver.voltage_power_supply = 12;
  driver.voltage_limit = 6;
  driver.init();
  motor.linkDriver(&driver);

  motor.voltage_sensor_align = 2;
  motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
  motor.controller = MotionControlType::torque;
  motor.init();
  motor.initFOC();

  dut.begin();
  calibrateSensor();

  pidController.SetOutputLimits(-2, -1);
  pidController.SetMode(AUTOMATIC);
  pidController.SetSampleTime(10);

  delay(1000);
}

void loop() {
  handleSerial();

  double x, y, z;
  dut.setSensitivity(TLx493D_FULL_RANGE_e);
  dut.getMagneticField(&x, &y, &z);
  x -= xOffset;
  y -= yOffset;
  z -= zOffset;

  z_before = z_after;
  z_after = z;
  double dz = fabs(z_after - z_before);

  if (pid_active) {
    if (abs(z) > THRESHOLD_Z || dz > THRESHOLD_DZ) {
      pid_input = z;
      pidController.Compute();
      target_voltage = pid_output;
      target_final = pid_output;
      grip_count++;
      if (grip_count > 30) {
        gripping = true;
        pid_active = false;
      }
    } else {
      target_voltage = -1;
      grip_count = 0;
    }
  } else if (gripping) {
    target_voltage = target_final;
  }

  tle5012Sensor.update();
  motor.loopFOC();
  motor.move(target_voltage);

  Serial.print(tle5012Sensor.getSensorAngle());
  Serial.print(",");
  Serial.print(z);
  Serial.print(",");
  Serial.println(target_voltage);

  delay(10);
}

void calibrateSensor() {
  double sumX = 0, sumY = 0, sumZ = 0;
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    double t, x, y, z;
    dut.getMagneticFieldAndTemperature(&x, &y, &z, &t);
    sumX += x;
    sumY += y;
    sumZ += z;
    delay(10);
  }
  xOffset = sumX / CALIBRATION_SAMPLES;
  yOffset = sumY / CALIBRATION_SAMPLES;
  zOffset = sumZ / CALIBRATION_SAMPLES;
}

void handleSerial() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') {
      parseCommand(serial_command);
      serial_command = "";
    } else {
      serial_command += ch;
    }
  }
}

void parseCommand(String cmd) {
  cmd.trim();
  if (cmd == "B11") {
    pid_active = true;
    gripping = false;
    grip_count = 0;
  } else if (cmd == "B10") {
    pid_active = false;
  } else if (cmd == "B21") {
    target_voltage = 1;
    gripping = false;
    pid_active = false;
  } else if (cmd == "B20") {
    target_voltage = 0;
  } else if (cmd == "RESET") {
    target_voltage = 0;
    pid_active = false;
    gripping = false;
    grip_count = 0;
  } else if (cmd.startsWith("MATERIAL:")) {
    material = cmd.substring(9);
    material.toLowerCase();
    if (material == "plastic") {
      THRESHOLD_Z = 0.3;
      THRESHOLD_DZ = 0.3;
    } else {
      THRESHOLD_Z = 0.5;
      THRESHOLD_DZ = 0.5;
    }
  }
}
