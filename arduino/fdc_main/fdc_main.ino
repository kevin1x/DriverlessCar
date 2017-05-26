/** ===== PERMISSION NOTICE ===================================================
  * 
  * Copyright 2017 FPT Corporation
  *
  * Permission is hereby granted, free of charge, to any person obtaining a 
  * copy of this software and associated documentation files (the "Software"), 
  * to deal in the Software without restriction, including without limitation 
  * the rights to use, copy, modify, merge, publish, distribute, sublicense, 
  * and/or sell copies of the Software, and to permit persons to whom the 
  * Software is furnished to do so, subject to the following conditions:
  *
  * The above copyright notice and this permission notice shall be included in 
  * all copies or substantial portions of the Software.
  *
  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
  * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
  * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
  * DEALINGS IN THE SOFTWARE.
  *
  * ===========================================================================
*/


/** ===== FILE & SKETCH DESCRIPTION ===========================================
  * @file     fdc_main.ino
  * @version  v1.0 2017/01/25
  *
  * @about    Main Arduino firmware for FPT Driverless Car
  *           Behavior: 
  *             Refer to accompanied documentation files.
  * 
  */


#include <TimerOne.h>
#include <Servo.h>
#include <Encoder.h>

#include "fdc_cfg.h"
#include "processSerialInput.h"

#define MAIN_DEBUG_ENABLE  true

#if (CFG_MASTER_DEBUG_ENABLE && MAIN_DEBUG_ENABLE)
# define SERIAL_DEBUG_PRINT(x)    Serial.print(x)
# define SERIAL_DEBUG_PRINTLN(x)  Serial.println(x)
#else
# define SERIAL_DEBUG_PRINT(x)
# define SERIAL_DEBUG_PRINTLN(x)
#endif

#define LED_ON LOW
#define LED_OFF HIGH
#define BUTTON_PRESSED LOW
#define BUTTON_RELEASED HIGH

#define ESC_CONTROL_VALUE_FROM_SET_FW_COMMAND_VALUE(commandValue)  (CFG_ESC_CONTROL_NEUTRAL_ON_TIME + 5 * commandValue)

typedef enum {
  SYSTEM_STATE_WAIT_BUTTON,
  SYSTEM_STATE_WAIT_SONAR,
  SYSTEM_STATE_RUN,
  SYSTEM_STATE_COMPONENT_TEST,
} sys_state_t;


static volatile unsigned long mEncoderCount = 0;
static Encoder mEncoder(CFG_ENCODER_PIN_A, CFG_ENCODER_PIN_B);

static sys_state_t mCurrentSystemState = CFG_SYSTEM_FIRST_STATE;

// FSM input:
static volatile bool mTimer1InterruptFlag = false;
// Button pressed
static unsigned long mLastSonarPulseWidth = 0;
// Serial

// FSM output:
static sys_state_t mNextSystemState;
static Servo mEscControlLer;
static uint32_t mEscControlValue = CFG_ESC_CONTROL_NEUTRAL_ON_TIME;
// LED


#define BLINK_LED() digitalWrite(CFG_BUTTON_LED_PIN, !digitalRead(CFG_BUTTON_LED_PIN))

void timer2Setup() {
  // Draft:
  // Timer2 clock source is 16,000,000 MHz
  // At maximum prescaler of 1024, this means 15625 ticks/s
  // Because Timer2 is a 8-bit timer, so the Output Compare Match value cannot be greater than 255
  // Also 15625 = 5^6
  // So the lowest round frequency we can achieve is 15625 / 125 = 125 Hz
  // With OC = 250, we have 62.5Hz, not so horrible to work with
  // Add a app level prescaler of 16, we got 3.90625Hz. Not too far from intended 4Hz.
  // Eh, good enough
  // So, prescale by 1024, so CS2[2:0] = b111. Output Compare = 250, so OCR2A = 250

  // Stop interrupts
  cli();
  
  // Clear TC2 Control Register A & B (TCCR2A, TCCR2B)
  TCCR2A = 0;
  TCCR2B = 0;

  // Clear TC2 Counter Value Register (TCNT2)
  TCNT2 = 0;
  
  // Write TC2 Control Registers to set Waveform Generation Mode (WGM) to Clear Timer on Compare Match (CTC) mode
  // WGM[2:0] = b010
  TCCR2A |= (1 << WGM21);
  
  // Write TC2 Control Register B to set Clock Select (CS) to clkIO/1024 (prescale by 1024)
  // CS[2:0] = b111
  TCCR2B |= (1 << CS20);
  TCCR2B |= (1 << CS21);
  TCCR2B |= (1 << CS22);

  // Set TC2 Output Compare Register A to 250
  OCR2A = 250;

  // Write TC2 Interrupt Mask Register to enable interrupt on Output Compare A Match
  // OCIE2A = b1
  TIMSK2 |= (1 << OCIE2A);

  // Enable interrupts
  sei();
}

ISR(TIMER2_COMPA_vect) {
  static volatile uint8_t appPrescaler;
  appPrescaler++;
  if (appPrescaler >= CFG_TIMER2_APP_PRESCALER) {
    mTimer1InterruptFlag = true;
    mEncoderCount = mEncoder.read();
    mEncoder.write(0);
    appPrescaler = 0;
  }
}

void setup() {
  Serial.begin(CFG_SERIAL_BAUD_RATE);

  // Set up buttons and LEDs
  pinMode(CFG_BUTTON_INPUT_PIN, INPUT_PULLUP);
  pinMode(CFG_BUTTON_LED_PIN, OUTPUT);
  digitalWrite(CFG_BUTTON_LED_PIN, LED_OFF);

  // Timer2 provides periodic interrupt for LED display and encoder operations
  timer2Setup();

  // ESC control set up
  mEscControlLer.attach(CFG_ESC_CONTROL_PIN);
  mEscControlLer.writeMicroseconds(CFG_ESC_CONTROL_NEUTRAL_ON_TIME);

  // Encoder set up
  mEncoder.write(0); // Ensure encoder count start at 0

  // Sonar sensor set up
  pinMode(CFG_HC_SR04_TRIG_PIN, OUTPUT);
  pinMode(CFG_HC_SR04_ECHO_PIN, INPUT);

  // Set up initial state
  mCurrentSystemState = CFG_SYSTEM_FIRST_STATE;
  mNextSystemState = CFG_SYSTEM_FIRST_STATE;

  SERIAL_DEBUG_PRINTLN("Arduino init completed");
}


void loop() {
  static uint8_t ledBlinkCounter = 0;
  bool buttonState = BUTTON_RELEASED;
  static bool lastButtonState = BUTTON_RELEASED;
  static bool sendEncoderCountEnable = false;

  processSerialInput();

  // Default control variable values:
  mNextSystemState = mCurrentSystemState;

  switch (mCurrentSystemState) {
    case SYSTEM_STATE_WAIT_BUTTON: {
      // Input 1: Button - move to WAIT_SONAR state
      buttonState = digitalRead(CFG_BUTTON_INPUT_PIN);
      if ((buttonState == BUTTON_PRESSED) && (lastButtonState == BUTTON_RELEASED)) {
        mNextSystemState = SYSTEM_STATE_WAIT_SONAR;
        SERIAL_DEBUG_PRINTLN("d Enter SYSTEM_STATE_WAIT_SONAR");
      }
      lastButtonState = buttonState;

      // Input 2: Sonar - unused

      // Input 3: Timer - blink LED
      if (mTimer1InterruptFlag) {
        ledBlinkCounter++;
        if (ledBlinkCounter >= CFG_WAIT_BUTTON_STATE_LED_BLINK_PERIOD) {
          BLINK_LED();
          ledBlinkCounter = 0;
        }
        mTimer1InterruptFlag = false;
      }

      // Input 4: Serial - update ESC control value, switch mode
      
      // Update control value so that the car can start running as soon as button and sonar conditions are met
      if (serialInput.command == SERIAL_INPUT_CMD_SET_FORWARD) {
        mEscControlValue = ESC_CONTROL_VALUE_FROM_SET_FW_COMMAND_VALUE(serialInput.value);
        SERIAL_DEBUG_PRINT("d mEscControlValue == "); SERIAL_DEBUG_PRINTLN(mEscControlValue);
        Serial.print("af"); Serial.println(serialInput.value);
      }
      if (serialInput.command == SERIAL_INPUT_CMD_SWITCH_MODE) {
        if (serialInput.value == 2) {
          mNextSystemState = SYSTEM_STATE_COMPONENT_TEST;
          lastButtonState = BUTTON_RELEASED;
          sendEncoderCountEnable = false;
          mEscControlLer.writeMicroseconds(CFG_ESC_CONTROL_NEUTRAL_ON_TIME);
          SERIAL_DEBUG_PRINTLN("d Enter SYSTEM_STATE_COMPONENT_TEST");
          Serial.println("am2");
        }
      }
    }
    break;



    case SYSTEM_STATE_WAIT_SONAR: {
      // Input 1: Button - report status change over Serial
      buttonState = digitalRead(CFG_BUTTON_INPUT_PIN);
      if (buttonState != lastButtonState) {
        Serial.print("b"); Serial.println(buttonState);
      }
      lastButtonState = buttonState;

      // Input 2: Sonar - move to RUN state
      pollSonarSensor();
      SERIAL_DEBUG_PRINT("d mLastSonarPulseWidth == "); SERIAL_DEBUG_PRINTLN(mLastSonarPulseWidth);
      if ((mLastSonarPulseWidth >= CFG_SONAR_NO_OBSTACLE_PULSE_WIDTH) || (mLastSonarPulseWidth == 0)) {
        mNextSystemState = SYSTEM_STATE_RUN;
        mEscControlLer.writeMicroseconds(mEscControlValue);
        digitalWrite(CFG_BUTTON_LED_PIN, LED_ON);
        SERIAL_DEBUG_PRINTLN("d Enter SYSTEM_STATE_RUN");
      }

      // Input 3: Timer - blink LED
      if (mTimer1InterruptFlag) {
        ledBlinkCounter++;
        if (ledBlinkCounter >= CFG_WAIT_SONAR_STATE_LED_BLINK_PERIOD) {
          BLINK_LED();
          ledBlinkCounter = 0;
        }
        mTimer1InterruptFlag = false;
      }

      // Input 4: Serial - update ESC control value, switch mode

      // Update control value so that the car can start running as soon as button and sonar conditions are met
      if (serialInput.command == SERIAL_INPUT_CMD_SET_FORWARD) {
        mEscControlValue = ESC_CONTROL_VALUE_FROM_SET_FW_COMMAND_VALUE(serialInput.value);
        SERIAL_DEBUG_PRINT("d mEscControlValue == "); SERIAL_DEBUG_PRINTLN(mEscControlValue);
        Serial.print("af"); Serial.println(serialInput.value);
      }
      if (serialInput.command == SERIAL_INPUT_CMD_SWITCH_MODE) {
        if (serialInput.value == 2) {
          mNextSystemState = SYSTEM_STATE_COMPONENT_TEST;
          lastButtonState = BUTTON_RELEASED;
          sendEncoderCountEnable = false;
          mEscControlLer.writeMicroseconds(CFG_ESC_CONTROL_NEUTRAL_ON_TIME);
          SERIAL_DEBUG_PRINTLN("d Enter SYSTEM_STATE_COMPONENT_TEST");
          Serial.println("am2");
        }
      }

      if (mNextSystemState == SYSTEM_STATE_RUN) {
        digitalWrite(CFG_BUTTON_LED_PIN, LED_ON);
      }
    }
    break;



    case SYSTEM_STATE_RUN: {
      // Input 1: Button - report status change over Serial
      buttonState = digitalRead(CFG_BUTTON_INPUT_PIN);
      if (buttonState != lastButtonState) {
        Serial.print("b"); Serial.println(buttonState);
      }
      lastButtonState = buttonState;
      
      // Input 2: Sonar - unused

      // Input 3: Timer - send encoder count
      if (mTimer1InterruptFlag) {
        // Send encoder count over last period to Jetson
        Serial.print("e"); Serial.println(mEncoderCount);
        mTimer1InterruptFlag = false;
      }

      // Input 4: Serial - update ESC control value, switch mode
      if (serialInput.command == SERIAL_INPUT_CMD_SET_FORWARD) {
        mEscControlValue = ESC_CONTROL_VALUE_FROM_SET_FW_COMMAND_VALUE(serialInput.value);
        mEscControlLer.writeMicroseconds(mEscControlValue);
        SERIAL_DEBUG_PRINT("d mEscControlValue == "); SERIAL_DEBUG_PRINTLN(mEscControlValue);
        Serial.print("af"); Serial.println(serialInput.value);
      }
      if (serialInput.command == SERIAL_INPUT_CMD_SWITCH_MODE) {
        if (serialInput.value == 2) {
          mNextSystemState = SYSTEM_STATE_COMPONENT_TEST;
          lastButtonState = BUTTON_RELEASED;
          sendEncoderCountEnable = false;
          mEscControlLer.writeMicroseconds(CFG_ESC_CONTROL_NEUTRAL_ON_TIME);
          SERIAL_DEBUG_PRINTLN("d Enter SYSTEM_STATE_COMPONENT_TEST");
          Serial.println("am2");
        }
      }
    }
    break;



    case SYSTEM_STATE_COMPONENT_TEST: {
      // Input 1: Button - report status change over Serial
      buttonState = digitalRead(CFG_BUTTON_INPUT_PIN);
      if (buttonState != lastButtonState) {
        Serial.print("b"); Serial.println(buttonState);
      }
      lastButtonState = buttonState;
      
      // Input 2: Sonar - unused

      // Input 3: Timer - blink LED and send encoder count enabled
      if (mTimer1InterruptFlag) {
        ledBlinkCounter++;
        if (ledBlinkCounter >= CFG_COMPONENT_TEST_STATE_LED_BLINK_PERIOD) {
          BLINK_LED();
          ledBlinkCounter = 0;
        }
        
        // Send encoder count over last period to Jetson
        if (sendEncoderCountEnable) {
          Serial.print("e"); Serial.println(mEncoderCount);
        }
        mTimer1InterruptFlag = false;
      }

      // Input 4: Serial - switch mode, poll sonar, update speed
      if (serialInput.command == SERIAL_INPUT_CMD_TEST_ESC) {
        // Control ESC
        SERIAL_DEBUG_PRINT("d ESC TEST "); SERIAL_DEBUG_PRINTLN(serialInput.value);
        mEscControlLer.writeMicroseconds(serialInput.value);
        Serial.print("at1"); Serial.println(serialInput.value);
      }
      if (serialInput.command == SERIAL_INPUT_CMD_TEST_ENCODER) {
        SERIAL_DEBUG_PRINTLN("d ENCODER TEST");
        sendEncoderCountEnable = !sendEncoderCountEnable;
        Serial.println("at2");
      }
      if (serialInput.command == SERIAL_INPUT_CMD_TEST_SONAR) {
        // Poll sonar
        SERIAL_DEBUG_PRINTLN("d SONAR TEST");
        Serial.println("at3");
        pollSonarSensor();
        Serial.print("s"); Serial.println(mLastSonarPulseWidth);
      }
      if (serialInput.command == SERIAL_INPUT_CMD_SWITCH_MODE) {
        // Switch mode to RUN (start with WAIT_BUTTON state)
        if (serialInput.value == 1) {
          mNextSystemState = SYSTEM_STATE_WAIT_BUTTON;
          lastButtonState = BUTTON_RELEASED;
          mEscControlLer.writeMicroseconds(CFG_ESC_CONTROL_NEUTRAL_ON_TIME);
          SERIAL_DEBUG_PRINTLN("d Enter SYSTEM_STATE_WAIT_BUTTON");
          Serial.println("am1");
        }
      }
    }
    break;



    default:
    {
      // Could only ever get here through an error
      // Reset control variable and put system back in SYSTEM_STATE_WAIT_BUTTON OR reset the Arduino
      mNextSystemState = SYSTEM_STATE_WAIT_BUTTON;
      mEscControlLer.writeMicroseconds(CFG_ESC_CONTROL_NEUTRAL_ON_TIME);
    }
  }

  mCurrentSystemState = mNextSystemState;
  serialInput.command = SERIAL_INPUT_CMD_INVALID;
}

void pollSonarSensor() {
  // Generate a 5ms pulse at Trigger pin
  digitalWrite(CFG_HC_SR04_TRIG_PIN, LOW);
  delayMicroseconds(5);
  digitalWrite(CFG_HC_SR04_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(CFG_HC_SR04_TRIG_PIN, LOW);

  // Measure the Echo pulse width
  mLastSonarPulseWidth = pulseIn(CFG_HC_SR04_ECHO_PIN, HIGH);
}

