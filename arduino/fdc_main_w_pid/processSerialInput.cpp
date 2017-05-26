#include <Arduino.h>
#include "processSerialInput.h"
#include "fdc_cfg.h"

serial_input_t serialInput;

#define PROCESS_SERIAL_INPUT_DEBUG_ENABLE  true

#if (CFG_MASTER_DEBUG_ENABLE && PROCESS_SERIAL_INPUT_DEBUG_ENABLE)
# define SERIAL_DEBUG_PRINT(x)    Serial.print(x)
# define SERIAL_DEBUG_PRINTLN(x)  Serial.println(x)
#else
# define SERIAL_DEBUG_PRINT(x)
# define SERIAL_DEBUG_PRINTLN(x)
#endif

typedef enum {
//  SERIAL_STATE_WAIT_END_OF_INVALID,
  
  SERIAL_STATE_RX_CMD_CODE,
  SERIAL_STATE_RX_TEST_CODE,

  SERIAL_STATE_RX_VALUE_TILL_NEWLINE,
} serial_rx_state_t;


// TODO: consider optimize the use of serialInputBuffer
void processSerialInput() {
  
  static char rxChar;
//  static serial_rx_state_t serialRxState = SERIAL_STATE_WAIT_END_OF_INVALID;
  static serial_rx_state_t serialRxState = SERIAL_STATE_RX_CMD_CODE;
  static serial_input_t serialInputBuffer;

  while (Serial.available()) {
    rxChar = (char) Serial.read();
    switch (serialRxState) {

      case SERIAL_STATE_RX_CMD_CODE: {
        if (rxChar == 'm') {
          serialInputBuffer.command = SERIAL_INPUT_CMD_SWITCH_MODE;
          serialRxState = SERIAL_STATE_RX_VALUE_TILL_NEWLINE;
        }
        else if (rxChar == 't') {
          serialRxState = SERIAL_STATE_RX_TEST_CODE;
        }
        else if (rxChar == 'f') {
          serialInputBuffer.command = SERIAL_INPUT_CMD_SET_FORWARD;
          serialRxState = SERIAL_STATE_RX_VALUE_TILL_NEWLINE;
        }
        else {
          serialInputBuffer.command = SERIAL_INPUT_CMD_INVALID;
          serialRxState = SERIAL_STATE_RX_VALUE_TILL_NEWLINE;
        }
      }
      break;

      case SERIAL_STATE_RX_TEST_CODE: {
        if (rxChar == '1') {
          serialInputBuffer.command = SERIAL_INPUT_CMD_TEST_ESC;
          serialRxState = SERIAL_STATE_RX_VALUE_TILL_NEWLINE;
        }
        else
        if (rxChar == '2') {
          serialInputBuffer.command = SERIAL_INPUT_CMD_TEST_ENCODER;
          serialRxState = SERIAL_STATE_RX_VALUE_TILL_NEWLINE;
        }
        else
        if (rxChar == '3') {
          serialInputBuffer.command = SERIAL_INPUT_CMD_TEST_SONAR;
          serialRxState = SERIAL_STATE_RX_VALUE_TILL_NEWLINE;
        }
      }
      break;
      
      default: {
        if ((rxChar >= '0') && (rxChar <= '9')) {
          serialInputBuffer.value *= 10;
          serialInputBuffer.value += (unsigned int) rxChar - 0x30;
        }
        else if (rxChar == '\n') {
          // End of command
          serialInput.command = serialInputBuffer.command;
          serialInput.value = serialInputBuffer.value;
          serialInputBuffer.value = 0;
          serialRxState = SERIAL_STATE_RX_CMD_CODE;
        }
        else {
          serialInput.command = SERIAL_INPUT_CMD_INVALID;
          serialRxState = SERIAL_STATE_RX_VALUE_TILL_NEWLINE;
        }
      }
    }
  }
}

