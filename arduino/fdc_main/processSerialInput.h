#ifndef PROCESS_SERIAL_INPUT_H
#define PROCESS_SERIAL_INPUT_H

#include <stdint.h>

typedef enum {
  SERIAL_INPUT_CMD_INVALID       = 0,
  SERIAL_INPUT_CMD_SWITCH_MODE   = 1,
  SERIAL_INPUT_CMD_SET_FORWARD   = 2,
  // SERIAL_INPUT_CMD_SET_BACKWARD  = 3,
  SERIAL_INPUT_CMD_TEST_ESC      = 11,
  SERIAL_INPUT_CMD_TEST_ENCODER  = 12,
  SERIAL_INPUT_CMD_TEST_SONAR    = 13,
} serial_input_cmd_t;

#define SERIAL_INPUT_VALUE_MODE_RUN               1
#define SERIAL_INPUT_VALUE_MODE_COMPONENT_TEST    2

typedef struct {
  serial_input_cmd_t command;
  uint32_t value;
} serial_input_t;

extern serial_input_t serialInput;

void processSerialInput();

#endif // #ifndef PROCESS_SERIAL_INPUT_H
