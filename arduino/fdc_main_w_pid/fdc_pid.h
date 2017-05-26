#ifndef FDC_PID_H
#define FDC_PID_H

#include <stdint.h>
#include "fdc_cfg.h"

#define CFG_PID_VAR_TYPE float
#define CFG_PID_DEFAULT_KP 0.04
#define CFG_PID_DEFAULT_KI 0.02
#define CFG_PID_DEFAULT_KD 0.01
#define CFG_PID_DEFAULT_I_TERM_MAX 200
#define CFG_PID_CTRL_VAR_MAX 1850
#define CFG_PID_CTRL_VAR_MIN 1600


extern CFG_PID_VAR_TYPE fdc_pid_set_point;

extern volatile CFG_PID_VAR_TYPE fdc_pid_process_variable; // Encoder value, read in interrupt context
extern volatile CFG_PID_VAR_TYPE fdc_pid_control_variable; // ESC control value, calculated in interrupt context
extern volatile CFG_PID_VAR_TYPE fdc_pid_corrected_control_variable; // Corrected ESC control value, calculated in interrupt context
extern volatile CFG_PID_VAR_TYPE fdc_pid_error; // Variable storing current cycle error, calculated in interrupt context
extern volatile CFG_PID_VAR_TYPE fdc_pid_last_error; // Record error for next cycle derivative gain calculation, assigned in interrupt context
extern volatile CFG_PID_VAR_TYPE fdc_pid_p_term;
extern volatile CFG_PID_VAR_TYPE fdc_pid_i_term;
extern volatile CFG_PID_VAR_TYPE fdc_pid_d_term;



#define CFG_PID_TUNING true
#if CFG_PID_TUNING
extern CFG_PID_VAR_TYPE FDC_PID_OFFSET;
extern CFG_PID_VAR_TYPE FDC_PID_KP;
extern CFG_PID_VAR_TYPE FDC_PID_KI;
extern CFG_PID_VAR_TYPE FDC_PID_KD;
extern CFG_PID_VAR_TYPE FDC_PID_CTRL_VAR_MAX;
extern CFG_PID_VAR_TYPE FDC_PID_CTRL_VAR_MIN;
extern CFG_PID_VAR_TYPE FDC_PID_I_TERM_MAX;

#else // CFG_PID_TUNING
static const CFG_PID_VAR_TYPE FDC_PID_OFFSET = 1700;
static const CFG_PID_VAR_TYPE FDC_PID_KP = CFG_PID_DEFAULT_KP; // Always constant. To be configure.
static const CFG_PID_VAR_TYPE FDC_PID_KI = CFG_PID_DEFAULT_KI; // Always constant. To be configure.
static const CFG_PID_VAR_TYPE FDC_PID_KD = CFG_PID_DEFAULT_KD; // Always constant. To be configure.
static const CFG_PID_VAR_TYPE FDC_PID_CTRL_VAR_MAX = 1750;
static const CFG_PID_VAR_TYPE FDC_PID_CTRL_VAR_MIN = 1650;
static const CFG_PID_VAR_TYPE FDC_PID_I_TERM_MAX = 50;

#endif // CFG_PID_TUNING

/*
Last Error = Error
Error = Set Point – Process Variable
Integral = Integral + Error
Derivative = Error – Last Error
Control Variable = (Kp * Error) + (Ki * Integral) + (Kd * Derivative)
*/

#define FDC_PID_INTERRUPT_LOOP \
    do { \
      fdc_pid_error = fdc_pid_set_point - fdc_pid_process_variable;                                                               \
                                                                                                                                  \
      /* Calculate Proportional term */                                                                                           \
      fdc_pid_p_term = FDC_PID_KP * fdc_pid_error;                                                                                \
                                                                                                                                  \
      /* Calculate Integral term */                                                                                               \
      fdc_pid_i_term += FDC_PID_KI * fdc_pid_error;                                                                               \
      if (fdc_pid_i_term > FDC_PID_I_TERM_MAX) {                                                                                  \
        fdc_pid_i_term = FDC_PID_I_TERM_MAX;                                                                                      \
      }                                                                                                                           \
      if (fdc_pid_i_term < - FDC_PID_I_TERM_MAX) {                                                                                \
        fdc_pid_i_term = - FDC_PID_I_TERM_MAX;                                                                                    \
      }                                                                                                                           \
                                                                                                                                  \
      /* Calculate Derivative term */                                                                                             \
      fdc_pid_d_term = FDC_PID_KD * (fdc_pid_error - fdc_pid_last_error);                                                         \
                                                                                                                                  \
      /* Update last error value */                                                                                               \
      fdc_pid_last_error = fdc_pid_error;                                                                                         \
                                                                                                                                  \
      /* Calculate control variable */                                                                                            \
      fdc_pid_control_variable = fdc_pid_p_term + fdc_pid_i_term + fdc_pid_d_term;                                                \
                                                                                                                                  \
      /* Apply offset and saturate control variable */                                                                            \
      fdc_pid_corrected_control_variable = fdc_pid_control_variable + FDC_PID_OFFSET;                                             \
      if (fdc_pid_corrected_control_variable > FDC_PID_CTRL_VAR_MAX) {                                                            \
        fdc_pid_corrected_control_variable = FDC_PID_CTRL_VAR_MAX;                                                                \
      }                                                                                                                           \
      if (fdc_pid_corrected_control_variable < FDC_PID_CTRL_VAR_MIN) {                                                            \
        fdc_pid_corrected_control_variable = FDC_PID_CTRL_VAR_MIN;                                                                \
      }                                                                                                                           \
                                                                                                                                  \
      /* Stop engine when Set Point is 0 */                                                                                       \
      if (fdc_pid_set_point == 0) {                                                                                               \
        fdc_pid_corrected_control_variable = 1500;                                                                                \
      }                                                                                                                           \
    } while(0);

#define FDC_PID_MAIN_LOOP \
    do { \
    } while (0);

#endif // FDC_PID_H

