#include "fdc_pid.h"

// Variables updated in main loop
CFG_PID_VAR_TYPE fdc_pid_set_point = 0;

// Variables updated and calculated in interrupt handler
volatile CFG_PID_VAR_TYPE fdc_pid_process_variable;
volatile CFG_PID_VAR_TYPE fdc_pid_control_variable;
volatile CFG_PID_VAR_TYPE fdc_pid_corrected_control_variable; // Corrected ESC control value, calculated in interrupt context
volatile CFG_PID_VAR_TYPE fdc_pid_error;
volatile CFG_PID_VAR_TYPE fdc_pid_last_error;
volatile CFG_PID_VAR_TYPE fdc_pid_p_term;
volatile CFG_PID_VAR_TYPE fdc_pid_i_term = 0;
volatile CFG_PID_VAR_TYPE fdc_pid_d_term;

// Variables only updated in main loop during tuning
#if CFG_PID_TUNING
CFG_PID_VAR_TYPE FDC_PID_OFFSET = ((CFG_PID_CTRL_VAR_MAX + CFG_PID_CTRL_VAR_MIN) / 2);
CFG_PID_VAR_TYPE FDC_PID_KP = CFG_PID_DEFAULT_KP;
CFG_PID_VAR_TYPE FDC_PID_KI = CFG_PID_DEFAULT_KI;
CFG_PID_VAR_TYPE FDC_PID_KD = CFG_PID_DEFAULT_KD;
CFG_PID_VAR_TYPE FDC_PID_CTRL_VAR_MAX = CFG_PID_CTRL_VAR_MAX;
CFG_PID_VAR_TYPE FDC_PID_CTRL_VAR_MIN = CFG_PID_CTRL_VAR_MIN;
CFG_PID_VAR_TYPE FDC_PID_I_TERM_MAX = CFG_PID_DEFAULT_I_TERM_MAX;
#endif // CFG_PID_TUNING



