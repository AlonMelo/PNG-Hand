from hands import *
from src.openhand_node import hands

reset_servo_ids = [1]
reset_servo_type = 'X_series'
# reset_servo_port = "COM3"
reset_servo_port = "/dev/ttyUSB0"
reset_motor_offset = [0.0]
reset_motor = hands.Reset_Motor(reset_servo_port,  reset_servo_ids, 'XM')

def user_input1():
    """ open or close """
    ans = input('Continue? : o/c ')
    if ans == 'o':
        return False
    else:
        return True

def user_input2():
    """ Check to see if user wants to continue """
    ans = input('Continue? : y/n ')
    if ans == 'n':
        return False
    else:
        return True

def test_pos(motor_object):
    finito = True
    while finito:
        bool_test = user_input1()
        if bool_test:
            reset_motor.close_torque(0.2)
            # reset_motor.diagnostics()
            # reset_motor.alon_incoder(0, 1000)
        else:
            reset_motor.close_torque(-0.2)
            # reset_motor.alon_incoder(0, 0)
        finito = user_input2()
    # go back to initial position
    reset_motor.reset()

test_pos(reset_motor)
