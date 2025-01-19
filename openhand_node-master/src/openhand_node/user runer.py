from hands import *
from src.openhand_node import hands

reset_servo_ids = [1]
reset_servo_type = 'X_series'
reset_servo_port = "COM4"
reset_motor_offset = [0.0]
reset_motor = hands.Reset_Motor(reset_servo_port, reset_servo_ids, 'XM')


def user_input1():
    """ Check to see if user wants to continue """
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


def user_input3():
    """ Check to see if user wants to continue """
    ans = input('Continue? : o/r/c ')
    if ans == 'o':
        return 1
    elif ans == 'c':
        return 2
    else:
        return True


def test_pos(motor_object):
    finito = True
    while finito:
        motor_object.motorDir[0] = -1
        bool_test = user_input3()
        if bool_test:
            motor_object.motorDir[0] = reset_motor.motorDir[0] * (-1)
            motor_object.release_()
        elif bool_test == 'o':
            motor_object.motorDir[0] = reset_motor.motorDir[0] * (-1)
            reset_motor.close_torque(0.3)
            motor_object.motorDir[0] = reset_motor.motorDir[0] * (-1)
        else:
            reset_motor.close_torque(0.3)
        finito = user_input2()


test_pos(reset_motor)
