# from hands import *
# from src.openhand_node import hands
#
# reset_servo_ids = [1]
# reset_servo_ids2 = [2]
# reset_servo_type = 'X_series'
# reset_servo_port = "COM4"
# reset_motor_offset = [0.0]
# # reset_motor = hands.Reset_Motor(reset_servo_port,  reset_servo_ids, 'XM')
# reset_motor = hands.Reset_Motor(reset_servo_port,  reset_servo_ids2, 'XM')
# def user_input1():
#     """ open or close """
#     ans = input('Continue?
#     else:from openhand_node 
#         return TrueFalseFalse
# : o/c ')
#     if ans == 'o':
#         return False
#     else:from openhand_node 
#         return True
#
# def user_input2():
#     """ Check to see if user wants to continue """
#     ans = input('Continue? : y/n ')
#     if ans == 'n':
#         return False
#     else:
#         return True
#
# def test_pos(motor_object):
#     finito = True
#     while finito:
#         bool_test = user_input1()
#         if bool_test:
#             reset_motor.close_torque(0.5)
#         else:
#             reset_motor.close_torque(-0.5)
#         finito = user_input2()
#     # go back to initial position
#     reset_motor.reset()
#
# test_pos(reset_motor)


from hands import *
import time

# from src.openhand_node import hands
import hands

reset_servo_ids = [1]
reset_servo_ids2 = [2]
reset_servo_type = 'X_series'
reset_servo_port = "/dev/ttyUSB0"
reset_motor_offset = [0.0]
yarden = hands.bio_proj(reset_servo_port, 1, 2, 'XM')
yarden.alon_incoder(1, 0)
yarden.alon_incoder(0, 0)

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
            yarden.close_melon(1, -0.5)
            # yarden.close_torque(0.5)
            # yarden.diagnostics()
            # yarden.moveMotor(1, 0.5)
        else:
            # yarden.close_melon(1, 0.5)
            i=0
            yarden.alon_incoder(1, 3000)
            time.sleep(0.2)
            while yarden.servos[1].is_moving():
                i = i+1
            yarden.alon_incoder(0, 3000)
            time.sleep(0.2)
            while yarden.servos[0].is_moving():
                i = i+1
            yarden.alon_incoder(1, 0)
            time.sleep(0.2)
            while yarden.servos[1].is_moving():
                i = i+1
            yarden.alon_incoder(0, 0)
            time.sleep(0.2)
            while yarden.servos[0].is_moving():
                i = i+1
            # yarden._close_torques_melon(1)
            # yarden._close_torques_melon(0)
            # yarden.close_torque(-0.5)
            # yarden.moveMotor(0, 0.8)
        finito = user_input2()


test_pos(yarden)
