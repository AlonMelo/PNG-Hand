from hands import *
from src.openhand_node import hands


reset_servo_ids = [1]
reset_servo_type = 'X_series'
reset_servo_port = "COM4"
#reset_openhand_type: 'Reset_Motor'
#reset_direction: 1
reset_motor_offset = [0.0]
reset_motor = hands.Reset_Motor(reset_servo_port,  reset_servo_ids, 'XM')

reset_motor.close_torque(0.3)
time.sleep(4)
reset_motor.motorDir[0] = reset_motor.motorDir[0]*(-1)
reset_motor.release_()
