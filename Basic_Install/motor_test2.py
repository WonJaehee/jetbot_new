import time

from Adafruit_MotorHAT import Adafruit_MotorHAT
import time

# sets motor speed between [-1.0, 1.0]
def set_speed(motor_ID, value):
	max_pwm = 115.0
	speed = int(min(max(abs(value * max_pwm), 0), max_pwm))

	if motor_ID == 1:
		motor = motor_left
	elif motor_ID == 2:
		motor = motor_right
	else:
		return
	
	motor.setSpeed(speed)

	if value > 0:
		motor.run(Adafruit_MotorHAT.FORWARD)
	else:
		motor.run(Adafruit_MotorHAT.BACKWARD)


# stops all motors
def all_stop():
	motor_left.setSpeed(0)
	motor_right.setSpeed(0)

	motor_left.run(Adafruit_MotorHAT.RELEASE)
	motor_right.run(Adafruit_MotorHAT.RELEASE)




# setup motor controller
motor_driver = Adafruit_MotorHAT(i2c_bus=1)

motor_left_ID = 1
motor_right_ID = 2

motor_left = motor_driver.getMotor(motor_left_ID)
motor_right = motor_driver.getMotor(motor_right_ID)

set_speed(motor_left_ID,   0.5)
set_speed(motor_right_ID,  0.5)

time.sleep(1.0)

try:
    while 1:
        set_speed(motor_left_ID,   0.5)
        set_speed(motor_right_ID,  0.5)

        time.sleep(1.0)

except KeyboardInterrupt:  
    print("key int")
    all_stop()

# stop the motors as precaution
all_stop()

