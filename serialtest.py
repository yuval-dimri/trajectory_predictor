import serial
arduino = serial.Serial('COM5', 115200, timeout=.1)


def moveServo(x, y):
    arduino.write(b'!d180,180,\n')


while True:
    arduino.write(b'!d180,180,\n')
    data = arduino.readline()[:-2]  # the last bit gets rid of the new-line chars
    if(data):
        print(data)
