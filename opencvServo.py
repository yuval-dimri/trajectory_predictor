import cv2
import numpy as np
import serial
# arduino = serial.Serial('COM5', 115200, timeout=.1)

calibrationPointsCounter = 0


def map(x,  in_min,  in_max,  out_min,  out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def moveServo(x, y):
    servoDegX = 180 - x*180/width
    servoDegY = 180 - y*180/height
    global servoX, servoY
    x = round(x)
    y = round(y)
    x = map(x, 0, 180,)
    servoX, servoY = x, y
    # print("x={}, y={}".format(x, y))
    print("moving servo to {}, {}".format(180 - x, 180-y))
    sendStr = "!d"+str(x)+","+str(y)+",\n"
    # arduino.write(sendStr.encode())


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    global calibrationPointsCounter
    global calibrationPoints
    # print(event)
    if(calibrationPointsCounter > 2):
        calibrationPointsCounter = 0
        calibrationPoints = [[-10, -10], [-10, -10]]
    if(event == cv2.EVENT_LBUTTONDOWN):
        if(calibrationPointsCounter <= 1):
            print(x, y)
            servoX, servoY = x, y
            moveServo(servoX, servoY)
            calibrationPoints[calibrationPointsCounter] = [x, y]
        calibrationPointsCounter += 1
    if event == 0:  # :
        cv2.circle(img, (x, y), 10, (255, 0, 0), -1)
        mouseX, mouseY = x, y


mouseX, mouseY = 0, 0

img = np.zeros((1000, 1000, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
height = img.shape[0]
width = img.shape[1]

calibrationPoints = [[-10, -10], [-10, -10]]
if(__name__ == '__main__'):
    global servoX, servoY
    servoY, servoX = 0, 0
    while(1):
        img = np.zeros((1000, 1000, 3), np.uint8)

        if(not any(-10 in xy for xy in calibrationPoints)):  # there is no -10 in any of the points
            cv2.rectangle(img, tuple(calibrationPoints[0]), tuple(calibrationPoints[1]), (255, 255, 0), 2)

        cv2.circle(img, tuple(calibrationPoints[0]), 10, (255, 255, 0))
        cv2.circle(img, tuple(calibrationPoints[1]), 10, (255, 255, 0))
        cv2.imshow('image', img)

        key = cv2.waitKey(20)
        if key == ord('q'):
            break
        elif key == 13:
            if(not any(-10 in xy for xy in calibrationPoints)):  # there is no -10 in any of the points
                calibrationPoints = [[-10, -10], [-10, -10]]
                servoCalibrationDegrees = [servoX, servoY]
                calibrationPointsCounter += 1
        elif key == ord('a'):
            servoX += 2
            moveServo(servoX, servoY)
        elif key == ord('d'):
            servoX -= 2
            moveServo(servoX, servoY)
        elif key == ord('w'):
            servoY += 2
            moveServo(servoX, servoY)
        elif key == ord('s'):
            servoY -= 2
            moveServo(servoX, servoY)
