import cv2
import numpy as np
import time
from collections import deque
import numpy
from numpy.polynomial import polynomial as P

lowerBound = np.array([40, 50, 30])
upperBound = np.array([90, 255, 255])
ballPts = deque(maxlen=64)  # remember maximum 50 points
yPos, xPos = -1, -1
annotationsCanvas = np.zeros((1080, 1920, 3), np.uint8)


cam = cv2.VideoCapture("ball_throw_6.mp4")
out = cv2.VideoWriter('output.mp4', -1, 8.0, (1920, 1080))
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

last_millis_time = 0
xPosLast = 0
yPosLast = 0


def openCvShit(deisplayThings):
    global img
    ret, img = cam.read()
    img = cv2.resize(img, (1920, 1080))

    # convert BGR to HSV
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    imgHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # create the Mask
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    # morphology
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    maskFinal = maskClose
    conts, h = cv2.findContours(
        maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    biggestArea = 0
    ballContsIndex = 0
    thereIsBall = 0
    for i in range(len(conts)):
        area = cv2.contourArea(conts[i])
        if(area >= biggestArea and area > 20):
            thereIsBall = 1
            biggestArea = area
            ballContsIndex = i
    global xPos, yPos

    if(len(conts) and thereIsBall):
        M = cv2.moments(conts[ballContsIndex])
        xPos = round(M['m10'] / M['m00'])
        yPos = round(M['m01'] / M['m00'])
        cv2.circle(img, (xPos, yPos), 5, (0, 255, 0), -1)
        ballPts.appendleft([xPos, yPos])
    # display line after ball

    for i in range(1, len(ballPts)):
        # if either of the tracked points are None, ignore
        # them
        if ballPts[i - 1] is None or ballPts[i] is None or ballPts[i - 1] == -1 or ballPts[i] == -1:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(img, (ballPts[i-1][0], ballPts[i-1][1]), (ballPts[i][0], ballPts[i][1]), (0, 160, 0), thickness)

    # cv2.cv.PutText(cv2.cv.fromarray(img), str(
    # i+1), (x, y+h), font, (0, 255, 255))
    if(deisplayThings):
        x, y, w, h = cv2.boundingRect(conts[ballContsIndex])
        cv2.drawContours(img, conts, -1, (255, 0, 0), 3)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("maskClose", maskClose)
        cv2.imshow("maskOpen", maskOpen)
        cv2.imshow("mask", mask)


def shift_right(lst):
    try:
        return [lst[-1]] + lst[:-1]
    except IndexError:
        return lst


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    return A, B, C


def parabola(a, b, c, x):
    return a+b*x+c*x**2


firstPoint = None
secondPoint = None
thirdPoint = None
parabulaDrawn = False
ballPointsX = np.array([])
ballPointsY = np.array([])
captureBallXY = False
while True:
    openCvShit(False)  # do all opencv shit without verbose display
    # key = cv2.waitKey(10)
    # if(key == ord('s')):
    #     startCapturingBallXY = True

    if(xPos != -1 or yPos != -1):
        if(xPos > 200):
            ballPointsX = np.append(ballPointsX, xPos)
            ballPointsY = np.append(ballPointsY, yPos)
            # print(fit)
            # cv2.circle(annotationsCanvas, (round(firstPoint[0]), round(firstPoint[1])), 10, (0, 0, 255), -1)
            # cv2.circle(annotationsCanvas, (round(secondPoint[0]), round(secondPoint[1])), 10, (0, 0, 255), -1)
            # cv2.circle(annotationsCanvas, (round(thirdPoint[0]), round(thirdPoint[1])), 10, (0, 0, 255), -1)
            print("got three points!!!!")
            # a, b, c = calc_parabola_vertex(firstPoint[0], firstPoint[1], secondPoint[0], secondPoint[1], thirdPoint[0], thirdPoint[1])
            # print("a = {0}, b = {1}, c = {2}".format(a, b, c))
            if(len(ballPointsX) > 6):
                fit = P.polyfit(ballPointsX, ballPointsY, 2)
                parabulaDrawn = True
                for x in range(xPos, 1920):
                    y = parabola(fit[0], fit[1], fit[2], x)
                    # y = parabola(a, b, c, x)
                    # print("y = {}".format(y))
                    cv2.circle(annotationsCanvas, (round(x), round(y)), 10, (0, 0, 255), -1)

    img = cv2.addWeighted(img, 1, annotationsCanvas, 1, 0.0)
    out.write(img)
    cv2.imshow("main", img)
    cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
out.release()
cv2.destroyAllWindows()
