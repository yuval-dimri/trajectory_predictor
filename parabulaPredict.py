import cv2
import numpy as np
import time
from collections import deque
import numpy
from numpy.polynomial import polynomial as P


# green
lowerBound = np.array([40, 50, 30])
upperBound = np.array([90, 255, 255])
# red
# lowerBound = np.array([0, 70, 60])
# upperBound = np.array([4, 240, 200])

ballPts = deque(maxlen=64)  # remember maximum 50 points
last_millis_time = 0
xyPosAndTime = []  # [[x, y, t],[x, y, t]]

yPos, xPos = -1, -1
annotationsCanvas = np.zeros((720, 1280, 3), np.uint8)


cam = cv2.VideoCapture("ball_throw_3.mp4")
# out = cv2.VideoWriter('output.mp4', -1, 8.0, (1920, 1080))
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))


def openCvShit(deisplayThings):
    global img
    ret, img = cam.read()
    img = cv2.resize(img, (1280, 720))

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


last_millis_time = 0
xPosLast = 0
yPosLast = 0


def updateBallValues():
    global xSpeedMean, xSpeedSum, ySpeedSum, xAcclSum, yAcclSum, ySpeedMean, xAcclMean, yAcclMean, xyPosAndTime
    xSpeedSum, ySpeedSum, xAcclSum, yAcclSum = 0, 0, 0, 0
    changeInXPos = xPos - xPosLast
    changeInYPos = yPos - yPosLast
    #### set time and pos array ####
    if(xPos != -1 or yPos != -1):
        if(len(xyPosAndTime) > 2):  # sum time is above the max #
            xyPosAndTime[-1] = [changeInXPos, changeInYPos, changeInTime]  # replace pos and time at the list's end
        else:
            xyPosAndTime.append([changeInXPos, changeInYPos, changeInTime])  # add pos and time to the list end
    # print(xyPosAndTime)
    if(xPos != -1 or yPos != -1):
        xyPosAndTime = shift_right(xyPosAndTime)
        for i in range(len(xyPosAndTime)):
            # print(xyPosAndTime)
            if(xyPosAndTime[i][2] != 0):
                xSpeedSum += xyPosAndTime[i][0] / xyPosAndTime[i][2]
                ySpeedSum += xyPosAndTime[i][1] / xyPosAndTime[i][2]
                xAcclSum += xyPosAndTime[i][0] / (xyPosAndTime[i][2]**2)
                yAcclSum += xyPosAndTime[i][1] / (xyPosAndTime[i][2]**2)
    # sumTime = sumInnerIndex(xyPosAndTime, 2)
    # xSpeedSum = sumInnerIndex(xyPosAndTime, 0) / sumTime
    # ySpeedSum = sumInnerIndex(xyPosAndTime, 1) / sumTime

        xSpeedMean = (xSpeedSum / len(xyPosAndTime))  # calculate speed mean
        ySpeedMean = (ySpeedSum / len(xyPosAndTime))  # calculate speed mean
        xAcclMean = xAcclSum / len(xyPosAndTime)
        yAcclMean = yAcclSum / len(xyPosAndTime)
    if(xPos != -1 or yPos != -1):
        cv2.line(img, (xPos, yPos), (xPos+round(xSpeedMean), yPos+round(ySpeedMean)), (0, 255, 0), 5)
        cv2.line(img, (xPos, yPos), (xPos+round(xAcclMean), yPos+round(yAcclMean)), (0, 255, 255), 5)


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


def displayParabola(image, parabola, xStart, xEnd, color):
    for x in range(xStart, xEnd):
        y = round(P.polyval(x, parabola))
        x = round(x)
        cv2.circle(image, (x, y), 3, color, -1)


firstPoint = None
secondPoint = None
thirdPoint = None
goodParabulaDrawn = False
ballPointsX = np.array([])
ballPointsY = np.array([])
captureBallXY = False
maxParabolaOffset = 20
lastParabolaC = None
displayBadParabolas = False
startCapturingBallXY = None
guessPointX = 900
guessPointY = 0
guessPointTime = -1
guessPointTimeCalculated = None

while True:
    current_seconds_time = time.time()
    # time.sleep(0.1)
    changeInTime = (current_seconds_time - last_millis_time)
    openCvShit(False)  # do all opencv shit without verbose display

    key = cv2.waitKey(10)
    if(key == ord('s') or xPos > 100):
        startCapturingBallXY = True
        print("capturing")
    if(xPos != -1 or yPos != -1):
        updateBallValues()
        print("x speed: {}".format(xSpeedMean))
        if(startCapturingBallXY is not None):
            ballPointsX = np.append(ballPointsX, xPos)  # make history from ball points
            ballPointsY = np.append(ballPointsY, yPos)  # make history from ball points
            if(len(ballPointsX) > 2):
                parabolaC = P.polyfit(ballPointsX, ballPointsY, 2)
                if(lastParabolaC is not None):
                    diffFromLastParabola = abs(P.polyval(1920, parabolaC) - P.polyval(1920, lastParabolaC))
                    if(diffFromLastParabola < maxParabolaOffset and not goodParabulaDrawn):
                        goodParabulaDrawn = True
                        guessPointY = P.polyval(guessPointX, parabolaC)
                        guessPointTime = abs(guessPointX - xPos)/xSpeedMean  # s/v = t
                        guessPointTimeCalculated = current_seconds_time
                        displayParabola(annotationsCanvas, parabolaC, xPos, 1920, (255, 0, 255))
                    elif(displayBadParabolas):
                        displayParabola(annotationsCanvas, parabolaC, xPos, 1920, (0, 0, 255))

                lastParabolaC = parabolaC
        print("guessPointTime = {}".format(guessPointTime))
        if(guessPointTimeCalculated is not None and current_seconds_time - guessPointTimeCalculated >= guessPointTime):
            cv2.circle(annotationsCanvas, (round(guessPointX), round(guessPointY)), 30, (255, 255, 0), -1)

        xPosLast = xPos
        yPosLast = yPos
    img = cv2.addWeighted(img, 1, annotationsCanvas, 1.0, 5.0)
    last_millis_time = current_seconds_time

    # out.write(img)
    cv2.imshow("main", img)
    cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
# out.release()
cv2.destroyAllWindows()
