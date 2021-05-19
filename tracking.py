import cv2
import numpy as np
import time
from collections import deque
import numpy


lowerBound = np.array([40, 50, 30])
upperBound = np.array([90, 255, 255])
ballPts = deque(maxlen=64)  # remember maximum 50 points
trajectoryMemory = deque(maxlen=30)
trajRail = deque(maxlen=1280)
yPos, xPos = -1, -1

annotationsCanvas = np.zeros((1080, 1920, 3), np.uint8)

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("ball_throw_3.mp4")
out = cv2.VideoWriter('output.mp4', -1, 5.0, (1920, 1080))
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

maxRememberTime = 0.1  # millis

last_millis_time = 0
xPosLast = 0
yPosLast = 0

# font = cv2.cv.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 0.5, 0, 3, 1)


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


xyPosAndTime = []
# [[x, y, t],[x, y, t]]


def sumInnerIndex(list, index):
    sum = 0
    for i in range(len(list)):
        sum += list[i][index]
    return sum


def updateBallValues():
    global xSpeedMean, xSpeedSum, ySpeedSum, xAcclSum, yAcclSum, ySpeedMean, xAcclMean, yAcclMean, xyPosAndTime
    xSpeedSum, ySpeedSum, xAcclSum, yAcclSum = 0, 0, 0, 0
    changeInXPos = xPos - xPosLast
    changeInYPos = yPos - yPosLast
    #### set time and pos array ####
    if(xPos != -1 or yPos != -1):
        if(len(xyPosAndTime) > 0):  # sum time is above the max #
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

        xSpeedMean = (xSpeedSum / len(xyPosAndTime))*0.006  # calculate speed mean
        ySpeedMean = (ySpeedSum / len(xyPosAndTime))*0.007  # calculate speed mean
        xAcclMean = xAcclSum / len(xyPosAndTime)*0.01
        yAcclMean = yAcclSum / len(xyPosAndTime)*0.01
    if(xPos != -1 or yPos != -1):
        cv2.line(img, (xPos, yPos), (xPos+round(xSpeedMean), yPos+round(ySpeedMean)), (0, 255, 0), 5)
        cv2.line(img, (xPos, yPos), (xPos+round(xAcclMean), yPos+round(yAcclMean)), (0, 255, 255), 5)


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A = 0.0013708119143971708  # (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    return A, B, C


def parabola(a, b, c, x):
    return ((a * x**2) + (b * x) + c)


t = 0
lastTimeCalculated = 0

# trajRail = [[],[],[]]
trajRail = list()
for i in range(1920):
    trajRail.append([0, 0])

recentRails = list()
for i in range(1920):
    recentRails.append(trajRail)

firstPoint = None
secondPoint = None
thirdPoint = None
parabulaDrawn = False
beenHere = False
startedTime = time.time()
while True:
    current_millis_time = time.time()
    changeInTime = (current_millis_time - last_millis_time)
    openCvShit(False)  # do all opencv shit without verbose display
    updateBallValues()

    # print("time = {}".format(changeInTime))

    if(xPos != -1 or yPos != -1):
        # v0 = 1
        # if key == ord('s'):
        #     v0 = ySpeedMean
        #     print("v0 = {}".format(v0))

        # print(ySpeedMean)
        # step = 1
        # if (xSpeedMean < 0):
        #     step = -1
        # yFormula = 0
        # key = cv2.waitKey(10)
        # if(key == ord('s') or beenHere):
        # beenHere = True
        print(np.sqrt(xSpeedMean**2+ySpeedMean**2))
        if(200 < xPos < 250):
            print("first point")
            firstPoint = [xPos, yPos]
        if(500 < xPos < 700):
            print("second point")
            secondPoint = [xPos, yPos]
        if(800 < xPos < 900):
            print("third point")
            thirdPoint = [xPos, yPos]
        if(firstPoint and secondPoint and thirdPoint):
            print("got three points!!!!")
            a, b, c = calc_parabola_vertex(firstPoint[0], firstPoint[1], secondPoint[0], secondPoint[1], thirdPoint[0], thirdPoint[1])
            print("a = {0}, b = {1}, c = {2}".format(a, b, c))
            if(not parabulaDrawn):
                parabulaDrawn = True
                for x in range(xPos, 1920):
                    y = parabola(a, b, c, x)
                    cv2.circle(annotationsCanvas, (round(x), round(y)), 10, (0, 0, 255), -1)
        lastTimeCalculated = time.time()
        # if(time.time() - lastTimeCalculated > 0.5):
        #     for i in range(0, 1920):
        #         xTraj = xPos + (i*xSpeedMean)
        #         yFormula = (ySpeedMean*i + 4.9*0.001*(i**2))
        #         yTraj = yPos + yFormula
        #         cv2.circle(annotationsCanvas, (round(xTraj),round(yTraj)), 10, (0, 0, 255), -1)
        # last_millis_time = current_millis_time
        xPosLast = xPos
        yPosLast = yPos

        # trajRail[i][0],trajRail[i][1] = xTraj,yTraj

        # trajectoryMemory[i].appendleft(trajRail)

    # for rail in trajectoryMemory:
    #     for xyTraj in rail:
    #         cv2.circle(img, (xyTraj[0], xyTraj[1]), 10, (0, 0, 255), -1)

    # t = xPos if t > 1280 else t+1
    # y = 0
    # t = current_millis_time - last_millis_time
    # y = y+v0*t - 4.9*(t**2)
    # print(xSpeedMean)

    # time.sleep(1/30)
    # out.write(img)
    img = cv2.addWeighted(img, 1, annotationsCanvas, 1, 0.0)
    cv2.imshow("main", img)
    # cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
out.release()
cv2.destroyAllWindows()
