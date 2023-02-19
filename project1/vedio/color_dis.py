import cv2 as cv
import numpy as np
lower_red=np.array([150,43,46])
upper_red=np.array([179,255,255])
lower_green=np.array([30,100,100])
upper_green=np.array([80,255,255])
lower_blue=np.array([100,100,100])
upper_blue=np.array([125,255,255])
lower_yellow=np.array([20,30,30])#ok
upper_yellow=np.array([70,255,255])
lower_purple=np.array([0,0,221])
upper_purple=np.array([160,30,255])
color_array=[[lower_red,upper_red],[lower_green,upper_green]]
capture1 = cv.VideoCapture(0)
font=cv.FONT_ITALIC


def process(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    line = cv.getStructuringElement(cv.MORPH_RECT, (15, 15), (-1, -1))#构造一个特定大小和形状的结构元素,形状，结构元素大小，(-1,-1)表示锚点在中心
    # for (lower,upper) in color_array:
    mask_red = cv.inRange(hsv, lower_red, upper_red)
    mask_red = cv.morphologyEx(mask_red, cv.MORPH_OPEN, line)
    contours_red, hierarchy = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    index = -1
    max = 0
    for c in range(len(contours_red)):
        area = cv.contourArea(contours_red[c])

        if area > max:
            max = area
            index = c

# 绘制
    if index >= 0:
        rect = cv.minAreaRect(contours_red[index])
        cv.ellipse(image, rect, (0, 255, 0), 2, 8)
        cv.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
        cv.putText(image, 'red', (50,150), cv.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 12)

    return image

while True:
    ret, frame = capture1.read()
    if ret is True:
        #cv.imshow("video-input", frame)
        result = process(frame)
        cv.imshow("result", result)
        c = cv.waitKey(50)
        print(c)
        if c == 27:  #ESC
            break
    else:
        break
cv.waitKey(0)
cv.destroyAllWindows()

