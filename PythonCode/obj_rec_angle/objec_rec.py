import cv2
import numpy as np
import math
import argparse



def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255)

def main(is_circle = True):
    cap = cv2.VideoCapture(0)
    cap.set(10,160)
    heightImg = 640
    widthImg  = 480

    def gradient(pt1, pt2):
        return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

    # initializeTrackbars()

    while True:
        ret,frame = cap.read() #ret是一个bool值，代表有没有读取到图片；frame表示截取到的一帧图片
        frame = cv2.resize(frame,(heightImg,widthImg))
        img = frame.copy()
        img1 = frame.copy()
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
        # thres = utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
        imgThreshold = cv2.Canny(imgBlur, 75, 200)

        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:-1] # False升序，True降序
        cv2.drawContours(frame, contours, 0, (0, 255, 0), 10) #画出最大的轮廓
        if len(contours)>=1:
            position = np.squeeze(contours[0])#删除维度为一的维
            is_pass = True
        else:
            is_pass = False

        if is_circle:
            if is_pass:
                (x,y),radius = cv2.minEnclosingCircle(np.array(position))
                x,y,radius = int(x),int(y),int(radius)
                img = cv2.circle(img, (x, y), 10, (255, 255, 0), -1)
                frame_circle = cv2.circle(img,(x,y),radius,(0,255,255),5)
                print(x,y)
                cv2.putText(frame_circle, "Circle-Detection", (20, 25),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 255), 2)
                cv2.putText(frame_circle,"position: "+str(x)+", "+str(y),(20,60),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)
                # cv2.imshow("frame2", frame_circle)
        else:
            rect = cv2.minAreaRect(np.array(position))
            center_x,center_y = int(rect[0][0]),int(rect[0][1])
            # angle = int(rect[2])
            box = cv2.boxPoints(rect) #获取矩形四个顶点坐标，浮点型
            box = np.int0(box) #将四个角点转换为整数类型
            # 获取四个顶点坐标
            left_point_x = np.min(box[:, 0])#取二维数组中所有第一维数据的最小值
            right_point_x = np.max(box[:, 0])
            top_point_y = np.min(box[:, 1])
            bottom_point_y = np.max(box[:, 1])

            left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
            right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
            top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
            bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
            # 上下左右四个点坐标
            vertices = np.array([[top_point_x, top_point_y],  [left_point_x, left_point_y],[bottom_point_x, bottom_point_y],
                                 [right_point_x, right_point_y]])
            pt2, pt1, pt3 = vertices[-3:]
            l1 = (top_point_x - left_point_x) ** 2 + (top_point_y - left_point_y) ** 2
            l2 = (bottom_point_x - left_point_x) ** 2 + (bottom_point_y - left_point_y) ** 2
            if l1 > l2:
                angle = gradient(pt1, pt2)
            else:
                angle = gradient(pt1, pt3)
            angle = math.atan(angle)
            if angle == angle:  #判断是否为空值
                angle = round(math.degrees(angle))
            else:
                angle = 0


            vertices = vertices.reshape(-1,1,2)
            cv2.polylines(img,[vertices],isClosed=True,color=(255,255,0),thickness=5)
            cv2.putText(img, "Rect-Detection" , (20, 25),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 255), 2)
            cv2.putText(img, "position: " + str(center_x) + ", " + str(center_y), (20, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (0, 255, 255), 2)
            cv2.putText(img, "angle: " + str(angle), (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (0, 255, 255), 2)


        result = np.concatenate((frame,img),axis=1)

        cv2.imshow("result", result)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--iscircle", help="true:circle,false:rectangle",
                        type=bool,default=False)
    args = parser.parse_args()

    main(is_circle = False)
