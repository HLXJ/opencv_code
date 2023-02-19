# coding=utf-8
from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from math import *
import math



def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv.getPerspectiveTransform(rect, dst)
	warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped,maxWidth,maxHeight



def resize(image, width=None, height=None, inter=cv.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv.resize(image, dim, interpolation=inter)
	return resized


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # mean, eigenvectors, eigenvalues = cv.PCACompute(data_pts, mean, 2) #image, mean=None, maxComponents=10
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv.circle(img, cntr, 3, (255, 0, 255), 2)  # 在PCA中心位置画一个圆圈
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)  # 绿色，较长轴
    drawAxis(img, cntr, p2, (255, 255, 0), 1)  # 黄色
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians #PCA第一维度的角度
    return angle


def rotate_img(img, angle):
    """中心旋转图像,输入的angle为弧度制"""
    angle_o = (angle - pi / 2) * 180 / pi  # 将弧度制转为角度制
    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]  # 原始图像宽度
    rotateMat = cv.getRotationMatrix2D((width / 2, height / 2), angle_o, 1)  # 按angle角度旋转图像
    heightNew = int(width * math.fabs(math.sin(angle)) + height * math.fabs(math.cos(angle)))
    widthNew = int(height * math.fabs(math.sin(angle)) + width * math.fabs(math.cos(angle)))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    # plt.figure(4);
    # plt.imshow(imgRotation, cmap='gray')

    return imgRotation

def transform_fun(orig_prarm, cnts_prarm):  #透视变换函数
    position = np.squeeze(cnts_prarm)  # 删除维度为一的维
    rect = cv.minAreaRect(np.array(position))
    center_x, center_y = int(rect[0][0]), int(rect[0][1])
    # angle = int(rect[2])
    box = cv.boxPoints(rect)  # 获取矩形四个顶点坐标，浮点型
    box = np.intp(box)  # 将四个角点转换为整数类型
    # 获取四个顶点坐标
    left_point_x = np.min(box[:, 0])  # 取二维数组中所有第一维数据的最小值
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    # 上下左右四个点坐标
    vertices = np.array([[top_point_x, top_point_y], [left_point_x, left_point_y], [bottom_point_x, bottom_point_y],
                         [right_point_x, right_point_y]])
    # vertices = np.array([[x+w, y+h], [x+w,y], [x, y],[x, y+h]])
    vertices = vertices.reshape(-1, 1, 2)  # -1 表示该维度自动计算
    # cv.polylines(orig, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)  # 画多边形
    # 透视变换
    warped, maxWidth, maxHeight = four_point_transform(orig_prarm, vertices.reshape(4, 2))
    # cv.imshow("warped", warped)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return  warped, maxWidth, maxHeight

def angle_correct(warped):
    img = warped
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title('1')
    # 二值处理
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    ref = cv.threshold(warped, 50, 255, cv.THRESH_BINARY_INV)[1]

    plt.subplot(222), plt.imshow(ref, 'gray'), plt.title('2')

    contours, _ = cv.findContours(ref, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # contours = sorted(contours, key=cv.contourArea, reverse=True)[1]
    areas = []
    areas_angle = []
    for i, con in enumerate(contours):
        # Calculate the area of each contour
        area = cv.contourArea(con)
        # Ignore contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            continue
        areas.append(area)
        # Draw each contour only for visualisation purposes
        # cv.drawContours(img, contours, i, (0, 0, 255), 3)
        # Find the orientation of each shape
        angle = getOrientation(con, img)
        areas_angle.append(angle)
    # cv.imshow('output', src)
    plt.subplot(223), plt.imshow(img, 'gray'), plt.title('3')
    # cv.waitKey()

    # 计算面积最大的连通域的方向
    ind = np.argmax(areas)
    imgRotation = rotate_img(img, areas_angle[ind])
    plt.subplot(224), plt.imshow(imgRotation, 'gray'), plt.title('4')
    plt.show()
    print("done")
    return imgRotation

def temp_match(tar,temp):
    # 获得模板图片的高宽尺寸
    theight, twidth = temp.shape[:2]
    # 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
    result = cv.matchTemplate(tar, temp, cv.TM_SQDIFF_NORMED)
    # 归一化处理
    cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
    # 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    # 匹配值转换为字符串
    # 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
    # 对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
    strmin_val = str(min_val)
    # 绘制矩形边框，将匹配区域标注出来
    # min_loc：矩形定点
    # (min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
    # (0,0,225)：矩形的边框颜色；2：矩形边框宽度
    cv.rectangle(tar, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)
    # 显示结果,并将匹配值显示在标题栏上
    cv.imshow("MatchResult----MatchingValue=" + strmin_val, tar)
    cv.waitKey()
    cv.destroyAllWindows()
    return strmin_val

parser = argparse.ArgumentParser(description='Code for Introduction to Principal Component Analysis (PCA) tutorial.\
                                              This program demonstrates how to use OpenCV PCA to extract the orientation of an object.')
parser.add_argument('--input', help='Path to input image.', default='3.jpg')
args = parser.parse_args()
src = cv.imread(args.input)
template = cv.imread('temp.jpg')
temp = angle_correct(template)
# Check if image is loaded successfully
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
src = cv.resize(src, (620, 480))
orig = src.copy()
# 预处理
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(gray, 75, 200)
# ref_gray = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)[1]

# 展示预处理结果
print("STEP 1: 边缘检测")
# cv.imshow("Image", src)
# cv.imshow("Edged", edged)
# cv.imshow("gray", gray)
# cv.imshow("ref", ref_gray)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 轮廓检测
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:-1] #True 为降序

# cv.drawContours(src, cnts, -1, (0, 255, 0), 3)
# cv.imshow("Image", src)
# cv.waitKey(0)
# cv.destroyAllWindows()


if len(cnts) >= 1:
    for c in cnts:
        area = cv.contourArea(c)
        print(area)
        warped, maxWidth, maxHeight = transform_fun(orig, c)
        target = angle_correct(warped)
        temp = cv.resize(temp, (maxWidth, maxHeight))
        strmin_val = temp_match(target,temp)
        print("strmin_val:"+strmin_val)


# cv.imshow("Outline", orig)
# cv.waitKey(0)
# cv.destroyAllWindows()

