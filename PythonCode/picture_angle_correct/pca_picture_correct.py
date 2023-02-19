# coding=utf-8
from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from math import *
import math

"""
代码基于opencv例程修改，增加了图像矫正功能；
需要opencv3.4.3以上版本
"""


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
    plt.figure(4);
    plt.imshow(imgRotation, cmap='gray')
    return imgRotation


parser = argparse.ArgumentParser(description='Code for Introduction to Principal Component Analysis (PCA) tutorial.\
                                              This program demonstrates how to use OpenCV PCA to extract the orientation of an object.')
parser.add_argument('--input', help='Path to input image.', default='3.jpg')
args = parser.parse_args()
src = cv.imread(args.input)
# Check if image is loaded successfully
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
src = cv.resize(src, (200, 200))
# cv.imshow('src', src)
plt.figure(1);
plt.imshow(src, cmap='gray')
# Convert image to grayscale
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# Convert image to binary
_, bw = cv.threshold(gray, 200, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
plt.figure(2);
plt.imshow(bw, cmap='gray')
contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
areas = []
areas_angle = []
for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv.contourArea(c)
    # Ignore contours that are too small or too large
    if area < 1e2 or 1e5 < area:
        continue
    areas.append(area)
    # Draw each contour only for visualisation purposes
    cv.drawContours(src, contours, i, (0, 0, 255), 5)
    # Find the orientation of each shape
    angle = getOrientation(c, src)
    areas_angle.append(angle)
# cv.imshow('output', src)
plt.figure(3);
plt.imshow(src, cmap='gray')
# cv.waitKey()

# 计算面积最大的连通域的方向
ind = np.argmax(areas)
imgRotation = rotate_img(src, areas_angle[ind])
plt.show()
print("done")


