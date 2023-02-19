# 导入工具包
import numpy as np
import argparse
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# 读取输入
image = cv2.imread(args["image"])
orig = image.copy()

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 75, 200)
edged_inv = cv2.bitwise_not(edged)
# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.imshow("Edged_Inv", edged_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('scan.jpg', edged)


# # 轮廓检测
# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
# cnts = sorted(cnts,key = cv2.contourArea,reverse=True)[0:-1]
# # 遍历轮廓
# for c in cnts:
# 	# 计算轮廓近似
# 	peri = cv2.arcLength(c, True)
# 	# C表示输入的点集
# 	# epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
# 	# True表示封闭的
# 	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
# 	# 4个点的时候就拿出来
# 	if len(approx) == 4:
# 		screenCnt = approx
# 		break
#
# # 展示结果
# print("STEP 2: 获取轮廓")
# cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


