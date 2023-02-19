import cv2
import os

path = r"D:\PythonCode\GetPictureAngle\1"  # 存放原图片的文件夹路径
list = os.listdir(path)
for index, i in enumerate(list):
    path = r"D:\PythonCode\GetPictureAngle\1\{}".format(i)
    img = cv2.imread(path)
    img = cv2.resize(img, (591, 442))  # 修改为480*640
    path = r"D:\PythonCode\GetPictureAngle\1\{}.jpg".format(index) # 处理后的图片文件夹路径
    cv2.imwrite(path, img)
