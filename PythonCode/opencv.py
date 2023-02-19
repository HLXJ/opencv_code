import cv2
import numpy as np
import matplotlib.pylab as plt


# 读取图像（opencv读取的图像格式为BGR）
img = cv2.imread('C:/Users/1/Pictures/Saved Pictures/picture1.jpg')
# img = cv2.resize(img, (3104, 2048))
# 窗口大小可以改变
# cv2.namedWindow('imge', cv2.WINDOW_NORMAL)


cv2.namedWindow('imge', cv2.WINDOW_FREERATIO)
cv2.resizeWindow("imge", 640, 480)

# 显示图像，创建窗口(下列代码需在本地,notbook不支持弹框显示)
cv2.imshow('imge', img)
# 等待时间，毫秒级，0表示任意键终止，其余数字表示时间
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 后续可定义展示单张图片的方法
# def cv_show(name, img):
#     cv2.imshow(name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# # 使用matplotlib显示图像(读取的图片为RGB格式)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式
#     plt.imshow(img_rgb)  # 根据数组绘制图像
#     plt.show()  # 显示图像





# # 视频读取(这个也需要本地运行)
# vc = cv2.VideoCapture('C:/Users/1/Videos/Captures/video1.mp4')
# # 检查是否打开正确
# if vc.isOpened():
#     open, frame = vc.read()
# else:
#     open = False
#
# while open:
#     ret, frame = vc.read()
#     if frame is None:
#         break
#     if ret == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 单帧图像处理：图像转为灰度图，当然也可以去掉或转为其它，根据情况而论
#         cv2.imshow('result', gray)
#         if cv2.waitKey(100) & 0xFF == 27:
#             break
# vc.release()
# cv2.destroyAllWindows()




