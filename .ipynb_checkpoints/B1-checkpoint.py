首先，需要导入相关的库：
import cv2 #用于图像处理
import numpy as np #用于矩阵运算
import matplotlib.pyplot as plt #用于图像展示

然后，读取附件1中的三张甲骨文图像：（自己改一下路径）
img1 = cv2.imread('Pre_test/w01906.jpg')
img2 = cv2.imread('Pre_test/w01907.jpg')
img3 = cv2.imread('Pre_test/w01908.jpg')

接着，对读取的图像进行灰度处理：
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

然后，使用高斯滤波对图像进行平滑处理，以去除噪声：
blur1 = cv2.GaussianBlur(gray1, (3,3), 0)
blur2 = cv2.GaussianBlur(gray2, (3,3), 0)
blur3 = cv2.GaussianBlur(gray3, (3,3), 0)

接着，使用Canny边缘检测算法对图像进行边缘检测，以提取图像的边缘特征：
edges1 = cv2.Canny(blur1, 100, 200)
edges2 = cv2.Canny(blur2, 100, 200)
edges3 = cv2.Canny(blur3, 100, 200)

然后，对边缘检测结果进行膨胀操作，以填充边缘间的空隙：
kernel = np.ones((3,3),np.uint8)
dilation1 = cv2.dilate(edges1,kernel,iterations = 1)
dilation2 = cv2.dilate(edges2,kernel,iterations = 1)
dilation3 = cv2.dilate(edges3,kernel,iterations = 1)

最后，将处理后的图像和原始图像进行对比展示，以观察图像预处理效果：
plt.subplot(231),plt.imshow(img1),plt.title('Original')
plt.subplot(232),plt.imshow(dilation1),plt.title('Processed')
plt.subplot(233),plt.imshow(img2),plt.title('Original')
plt.subplot(234),plt.imshow(dilation2),plt.title('Processed')
plt.subplot(235),plt.imshow(img3),plt.title('Original')
plt.subplot(236),plt.imshow(dilation3),plt.title('Processed')
plt.show()

