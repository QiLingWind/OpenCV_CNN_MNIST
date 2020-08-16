'''
声明：*2020.6.23*实时手写数字识别V1版，本程序实现了单个分开的数字识别，对于两个数字离得太近，在裁剪时会
将一部分裁进去，未作处理，因此效果不好。。
'''

import cv2
import numpy as np
import tensorflow as tf


# #####设置参数#######################
widthImg = 640
heightImg = 480
kernal = np.ones((5, 5))
minArea = 800
# ##################################
cap = cv2.VideoCapture(1)
cap.set(3, widthImg)  # 设置参数，10为亮度
cap.set(4, heightImg)
cap.set(10, 150)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# 预处理函数
def preProccessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    imgDial = cv2.dilate(imgCanny, kernal, iterations=2)
    imgThres = cv2.erode(imgDial, kernal, iterations=1)
    return imgThres


def getContours(img):
    x, y, w, h, xx, yy, ss = 0, 0, 10, 10, 20, 20, 10         # 因为图像大小不能为0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 检索外部轮廓
    for cnt in contours:  # 每一个轮廓线
        area = cv2.contourArea(cnt)
        # print(area)
        if area > minArea:  # 面积大于800像素为封闭图形
            cv2.drawContours(imgCopy, cnt, -1, (255, 0, 0), 3)  # 不要在原图上面画，-1是所有的轮廓
            peri = cv2.arcLength(cnt, True)  # 计算周长
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 计算有多少个拐角
            x, y, w, h = cv2.boundingRect(approx)  # 得到外接矩形的大小
            a = (w+h)//2
            dd = abs((w-h)//2)      # 边框的差值
            # cv2.rectangle(imgContour,(x, y),(x+w,y+h),(255,255,255),2)    # 白色
            if w <= h:  # 得到一个正方形框，边界往外扩充10像素
                xx = x-dd-10
                yy = y-10
                ss = h+20
                cv2.rectangle(imgCopy, (x-dd-10, y-10), (x+a+10, y+h+10), (0, 0, 255), 2)    # 看看框选的效果，在imgCopy中
                print(a+dd, h)
            else:               # 边界往外扩充10像素值
                xx = x-10
                yy = y-dd-10
                ss = w+20
                cv2.rectangle(imgCopy, (x-10, y-dd-10), (x+w+10, y+a+10), (0, 0, 255), 2)
                print(a+dd, w)
    if x != 0 or y != 0:        # 图像不能为0
        return xx, yy, ss
    else:
        return 20, 20, 10



# 重载模型
Saved_model = tf.keras.models.load_model('./my_models/MNIST_CNN_model.h5')
image_size = 28

while True:
    success, img = cap.read()
    imgCopy = img.copy()
    imgCopy = imgCopy[100:400, 100:640]
    imgProcess = preProccessing(img)  # 可以很好地把图像取出来
    imgProcess = imgProcess[100:400, 100:640]  # 300*540 去掉水印
    x, y, s = getContours(imgProcess)
    if x >= 10 and y >= 10 and s > 10:
        imgRes = imgProcess[y-5:y+s+10, x-5:x+s+10]     # 得到数字区域图片,注意先是y，再是x
        imgVert = np.expand_dims(imgRes, axis=2).repeat(1, axis=2)  # 前面得到的图片是二维的，这里给他加上第三维
        # print(x,y,s)
        decode_img = tf.image.convert_image_dtype(imgVert, tf.float32)  # 转换为float32格式
        test_img = tf.image.resize(decode_img, [image_size, image_size])
        test_img = tf.reshape(test_img, [-1, image_size, image_size, 1])
        test_img = tf.keras.utils.normalize(test_img, axis=1)  # 归一化
        test_img = test_img.numpy()  # 需要转换为numpy类型

        predictions = Saved_model.predict(test_img)
        # print(predictions)
        print("预测结果为：{}".format(np.argmax(predictions[0])))


        # 显示
        stackImg = stackImages(0.6, ([img, imgCopy], [imgProcess, imgRes]))
        cv2.imshow("Video", stackImg)
    else:
        print("区域内无数字")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break