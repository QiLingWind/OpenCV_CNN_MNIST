'''
声明: 本程序实现了单个数字区域的框选
'''

import cv2
import numpy as np
######设置参数#######################
widthImg = 640
heightImg = 480
kernal = np.ones((5, 5))
minArea = 800
###################################
cap = cv2.VideoCapture(0)
cap.set(3, widthImg)      #设置参数，10为亮度
cap.set(4, heightImg)
cap.set(10,150)

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
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
    x, y, w, h, xx, yy, ss = 0, 0, 0, 0, 10, 10, 10
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    # 检索外部轮廓
    for cnt in contours:    # 每一个轮廓线
        area = cv2.contourArea(cnt)
        # print(area)
        if area > minArea:       # 面积大于5000像素为封闭图形
            cv2.drawContours(imgCopy, cnt, -1, (255, 0, 0), 3)     # 不要在原图上面画，-1是所有的轮廓
            peri = cv2.arcLength(cnt, True)      # 计算周长
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)     # 计算有多少个拐角
            x, y, w, h = cv2.boundingRect(approx)               # 得到外接矩形的大小
            a = (w+h)//2
            # d = abs((w-h)//2)
            # cv2.rectangle(imgContour,(x, y),(x+w,y+h),(255,255,255),2)
            if w <= h:                                          # 得到一个正方形框
                d = (h-w)//2
                xx = x-d-10
                yy = y-10
                ss = h+20
                cv2.rectangle(imgCopy, (x-d-10, y-10), (x+a+10, y+h+10), (0, 0, 255), 2)
                print(a+d, h)
            else:
                d = (w-h)//2
                xx = x-10
                yy = y-d-10
                ss = w+20
                cv2.rectangle(imgCopy, (x-10, y-d-10), (x+w+10, y+a+10), (0, 0, 255), 2)
                print(a+d, w)
    return xx, yy, ss





while True:
    success, img = cap.read()
    imgCopy = img.copy()
    imgCopy = imgCopy[100:400, 100:640]
    imgProcess = preProccessing(img)        # 可以很好地把图像取出来
    imgProcess = imgProcess[100:400, 100:640]   # 300*540 去掉水印
    imgContour = imgProcess     # 用于边缘检测
    x, y, s = getContours(imgProcess)
    if x>=10 and y>=10:
        imgRes = imgProcess[y-5:y+s+10, x-5:x+s+10]

        cv2.imshow("Video", imgRes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break