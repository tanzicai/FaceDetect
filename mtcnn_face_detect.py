# coding: utf-8
import cv2
import os
import numpy as np
from mtcnn import MTCNN
import face_alignment
import os
import threading
import time
import math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def is_None(image):
    if image is None:
        return True
        exit(0)
    else:
        return False


class Detect:
    def __init__(self):
        self.detector  = MTCNN()
        self.results = None
    def draw_img(self,pic):
        for i in self.results:
            if i['confidence'] >0.8:
                print()
                x, y, w, h = i['box']
                color = (0, 0, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(i['confidence'])
                cv2.rectangle(pic, (x, y), (x + w, y + h), color, 1)
                cv2.putText(pic, text[0:4], (x, y - 10), font, 0.5, color, 1)
        cv2.imshow('Pic_Frame', pic)
        cv2.waitKey(0)

    def detect_multiple_face(self, image):
        '''
        :param image:使用cv2读取的图片
        :return: 使用mtcnn检测得到的数据
        '''
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_None(image):
            exit(0)
        results = self.detector.detect_faces(img)
        self.results = results
        return results

    def caculate_angle(self,x1, y1, x2, y2):
        """ 已知两点坐标计算角度 -
        :param x1: 原点横坐标值
        :param y1: 原点纵坐标值
        :param x2: 目标点横坐标值
        :param y2: 目标纵坐标值
        """
        angle = 0.0
        dx = x2 - x1
        dy = y2 - y1
        if x2 == x1:
            angle = math.pi / 2.0
            if y2 == y1:
                angle = 0.0
            elif y2 < y1:
                angle = 3.0 * math.pi / 2.0
        elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
        elif x2 > x1 and y2 < y1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif x2 < x1 and y2 < y1:
            angle = math.pi + math.atan(dx / dy)
        elif x2 < x1 and y2 > y1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        return angle * 180 / math.pi - 90

    def rotate_bound_white_bg(self,image, angle, center):
        """ 已知两点坐标计算角度 -
        :param image: 输入图片
        :param angle: 旋转角度
        :param center:中心点
        """
        (h, w) = image.shape[:2]
        (cX, cY) = center
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))


    def detect_simgle_face(self,image):
        '''
        :param image:使用cv2读取的图片
        :return: 使用mtcnn数据修正人脸裁剪后的图片
        '''
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_None(image):
            print("图片为空")
            exit(0)
        results = self.detector.detect_faces(img)
        if results.__len__() >= 2:
            print("人脸超出一个，请使用多个人脸检测")
            exit(0)
        detect_data = self.detector.detect_faces(img)
        (x1, y1) = detect_data[0]['keypoints']['left_eye']
        (x2, y2) = detect_data[0]['keypoints']['right_eye']
        center = detect_data[0]['keypoints']['nose']
        angle = self.caculate_angle(x1, y1, x2, y2)
        img2 = self.rotate_bound_white_bg(img, angle, center)
        detect_data_temp = self.detector.detect_faces(img2)
        box = detect_data_temp[0]['box']
        # img2 = cv2.rectangle(img2, (box[0], box[1]), (box[0]+box[2], box[1] + box[3]), (0, 255, 0), 4)
        img2 = img2[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        return img2


class Camera(threading.Thread):
    __slots__ = ['camera', 'Flag', 'count', 'width', 'heigth', 'frame']

    def __init__(self):
        threading.Thread.__init__(self)
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.Flag = 0
        self.count = 1
        self.width = 1920
        self.heigth = 1080
        self.name = ''
        self.path = ''
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.heigth)
        # for i in range(46):
        # print("No.={} parameter={}".format(i,self.camera.get(i)))

    def run(self):
        while True:
            ret, self.frame = self.camera.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
            self.frame = cv2.flip(self.frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
            if self.Flag == 1:
                print("拍照")
                if self.name == '' and self.path == '':
                    cv2.imwrite(str(self.count) + '.jpg', self.frame)  # 将画面写入到文件中生成一张图片
                elif self.name != '':
                    cv2.imwrite(self.name + '.jpg', self.frame)
                self.count += 1
                self.Flag = 0
            if self.Flag == 2:
                print("退出")
                self.camera.release()  # 释放内存空间
                cv2.destroyAllWindows()  # 删除窗口
                break

    def take_photo(self):
        self.Flag = 1

    def exit_program(self):
        self.Flag = 2

    def set_name(self, str):
        self.name = str

    def set_path(self, str):
        self.path = str

    def show_window(cap):
        while True:
            cv2.namedWindow("window", 1)  # 1代表外置摄像头
            cv2.resizeWindow("window", cap.width, cap.heigth)  # 指定显示窗口大小
            cv2.imshow('window', cap.frame)
            c = cv2.waitKey(50)  # 按ESC退出画面
            if c == 27:
                cv2.destroyAllWindows()
                break
    def show_face_detect_window(cap):
        detect = Detect()
        while True:
            cv2.namedWindow("window", 1)  # 1代表外置摄像头
            cv2.resizeWindow("window", cap.width, cap.heigth)  # 指定显示窗口大小
            result = detect.detect_face(cap.frame)
            detect.draw_img(cap.frame)
            # cv2.imshow('window', cap.frame)
            c = cv2.waitKey(50)  # 按ESC退出画面
            if c == 27:
                cv2.destroyAllWindows()
                break



if __name__ == '__main__':
    '''
    mtcnn测试(OK)
    '''
    # pic = './data/img/multiply_girls.jpg'
    # pic = './data/img/boys.jpg'
    # pic = './data/img/girls.jpg'
    # pic = './data/img/simple_boys.jpg'
    # img = cv2.imread(pic)
    # detect = Detect()
    # result = detect.detect_face(img)
    # detect.draw_img(img)
    '''
    摄像头拍照测试(OK)
    '''
    # cap = Camera()
    # cap.start()
    # while True:
    #     i = int(input("input:"))
    #     if i == 1:
    #         cap.take_photo()
    #     if i == 2:
    #         cap.exit_program()
    #     if i == 3:
    #         cap.show_window()
    #     time.sleep(1)
    '''
    摄像头人脸检查(Error)
    '''
    # cap =Camera()
    # cap.start()
    # cap.show_window()
    '''
    mtcnn单人矫正裁剪测试(OK)
    '''
    # pic = './data/raw_data/other/IMG_20221215_153252.jpg'
    # img = cv2.imread(pic)
    # detect = Detect()
    # result = detect.detect_simgle_face(img)
    # cv2.imshow('Pic_Frame', cv2.resize(result,(300,300)))
    # cv2.waitKey(0)
    # result = detect.detect_face(img)
    # detect.draw_img(img)
    '''
    faceNet测试部分
    '''