#coding=utf-8
import cv2
import matplotlib.pyplot as plt
import os

detect_path = './file/data/haarcascades/haarcascade_frontalface_default.xml'

class normalFaceDetect:
    def __init__(self):
        detect_classification = cv2.CascadeClassifier(detect_path)
    def detect_faces(self,img):
        img_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = self.detect_classification.detectMultiScale(img_temp, scaleFactor=1.1, minNeighbors=4, minSize=(6, 6))

        for (x, y, w, h) in face:
            out_img = cv2.rectangle(img_temp, (x, y), (x + w, y + h), (0, 255, 0), 4)



if __name__ == '__main__':
