# coding: utf-8
import cv2
import os
import numpy as np
from mtcnn import MTCNN
import face_alignment
import os
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

    def detect_face(self, image):
        '''
        :param image:使用cv2读取的图片
        :return: 使用mtcnn检测得到的数据
        '''
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_None(image):
            exit(0)
        results = self.detector.detect_faces(img)
        boxes = []
        faceKeyPoint = []
        if results is not None:
            # box框
            boxes = results[0]['box']
            # 人脸5个关键点
            points = results[0]['keypoints']
            for i in results[0]['keypoints']:
                faceKeyPoint = []
                for p in points:
                    for i in range(5):
                        faceKeyPoint.append([p[i], p[i + 5]])
        return {"boxes": boxes, "face_key_point": faceKeyPoint}


if __name__ == '__main__':

    pic = './img/ivan.jpg'
    img = cv2.imread(pic)
    detect = Detect()
    result = detect.detect_face(img)
    print(result)
    pic = cv2.imread(pic)
    align_img_list = face_alignment.align_face(opic_array=pic, faceKeyPoint=result['face_key_point'])
    for i in align_img_list:
        cv2.imshow('Pic_Frame', i)
        cv2.waitKey(0)


# box框
#             boxes = results[0]
#             人脸5个关键点
#             points = results[1]
#
#         [{'box': [277, 93, 49, 62], 'confidence': 0.9999746680259705,
#           'keypoints': {'left_eye': (291, 117), 'right_eye': (314, 115), 'nose': (304, 130), 'mouth_left': (296, 143),
#                         'mouth_right': (313, 142)}}, {'box': [307, 173, 37, 55], 'confidence': 0.8657231330871582,
#                                                       'keypoints': {'left_eye': (327, 194), 'right_eye': (339, 191),
#                                                                     'nose': (341, 199), 'mouth_left': (334, 215),
#                                                                     'mouth_right': (342, 213)}}]
#         results[0]
#         {'box': [277, 93, 49, 62], 'confidence': 0.9999746680259705,
#          'keypoints': {'left_eye': (291, 117), 'right_eye': (314, 115), 'nose': (304, 130), 'mouth_left': (296, 143),
#                        'mouth_right': (313, 142)}}
#         results[1]
#         {'box': [307, 173, 37, 55], 'confidence': 0.8657231330871582,
#          'keypoints': {'left_eye': (327, 194), 'right_eye': (339, 191), 'nose': (341, 199), 'mouth_left': (334, 215),
#                        'mouth_right': (342, 213)}}
#
#             for i in results[0]:
#                 faceKeyPoint = []
#                 for p in points:
#                     for i in range(5):
#                         faceKeyPoint.append([p[i], p[i + 5]])
#         return {"boxes": boxes, "face_key_point": faceKeyPoint}