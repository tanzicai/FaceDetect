#coding=utf-8
import cv2
# import matplotlib.pyplot as plt
import os

detect_path = './model/haarcascade_frontalface_default.xml'
raw_data_path = './data/raw_data/other/'
target_data_path = './data/raw_data/target/'
output_path = './data/cropped_data/other/'

def show_image_with_opencv(img,name):
    cv2.namedWindow(name, 0)
    # cv2.resizeWindow(name, height,width)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class normalFaceDetect:
    def __init__(self):
        self.detect_classification = cv2.CascadeClassifier(detect_path)
        self.img = None
        self.face = None


    def detect_faces(self,img):
        img_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.face = self.detect_classification.detectMultiScale(img_temp, scaleFactor=1.1, minNeighbors=4, minSize=(6, 6))
        (x, y, w, h) = self.face[0]
        out_img = cv2.rectangle(img_temp, (x, y), (x + w, y + h), (0, 255, 0), 4)
        self.img = out_img
        return self.img

    def face_cropping(self):
        for (x, y, w, h) in self.face:
             img2 = self.img[y:y+h,x:x+w]
        img2 = cv2.resize(img2,(200,200))
        return img2


    def histogram_comparison(self,img1,img2):
        H1 = cv2.calcHist(img1,[0],None,[256],[0,255],False)
        H2 = cv2.calcHist(img2,[0],None,[256],[0,255],False)
        hist_compare_num = cv2.compareHist(H1,H2,method=cv2.HISTCMP_BHATTACHARYYA)
        return hist_compare_num


class preCropping:
    def __int__(self):
        self.similar_path = None

    def pre_do(self,path,output_path='./data/cropped_data/other/'):
        path_list = os.listdir(path)
        for i in range(len(path_list)):
            print("------------------------------------------")
            print(path_list[i] + "--------"+str(i)+"处理完成")
            img = cv2.imread(path + path_list[i])
            detect = normalFaceDetect()
            if img is not None:
                detect.detect_faces(img)
                new_img = detect.face_cropping()
                cv2.imwrite(output_path + str(i) + '.jpg', new_img, [cv2.IMWRITE_JPEG_QUALITY, 80])

    def choose_similar(self,num):
        similar = []
        similar_path = []
        # test the hist compare function
        target = cv2.imread('./data/cropped_data/target/0.jpg')
        for i in os.listdir('./data/cropped_data/other/'):
            test = cv2.imread('./data/cropped_data/other/' + i)
            calchist = detect.histogram_comparison(target, test)
            similar.append(calchist)
            similar.sort()
        for i in os.listdir('./data/cropped_data/other/'):
            test = cv2.imread('./data/cropped_data/other/' + i)
            calchist = detect.histogram_comparison(target, test)
            if calchist < similar[num]:
                similar_path.append(i)
        self.similar_path = similar_path
        return  similar_path


if __name__ == '__main__':
    img = cv2.imread('data/raw_data/other/IMG_20221215_153251.jpg')
    detect = normalFaceDetect()
    detect.detect_faces(img)
    new_img = detect.face_cropping()
    show_image_with_opencv(new_img,'测试')
    # preCropping().pre_do(raw_data_path)
    # preCropping().pre_do(target_data_path,'./data/cropped_data/target/')
    # similar = []
    # similar_path = []
    # # test the hist compare function
    # target = cv2.imread('./data/cropped_data/other/3.jpg')
    # for i in os.listdir('./data/cropped_data/other/'):
    #     test = cv2.imread('./data/cropped_data/other/' + i)
    #     calchist = detect.histogram_comparison(target, test)
    #     similar.append(calchist)
    #     similar.sort()
    # for i in os.listdir('./data/cropped_data/other/'):
    #     test = cv2.imread('./data/cropped_data/other/' + i)
    #     calchist = detect.histogram_comparison(target, test)
    #     if calchist < similar[9]:
    #         similar_path.append(i)
    # for i in similar_path:
    #     print(i)


    '''----------------------------------------
    主要流程
    -------------------------------------------'''
    # detect = normalFaceDetect()
    # pre_cropping = preCropping()
    # # pre_cropping.pre_do(raw_data_path)
    # # preCropping().pre_do(target_data_path,'./data/cropped_data/target/')
    # print(pre_cropping.choose_similar(10))
    # # detect.detect_faces(img)
    # # new_img = detect.face_cropping()
    #
