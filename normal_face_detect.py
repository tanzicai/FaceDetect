#coding=utf-8
import cv2
import  numpy as np
import os
from PIL import Image
import  matplotlib.pyplot as plt


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
    def __init__(self,imgs = []):
        self.detect_classification = cv2.CascadeClassifier(detect_path)
        self.img = None
        self.face = None
        self.imgs = imgs


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
        img2 = cv2.resize(img2,(32,32))
        return img2


    def histogram_comparison(self,img1,img2):
        H1 = cv2.calcHist(img1,[0],None,[256],[0,255],False)
        H2 = cv2.calcHist(img2,[0],None,[256],[0,255],False)
        hist_compare_num = cv2.compareHist(H1,H2,method=cv2.HISTCMP_BHATTACHARYYA)
        return hist_compare_num

    # def pca(self):
    #     """---------------------------------
    #     :return: pca结果数组
    #     ----------------------------------"""
    #     imageMatrix = []
    #     for i in self.imgs:
    #         mats = np.array(i)
    #         imageMatrix.append(mats.ravel())
    #     imageMatrix = np.array(imageMatrix)
    #     imageMatrix = np.transpose(imageMatrix)
    #     imageMatrix = np.mat(imageMatrix)
    #     # mean_img = np.mean(imageMatrix, axis=1)
    #     # mean_img1 = np.reshape(mean_img, (200,200))
    #     # im = Image.fromarray(np.uint8(mean_img1))
    #     # im.show()
    #     # # 均值中心化
    #     # imageMatrix = imageMatrix - mean_img
    #     # imag_mat = (imageMatrix.T * imageMatrix) / float(len(self.imgs))
    #     # W, V = np.linalg.eig(imag_mat)
    #     # V_img = imageMatrix * V
    #     # # 降序排序后的索引值
    #     # axis = W.argsort()[::-1]
    #     # V_img = V_img[:, axis]
    #     #
    #     # number = 0
    #     # x = sum(W)
    #     # for i in range(len(axis)):
    #     #     number += W[axis[i]]
    #     #     if float(number) / x > 0.9:  # 取累加有效值为0.9
    #     #         print('累加有效值是：', i)  # 前62个特征值保存大部分特征信息
    #     #         break
    #     # # 取前62个最大特征值对应的特征向量，组成映射矩阵
    #     # V_img_finall = V_img[:, :128]
    #     # # 降维后的训练样本空间
    #     # # projectedImage = V_img_finall.T * train_imageMatrix
    #     # np.savetxt('pca_train_matrix.csv', projectedImage, delimiter=',')
    #     pca = PCA(n_components = 10)
    #     all_images = pca.fit_transform(imageMatrix)
    #     print(pca.explained_variance_ratio_)
    #     print("finish pca")
    #     print(all_images)

    def pca(self, k):
        """

        :param data: 待降维的原始数据
        :param k: 保留的特征数，即降低到的维数
        :return: 降维后还原得到的数据
        """
        imageMatrix = []
        for i in self.imgs:
            mats = np.array(i)
            imageMatrix.append(mats.ravel())
        imageMatrix = np.array(imageMatrix)
        imageMatrix = np.transpose(imageMatrix)
        imageMatrix = np.mat(imageMatrix)
        data = imageMatrix.T
        print("原始数据行状：", data.shape)

        # 1. 均值归一化
        X_demean = data - np.mean(data, axis=0)  # 按行操作，取每一列的均值

        # 2. 计算数据的协方差矩阵

        C = X_demean.T @ X_demean / len(X_demean)
        print("协方差矩阵：", C)

        # 3. 计算特征值，特征向量
        # 奇异值分解
        U, S, V = np.linalg.svd(C)
        print("特征值：", S)
        print("特征向量：", U)

        # 4. 实现降维
        U1 = U[:, :100]  # 降到100维，那么取特征向量的前100列
        X_reduction = X_demean @ U1
        print("降维后的数据形状：", X_reduction.shape)  # 仅保留100个特征

        # 5. 数据还原
        # x做过均值归一化，因此还需要加上各个维度的均值。
        X_restore = X_reduction @ U1.T + np.mean(data, axis=0)
        return X_restore
    def draw(self,data):
        fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        for c in range(1):
            for r in range(1):
                # 显示单通道的灰度图像, cmap='Greys_r'
                axis[c, r].imshow(data[0].reshape(64, 64).T)
                axis[c, r].set_xticks([])
                axis[c, r].set_yticks([])
        plt.show()


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
        detect = normalFaceDetect()
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
    # img = cv2.imread('data/raw_data/other/IMG_20221215_153251.jpg')
    detect = normalFaceDetect()
    pre_cropping = preCropping()
    # detect.detect_faces(img)
    # new_img = detect.face_cropping()
    # show_image_with_opencv(new_img,'测试')
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
    # paths = pre_cropping.choose_similar(10)
    # imgs_test = []
    # for i in paths:
    #     image = cv2.imread('./data/cropped_data/other/'+i,cv2.IMREAD_GRAYSCALE)
    #     imgs_test.append(image)
    # detect = normalFaceDetect(imgs_test)
    # data = detect.pca(128)
    # detect.draw(data)

    '''----------------------------------------
    主要流程
    -------------------------------------------'''
    detect = normalFaceDetect()
    pre_cropping = preCropping()
    pre_cropping.pre_do(raw_data_path)
    preCropping().pre_do(target_data_path,'./data/cropped_data/target/')
    # print(pre_cropping.choose_similar(10))
    # detect.detect_faces(img)
    # new_img = detect.face_cropping()

