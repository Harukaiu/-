#hist matching
#類似度判定のみで植物の判別
import cv2
import os
import numpy as np
import shutil
import sys
import os,glob
import cv2

classes = ["Flower","Food"]

dir_name="" #ここ変える＆予備用意しとく

for index, classlabel in enumerate(classes):
    photos_dir = dir_name+ classlabel#ここ変える
    files = glob.glob(photos_dir + "/*")
    for i, f in enumerate(files):
        compare_result = []
        target_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(f)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        detector = cv2.AKAZE_create()
        (target_kp, target_des) = detector.detectAndCompute(target_img, None)

        Files = glob.glob(photos_dir + "/*")
        #フォルダ内走査
        for I, F in enumerate(Files):
            if F == '.DS_Store' or F == f:
                continue

            try:
                comparing_img = cv2.imread(F, cv2.IMREAD_GRAYSCALE)
                
                (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
                matches = bf.match(target_des, comparing_des)
                dist = [m.distance for m in matches]
                ret = sum(dist) / len(dist)
            except cv2.error:
                ret = 100000
            # print(F, ret)
            compare_result.append(ret)
        compare_result = np.array(compare_result)
        if len(compare_result) ==0:break
        else:
            np.min(compare_result) < 90#ここ変える
            os.remove(f)

       


