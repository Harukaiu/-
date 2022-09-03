#hist matching
#類似度判定のみで植物の判別
import cv2
import os
import numpy as np
import shutil
from PIL import Image
import judge
import sys
import os,glob
import cv2
from matplotlib import pyplot as plt
import copy
import openpyxl
import datetime

classes = ["Flower","Food"]

dir_name="" #ここ変える
output_dirname="" #ここ変える

for index, classlabel in enumerate(classes):
    photos_dir = dir_name+ classlabel#ここ変える
    files = glob.glob(photos_dir + "/*")
    IMG_DIR = dir_name +"/com/"
    for i, f in enumerate(files):
        compare_result = []

        TARGET_FILE = f
        
        target_img_path = f
        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(target_img_path)


        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        detector = cv2.AKAZE_create()
        (target_kp, target_des) = detector.detectAndCompute(target_img, None)

        # print('TARGET_FILE: %s' % (TARGET_FILE))
        Files = glob.glob(photos_dir + "/*")
        #フォルダ内走査
        for I, F in enumerate(Files):
            F = f
            if F == '.DS_Store' or F == TARGET_FILE:
                continue

            comparing_img_path = F
            try:
                comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
                
                (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
                matches = bf.match(target_des, comparing_des)
                dist = [m.distance for m in matches]
                ret = sum(dist) / len(dist)
                if ret>key_number:
                    ret=1000
            except cv2.error:
                ret = 100000
            # print(F, ret)
            compare_result.append(ret)
        compare_result = np.array(compare_result)
        if np.min(compare_result) < 10:#ここ変える
            os.remove(f)

       


