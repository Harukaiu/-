import numpy as np
import os
import glob
import cv2
import string
import random


image_size = 224
photos_dir = "Flower/"
output_dir= "Flower_aug/"
os.makedirs(output_dir,exist_ok=True)
files = glob.glob(photos_dir + "/*")
interval = image_size//np.random.uniform(2, 4) 
thresh=0.3

def grid_mixer(img_1, img_2, interval, thresh):
    #make checkerboad
    h, w, _ = img_1.shape
    h_start = np.random.randint(0,2*interval)
    w_start = np.random.randint(0,2*interval)
    h_grid = ((np.arange(h_start, h_start+h)//interval)%2).reshape(-1,1)
    w_grid = ((np.arange(w_start, w_start+w)//interval)%2).reshape(1,-1)
    checkerboard = np.abs(h_grid-w_grid)

    #reverse vertical and/or horizontal
    if np.random.rand()<thresh:
        checkerboard += h_grid*w_grid
    if np.random.rand()<thresh:
        checkerboard += (1-h_grid)*(1-w_grid)

    #mix images
    mixed_img = img_1*checkerboard[:, :, np.newaxis]+img_2*(1-checkerboard[:, :, np.newaxis])
    return mixed_img

                      

def ImgGamma(pathNow,filename,img):
    gamma1 = np.zeros((256,1),dtype = 'uint8')
    gamma2 = np.zeros((256,1),dtype = 'uint8')
    gamma3 = np.zeros((256,1),dtype = 'uint8')
    gamma4 = np.zeros((256,1),dtype = 'uint8')
    gamma5 = np.zeros((256,1),dtype = 'uint8')
    for i in range(256):
        gamma1[i][0] = 255 * (float(i)/255) ** (1.0/0.8)
        gamma2[i][0] = 255 * (float(i)/255) ** (1.0/1.0)
        gamma3[i][0] = 255 * (float(i)/255) ** (1.0/1.5)
        gamma4[i][0] = 255 * (float(i)/255) ** (1.0/2.0)
        gamma5[i][0] = 255 * (float(i)/255) ** (1.0/2.2)
    l = [1, 2, 3, 4, 5]
    c = string.ascii_lowercase + string.ascii_uppercase + string.digits
    rand_name = "".join([random.choice(c) for i in range(10)])
    # img = cv2.imread(name)
    gamma_now = random.choice(l)
    if(gamma_now == 1):
        img_gamma = cv2.LUT(img,gamma1)
        cv2.imwrite(pathNow+"/"+rand_name+".png", img_gamma)
    if(gamma_now == 2):
        img_gamma = cv2.LUT(img,gamma2)
        cv2.imwrite(pathNow+"/"+rand_name+".png", img_gamma)
    if(gamma_now == 3):
        img_gamma = cv2.LUT(img,gamma3)
        cv2.imwrite(pathNow+"/"+rand_name+".png", img_gamma)
    if(gamma_now == 4):
        img_gamma = cv2.LUT(img,gamma4)
        cv2.imwrite(pathNow+"/"+rand_name+".png", img_gamma)
    if(gamma_now == 5):
        img_gamma = cv2.LUT(img,gamma5)
        cv2.imwrite(pathNow+"/"+rand_name+".png", img_gamma)
def hsv_h(pathNow,filename,img,h_deg):
    img_hsv_h = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # 色空間をBGRからHSVに変
    img_hsv_h[:,:,(0)] = img_hsv_h[:,:,(0)]+h_deg # 色相の計算
    img_bgr = cv2.cvtColor(img_hsv_h,cv2.COLOR_HSV2BGR) # 色空間をHSVからBGRに変換
    rand_name = "".join([random.choice(c) for i in range(10)])
    cv2.imwrite(pathNow+"/"+rand_name+".png", img_bgr) # 画像の保存
  

def hsv_s(pathNow,filename,img,s_mag):
    img_hsv_s = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # 色空間をBGRからHSVに変
    img_hsv_s[:,:,(1)] = img_hsv_s[:,:,(1)]*s_mag # 彩度の計算
    img_bgr = cv2.cvtColor(img_hsv_s,cv2.COLOR_HSV2BGR) # 色空間をHSVからBGRに変換
    rand_name = "".join([random.choice(c) for i in range(10)])
    cv2.imwrite(pathNow+"/"+rand_name+".png", img_bgr) # 画像の保存
    
if __name__ == '__main__':
    f = files[-1]  
    img_2 = cv2.imread(f)
    img_2 = cv2.resize(img_2, dsize=(image_size, image_size))

    for i, file in enumerate(files):
        img_filename = os.path.split(file)[1]
        path, ext = os.path.splitext( os.path.basename(file) )
        # #元画像も保存
        img = cv2.imread(file)
        c = string.ascii_lowercase + string.ascii_uppercase + string.digits
        img = cv2.resize(img, dsize=(image_size, image_size))
        rand_name = "".join([random.choice(c) for i in range(10)])
        cv2.imwrite(output_dir +"/" + rand_name+".png" , img)#オリジナル保存
        

        ImgGamma(output_dir,img_filename,img)
        h_deg = random.randint(30, 60)#色味1
        hsv_h(output_dir,img_filename,img,h_deg)
        #h_deg = random.randint(90, 150)#色味2
        #hsv_h(output_dir,img_filename,img,h_deg)
        s_mag = 0.5 #彩度
        hsv_s(output_dir,img_filename,img,s_mag)  
        img_grid = grid_mixer(img, img_2, interval, thresh)
        rand_name = "".join([random.choice(c) for i in range(10)])
        cv2.imwrite(output_dir +"/" + rand_name+".png" , img_grid)#オリジナル保存
        img_2 = img
