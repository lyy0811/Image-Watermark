import cv2
import numpy as np
import os
import ssim

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

dir_1 = 'Datasets/images/image'
dir_2 = 'Datasets/Recover/result'
fliename_1 = os.listdir(dir_1)
fliename_2 = os.listdir(dir_2)
psnr_list = []
ssim_list = []
for i in range(0,25):
    img_path_1 = os.path.join(dir_1, fliename_1[i])
    #print(img_path_1)
    img_path_2 = os.path.join(dir_2, fliename_2[i])
    #print(img_path_2)
    img1 = cv2.imread(img_path_1)
    img1_1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
    img2 = cv2.imread(img_path_2)
    img1_1 = img1_1.astype(np.float32)
    img2 = img2.astype(np.float32)
    psnr_value = psnr(img1_1, img2)
    ssim_value = ssim.ssim(img1_1, img2)
    #print('psnr:', psnr_value)
    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)

avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
print('avg_PSNR:', avg_psnr)
print('avg_SSIM:', avg_ssim)
