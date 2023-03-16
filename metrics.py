import os
from PIL import Image
from PIL import ImageFile
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim #era 0.19.2

ImageFile.LOAD_TRUNCATED_IMAGES = True

pathname = os.path.realpath(__file__)
pathname = os.path.split(pathname)[0]
pathname = os.path.join(pathname)

lista_pastas = ["output_test"]

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0): # MSE is zero means no noise is present in the signal
    # Therefore PSNR have no importance
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def get_max_min(img,img_const):
    if img_const.max() > img.max():
        max_ = img_const.max()
    else:
        max_ = img.max()
    if img_const.min() < img.min():
        min_ = img_const.min()
    else:
        min_ = img.min()
    return max_, min_

def SSIM(img,img_const):
    ssim_const = ssim(img, img_const,channel_axis=2)
    return ssim_const

def metrics(pathname,psnr_file,ssim_file):
    data_dir = os.path.join(pathname,'Datasets','test')
    folders = os.listdir(data_dir)
    folder_dir = os.path.join(data_dir,folders[0])
    imgs = os.listdir(folder_dir)

    keys = {'0':'target/target',
            '1':'input/target'}
    folders = {'0':'target',
               '1':'input'}
    
    n = len(list(keys.keys()))
    psnr_file = open(psnr_file,'w+')
    ssim_file = open(ssim_file,'w+')
    psnr_file.write('psnr\nimg,')
    ssim_file.write('ssim\nimg,')
    for k in keys.keys():
        psnr_file.write(keys[k]+',')
        ssim_file.write(keys[k]+',')
    psnr_file.write('\n')
    ssim_file.write('\n')

    psnr_dic={}
    psnr_sum={}
    ssim_dic={}
    ssim_sum={}

    for i,img in enumerate(imgs):
        imgs={}
        for j in range(n):
            img_dir = os.path.join(data_dir,folders[str(j)],img)
            imgs[keys[str(j)].split('/')[0]] = np.array(Image.open(img_dir))

        for k in keys.keys():
            if i == 0:
                psnr_sum[keys[k]] = 0
                ssim_sum[keys[k]] = 0
            psnr_temp = PSNR(imgs[keys[k].split('/')[0]],imgs['target'])
            ssim_temp = SSIM(imgs[keys[k].split('/')[0]],imgs['target'])
            psnr_sum[keys[k]] = psnr_temp + psnr_sum[keys[k]]
            ssim_sum[keys[k]] = ssim_temp + ssim_sum[keys[k]]
            psnr_dic[keys[k]] = str(psnr_temp)
            ssim_dic[keys[k]] = str(ssim_temp)

        psnr_file.write(img+',')
        ssim_file.write(img+',')
        for k in keys.keys():
            psnr_file.write(psnr_dic[keys[k]]+',')
            ssim_file.write(ssim_dic[keys[k]]+',')
        psnr_file.write('\n')
        ssim_file.write('\n')
        
    psnr_avg={}
    ssim_avg={}
    for k in psnr_sum.keys():
        psnr_avg[k] = str(psnr_sum[k] / (i+1))
        ssim_avg[k] = str(ssim_sum[k] / (i+1))
        psnr_sum[k] = str(psnr_sum[k])
        ssim_sum[k] = str(ssim_sum[k])


    psnr_file.write('sum,')
    ssim_file.write('sum,')
    for k in keys.keys():
        psnr_file.write(psnr_sum[keys[k]]+',')
        ssim_file.write(ssim_sum[keys[k]]+',')
    psnr_file.write('\navg,')
    ssim_file.write('\navg,')
    for k in keys.keys():
        psnr_file.write(psnr_avg[keys[k]]+',')
        ssim_file.write(ssim_avg[keys[k]]+',')
    psnr_file.close()
    ssim_file.close()

metrics_dir = os.path.join(pathname,'Metrics')
if not os.path.exists(metrics_dir):
    os.mkdir(metrics_dir)
psnr_file = os.path.join(metrics_dir,'psnr.txt')
ssim_file = os.path.join(metrics_dir,'ssim.txt')
metrics(pathname,psnr_file,ssim_file)