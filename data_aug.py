import cv2, os
import numpy as np
import skimage
from skimage.util import random_noise
    
def noise_gauss(image, mean_x):
    noise_img = random_noise(image, mode='gaussian',mean = mean_x)
    noise_img = np.array(255*noise_img, dtype = 'uint8') 
    return noise_img

def blur(image, name_blur, kernel_size):   
    # make sure that you have saved it in the same folder 
    # Averaging -> You can change the kernel size as you want 
    if name_blur == 'avging':
        avging = cv2.blur(image,(kernel_size,kernel_size)) #10x10
        return avging
    
    elif name_blur == 'gaussian':
        # Gaussian Blurring -> Again, you can change the kernel size 
        gausBlur = cv2.GaussianBlur(image, (kernel_size,kernel_size),0) #5x5
        return gausBlur

def adjust_gamma(image, gamma=1.0):
    #O gama é o valor relativo de claro e escuro da imagem.
    #adjusted = adjust_gamma(image, gamma=2.0)#0.3
    #gamma maior clareia 
    #gamma menor escurece 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def rotacao(image, graus):
    altura, largura = image.shape[:2]
    ponto = (largura / 2, altura / 2) #ponto no centro da figura
    rotacao = cv2.getRotationMatrix2D(ponto, graus, 1.0)
    rotacionado = cv2.warpAffine(image, rotacao, (largura, altura))
    return rotacionado

def aplicarTransformacao(image, tipoTransformacao):
    new_image = image
    
    if tipoTransformacao == "normal":
        new_image = image

    elif tipoTransformacao == "noise1":
        new_image = noise_gauss(image,0.01)
    elif tipoTransformacao == "noise2":
        new_image = noise_gauss(image,0.03)
    elif tipoTransformacao == "noise3":
        new_image = noise_gauss(image,0.05)
    
    elif tipoTransformacao == "blur1":
        new_image = blur(image,"gaussian",3)
    elif tipoTransformacao == "blur2":
        new_image = blur(image,"gaussian",5)
    elif tipoTransformacao == "blur3":
        new_image = blur(image,"gaussian",9)
    
    elif tipoTransformacao == "gamma1":
        new_image = adjust_gamma(image,0.5)
    elif tipoTransformacao == "gamma2":
        new_image = adjust_gamma(image,1.5)
    elif tipoTransformacao == "gamma3":
        new_image = adjust_gamma(image,2.0)
    
    elif tipoTransformacao == "rotacao1":
        new_image = rotacao(image,5)
    elif tipoTransformacao == "rotacao2":
        new_image = rotacao(image,10)
    elif tipoTransformacao == "rotacao3":
        new_image = rotacao(image,15)
             
    return new_image

if __name__ == '__main__':
    pathname = os.path.realpath(__file__)
    pathname = os.path.split(pathname)[0]
    transformacoes = ['normal','noise1','noise2','noise3','gamma1','gamma2','gamma3','rotacao1','rotacao2','rotacao3']
    # transformacoes = ['normal']
    # transformacoes = ['noise1','noise2','noise3']
    # transformacoes = ['blur1','blur2','blur3']
    # transformacoes = ['gamma1','gamma2','gamma3']
    transformacoes = ['rotacao1','rotacao2','rotacao3']

    train_path = os.path.join(pathname, 'Dataset', 'train')
    folders = os.listdir(train_path)
    da_dir = os.path.join(pathname, 'Data_Aug', 'train')

    if not os.path.exists(da_dir):
        os.mkdir(da_dir)
        for folder in folders:
            if not os.path.exists(os.path.join(da_dir,folder)):
                os.mkdir(os.path.join(da_dir,folder))
    
    error_log = open(os.path.join(pathname,'data_aug_error_log.txt'),'a')
    for folder in folders:
        print(folder)
        imgs = os.listdir(os.path.join(train_path,folder))
        for img_name in imgs:
            img_path = os.path.join(train_path,folder,img_name)
            img = cv2.imread(img_path)
            for tipoTransformacao in transformacoes:
                new_img_name = img_name.split('.')[0] + '_' + tipoTransformacao + '.png'
                path = os.path.join(da_dir,folder,new_img_name)
                try:
                    new_img = aplicarTransformacao(img, tipoTransformacao)
                    # new_img = cv2.resize(new_img, dsize=(200, 200), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(path,new_img)
                except:
                    print('ERROR: '+folder+','+img_name+','+tipoTransformacao)
                    error_log.write(folder+','+img_name+','+tipoTransformacao+'\n')
    error_log.close()

print("fim")

