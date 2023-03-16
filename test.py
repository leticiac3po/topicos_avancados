import os
from PIL import Image

import torch.nn as nn
import torch
import utils

from MPRNet import MPRNet
from skimage import img_as_ubyte
import torchvision.transforms.functional as TF
import torch.nn.functional as F

def test(file_path):

    pathname = os.path.realpath(__file__)
    pathname = os.path.split(pathname)[0]
    pathname = os.path.join(pathname)

    derain_model = MPRNet()
    weights = os.path.join(pathname, "Deraining", "pretrained_models", "model_derain.pth")
    utils.load_checkpoint(derain_model,weights)
    derain_model.cuda()
    derain_model = nn.DataParallel(derain_model)
    derain_model.eval()

    deblur_model = MPRNet()
    weights = os.path.join(pathname, "Deblurring", "pretrained_models", "model_best.pth")
    utils.load_checkpoint(deblur_model,weights)
    deblur_model.cuda()
    deblur_model = nn.DataParallel(deblur_model)
    deblur_model.eval()

    img = Image.open(file_path).convert('RGB')
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    input_    = TF.to_tensor(img).unsqueeze(0).cuda()
    img_multiple_of = 8
    h,w = input_.shape[2], input_.shape[3]
    H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-h if h%img_multiple_of!=0 else 0
    padw = W-w if w%img_multiple_of!=0 else 0
    input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

    with torch.no_grad():
        restored = derain_model(input_)
        restored = deblur_model(restored)
    restored = torch.clamp(restored[0],0,1)
    restored = restored[:,:,:h,:w]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored_img = img_as_ubyte(restored[0])
    
    utils.save_img((file_path.split('.')[0]+'_restored'), restored_img)
