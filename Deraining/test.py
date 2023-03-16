import os
from PIL import Image

import torch.nn as nn
import torch
import utils

from MPRNet import MPRNet
from skimage import img_as_ubyte
import torchvision.transforms.functional as TF
import torch.nn.functional as F

pathname = os.path.realpath(__file__)
pathname = os.path.split(pathname)[0]
pathname = os.path.join(pathname)

def test(pathname,pasta):

    model_restoration = MPRNet()
    weights = os.path.join(pathname, "pretrained_models", "model_best.pth")
    utils.load_checkpoint(model_restoration,weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()
    img_multiple_of = 8

    folders_path_img = os.path.join(pathname,'Datasets', pasta,'input')
    folder = [f for f in os.listdir(folders_path_img)]
    result_dir = os.path.join(pathname,'Datasets','result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    
    for filename in folder:
        file_path = os.path.join(pathname,'Datasets',pasta,'input',filename)
        img = Image.open(file_path).convert('RGB')
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_    = TF.to_tensor(img).unsqueeze(0).cuda()

        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-h if h%img_multiple_of!=0 else 0
        padw = W-w if w%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        with torch.no_grad():
            restored = model_restoration(input_)
        restored = torch.clamp(restored[0],0,1)
        restored = restored[:,:,:h,:w]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored_img = img_as_ubyte(restored[0])
        
        utils.save_img((os.path.join(result_dir, filename)), restored_img)

test(pathname,'test')