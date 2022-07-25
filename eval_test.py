import os
import argparse

import torch

from datasets import is_image_file
from network import *
from network_old import Encoder, Img_decoder_v3
import torch.backends.cudnn as cudnn
from torchvision import transforms
from os.path import *
from os import listdir
import torchvision.utils as vutils
from PIL import Image
from img_utils import modcrop
import lpips

parser = argparse.ArgumentParser()

parser.add_argument('--pretrained', type=bool, default=True)
# parser.add_argument('--parent_dir', type=str, default='models')
parser.add_argument('--P2S_dir', type=str, default='models/P2S_v2.pth')
parser.add_argument('--enc_dir', type=str, default='models/enc.pth')
parser.add_argument('--dec_dir', type=str, default='models/dec.pth')
parser.add_argument("--image_dataset", default="Test/", help='image dataset')

################# PREPARATIONS #################
opt = parser.parse_args()

device = torch.device("cuda:1")
cudnn.benchmark = True


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


################# MODEL #################
P2S = P2Sv2()
enc = Encoder()
dec = Img_decoder_v3()
inv = Inverse()
VGG = Vgg19()
loss_fn_alex_sp = lpips.LPIPS(net='alex')


if opt.pretrained:
    if os.path.exists(opt.P2S_dir):
        pretrained_dict = torch.load(opt.P2S_dir, map_location=lambda storage, loc: storage)
        model_dict = P2S.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        P2S.load_state_dict(model_dict)
        print('pretrained Photo2Sketch model is loaded!')
    if os.path.exists(opt.dec_dir):
        pretrained_dict = torch.load(opt.dec_dir, map_location=lambda storage, loc: storage)
        model_dict = dec.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        dec.load_state_dict(model_dict)
        print('pretrained Decoder model is loaded!')
    if os.path.exists(opt.enc_dir):
        pretrained_dict = torch.load(opt.enc_dir, map_location=lambda storage, loc: storage)
        model_dict = enc.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        enc.load_state_dict(model_dict)
        print('pretrained Encoder model is loaded!')

################# GPU  #################
enc.to(device)
dec.to(device)
P2S.to(device)
VGG.to(device)
loss_fn_alex_sp.to(device)


################# Testing #################
def eval():
    enc.eval()
    dec.eval()

    HR_filename = os.path.join(opt.image_dataset, 'example')
    SR_filename = os.path.join(opt.image_dataset, 'our')

    gt_image = [join(HR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]
    output_image = [join(SR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]

    for i in range(gt_image.__len__()):
        HR = Image.open(gt_image[i]).convert('RGB')
        HR = modcrop(HR, 8)
        with torch.no_grad():
            img = transform(HR).unsqueeze(0).to(device)
            feat = enc(img)
            out_old = dec(feat)
            out_new, heat_map = P2S(img)
        torch.cuda.empty_cache()

        img = img.clamp(0, 1).cpu().data
        out_old = out_old.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
        out_new = out_new.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
        heat_map = heat_map.cpu().data
        concat = torch.cat((img, out_new, heat_map), dim=0)
        vutils.save_image(concat, f'{output_image[i][:-4]}_heatmap.png', normalize=True,
                          scale_each=True, nrow=3)

transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
])

eval()
