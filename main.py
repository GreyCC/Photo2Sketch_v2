import os
import torch
import argparse
import torch.optim as optim
from data import get_training_set, get_training_set_segment
from network import *
from libs import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from datasets import *
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from criterion import LossCriterion_v2, LapLoss, TV, mean_variance_norm
# from models import calc_kl, reparameterize
from lpips import lpips

parser = argparse.ArgumentParser()

parser.add_argument('--pretrained', type=bool, default=True)
# parser.add_argument('--enc_dir', type=str, default='output_residual_multi_fromZero/enc_iter_6300_epoch_1.pth')
# parser.add_argument('--dec_dir', type=str, default='output_residual_multi_fromZero/dec_iter_6300_epoch_1.pth')
parser.add_argument('--P2S_dir', type=str,
                    default='models/P2S_v2.pth')
parser.add_argument('--inv_dir', type=str,
                    default='models/INV_v2.pth')
parser.add_argument("--stylePath", default="data/sketch",
                    help='path to wikiArt dataset')
parser.add_argument("--contentPath", default="data/coco",
                    help='path to MSCOCO dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--fineSize', type=int, default=256, help='crop image size')
parser.add_argument('--crop', type=bool, default=True, help='crop training images')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument("--outf", default="output_residual_BCETwoSegClampv2_Tr31_newLossesLoss1Bigger/",
                    help='folder to output images and model checkpoints')
parser.add_argument("--content_weight", type=float, default=0.1, help='content loss weight')
parser.add_argument("--style_weight", type=float, default=0.4, help='style loss weight, 0.02 for origin')
parser.add_argument("--batchSize", type=int, default=8, help='batch size')
parser.add_argument("--lr", type=float, default=2e-5, help='learning rate')
parser.add_argument("--gpu_id", type=str, default="cuda:0", help='which gpu to use')
# parser.add_argument("--save_interval", type=int, default=5000, help='checkpoint save interval')
# parser.add_argument("--layer", default="r31", help='which features to transfer, either r31 or r41')
parser.add_argument('--nEpochs', type=int, default=1, help='number of epochs to train for')
# parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')

########### Weight of losses ###########
parser.add_argument("--Edge_weight", type=float, default=1, help='Edge loss weight')
parser.add_argument("--GAN_weight", type=float, default=1, help='GAN loss weight')
parser.add_argument("--Seg_weight", type=float, default=1, help='Seg loss weight')
parser.add_argument("--SaC_weight", type=float, default=1, help='Style and Content loss weight')


################# PREPARATIONS #################
opt = parser.parse_args()

# for 1 gpu/cpu-only user: replace with: ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(opt.gpu_id)

os.makedirs(opt.outf, exist_ok=True)
cudnn.benchmark = True


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


################# DATA #################
print('===> Loading datasets')
train_set = get_training_set_segment(opt.contentPath, opt.stylePath)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

################# MODEL #################
P2S = P2Sv2()
inv = Inverse()
VGG = Vgg19()
D = discriminator_v2(device, num_channels=1, base_filter=64)
criterion = LossCriterion_v2(opt.style_weight, opt.content_weight, device=device)

print('---------- encoder architecture -------------')
print_network(P2S)
#
if opt.pretrained:
    if os.path.exists(opt.P2S_dir):
        pretrained_dict = torch.load(opt.P2S_dir, map_location=lambda storage, loc: storage)
        model_dict = P2S.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        P2S.load_state_dict(model_dict)
        # dec.load_state_dict(torch.load(opt.dec_dir))
        print('pretrained Photo2Sketch model is loaded!')
    if os.path.exists(opt.inv_dir):
        pretrained_dict = torch.load(opt.inv_dir, map_location=lambda storage, loc: storage)
        model_dict = inv.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        inv.load_state_dict(model_dict)
        # dec.load_state_dict(torch.load(opt.dec_dir))
        print('pretrained INverser model is loaded!')


################# LOSS & OPTIMIZER #################
L1_criterion_avg = torch.nn.L1Loss(size_average=True)
L2_criterion_avg = torch.nn.MSELoss(size_average=True)
BCE_criterion = torch.nn.BCEWithLogitsLoss()
Lap_criterion = LapLoss(device=device, max_levels=5)
optimizer_G = optim.AdamW(list(P2S.parameters()) + list(inv.parameters()), lr=opt.lr)
optimizer_D = optim.AdamW(D.parameters(), lr=opt.lr)

################# GPU  #################
D.to(device)
P2S.to(device)
inv.to(device)
VGG.to(device)
L1_criterion_avg.to(device)
L2_criterion_avg.to(device)
BCE_criterion.to(device)
Lap_criterion.to(device)


################# TRAINING #################

def train(epoch):
    P2S.train()
    D.train()
    inv.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        img, edge, seg, ref = batch[0], batch[1], batch[2], batch[3]
        img = img.to(device)
        edge = edge.to(device)
        ref = ref.to(device)
        seg = seg.to(device)

        # Extract subject-only from COCO segment dataset
        seg[:, 1:2, :, :] = seg[:, 1:2, :, :] < 0.7
        seg = seg[:, 0:1, :, :] * 0 + seg[:, 1:2, :, :] + seg[:, 2:3, :, :] * 0

        # Set real/fake label for discriminator training
        b, c, h, w = img.shape
        real_label = torch.ones((b, 336)).to(device)
        fake_label = torch.zeros((b, 336)).to(device)

        # forward
        for param in D.parameters():
            param.requires_grad = False

        optimizer_G.zero_grad()

        output, heat_map = P2S(img)
        img_predict = inv(output)
        heat_map = heat_map[:, 0:1, :, :] * 0.299 + heat_map[:, 1:2, :, :] * 0.587 + heat_map[:, 2:3, :, :] * 0.114
        cF2 = VGG(img_predict.repeat(1, 3, 1, 1))
        cF1 = VGG(img)
        sF = VGG(ref)
        tF = VGG(output.repeat(1, 3, 1, 1))

        Edgeloss = L1_criterion_avg(output, edge)

        Segloss = BCE_criterion(output, seg)
        Segloss_map = BCE_criterion(heat_map, abs(1 - seg))

        loss_1, styleLoss, contentLoss1 = criterion(tF, sF, cF1)
        _, _, contentLoss2 = criterion(cF2, sF, cF1)

        D_fake_feat, D_fake_decision = D(output)

        GAN_loss = L1_criterion_avg(D_fake_decision, real_label)

        G_loss = opt.GAN_weight * 5 * GAN_loss + opt.Edge_weight * 200 * Edgeloss + opt.Seg_weight * \
                 (200 * Segloss + 4 * Segloss_map) + opt.SaC_weight * (2 * loss_1 + 0.1 * contentLoss2)  # Loss 1

        # backward & optimization
        G_loss.backward()
        optimizer_G.step()

        # Reset gradient
        for p in D.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        ref = ref[:, 0:1, :, :] * 0.299 + ref[:, 1:2, :, :] * 0.587 + ref[:, 2:3, :, :] * 0.114
        _, D_real_decision = D(ref)
        _, D_fake_decision = D(output.detach())

        real = real_label * np.random.uniform(0.7, 1.2)
        fake = fake_label + np.random.uniform(0.0, 0.3)

        D_loss = (L1_criterion_avg(D_real_decision, real)
                  + L1_criterion_avg(D_fake_decision, fake)) / 2.0

        # Back propagation
        D_loss.backward()
        optimizer_D.step()

        print("===> VAE Epoch[{}]({}/{}): G_loss: {:.4f} || GAN_loss: {:.4f} ||"
              " Edgeloss: {:.4f} || Segloss: {:.4f} || Segloss_map {:.4f} ||"
              " loss1: {:.4f} || contentloss2: {:.4f} || "
              "".format(epoch, iteration,
                        len(training_data_loader),
                        G_loss.data, GAN_loss, Edgeloss.data, Segloss.data,
                        Segloss_map.data, loss_1.data, contentLoss2.data))

        if iteration % 14700 == 0:
            HR_filename = os.path.join('Test', 'example')
            gt_image = [join(HR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]
            for i in range(gt_image.__len__()):
                HR = Image.open(gt_image[i]).convert('RGB')
                HR = modcrop(HR, 8)
                with torch.no_grad():
                    img_test = transform(HR).unsqueeze(0).to(device)
                    output_test, heat_map_test = P2S(img_test)
                img_test = img_test.clamp(0, 1).cpu().data
                output_test = output_test.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
                heat_map_test = heat_map_test.cpu().data
                concat = torch.cat((img_test, heat_map_test, output_test), dim=0)
                vutils.save_image(concat, '%s/%d_%d_test%d.png' % (opt.outf, epoch, iteration, i), normalize=True,
                                  scale_each=True, nrow=1)

            img = img.clamp(0, 1).cpu().data
            edge = edge.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
            seg = seg.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
            heat_map = heat_map.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
            output = output.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
            img_predict = img_predict.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
            concat = torch.cat((img, edge, seg, heat_map, output, img_predict), dim=0)
            vutils.save_image(concat, '%s/%d_%d.png' % (opt.outf, epoch, iteration), normalize=True,
                              scale_each=True, nrow=img.shape[0])

            torch.save(inv.state_dict(), '%s/inv_iter_%d_epoch_%d.pth' % (opt.outf, iteration, epoch))
            torch.save(P2S.state_dict(), '%s/P2S_iter_%d_epoch_%d.pth' % (opt.outf, iteration, epoch))

    return img, output


transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    img, output = train(epoch)
    # content = torch.cat((img1, img2), dim=3)
    # style = style.repeat(1, 1, 1, 2)
    # transfer = torch.cat((transfer1, transfer2), dim=3)
    # learning rate is decayed by a factor of 10 every half of total epochs
