# --------------------------------------------------------
# Copyright (c) 2019 
# Written by Kang Haidong
# --------------------------------------------------------


import argparse
import time
import os
import shutil

import numpy as np
import torch.nn.parallel
import torch.optim

from dataset import TSNDataSet
from models import TSN
from transforms import *


# options
parser = argparse.ArgumentParser(
        description="Standard video-level testing")
parser.add_argument('--cfg', dest='cfg_file', help='optional config file',
                    default='./configs/V1.0_test.yml', type=str)
from config import cfg, cfg_from_file
args = parser.parse_args()
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)


net = TSN(args.DATA.num_class, 1, args.DATA.modality,
          base_model=args.MODEL.arch,
          consensus_type=args.MODEL.consenus_type,
          dropout=args.OPT.dropout,before_softmax=False)

checkpoint = torch.load(args.TEST.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.TEST.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.TEST.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.TEST.test_crops))

data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.DATA.test_list, num_segments=args.TEST.test_segments,
                   new_length=1 if args.DATA.modality == "RGB" else 5,
                   modality=args.DATA.modality,
                   image_tmpl="image_{:05d}.jpg" if args.DATA.modality in ['RGB', 'RGBDiff'] else args.RUN.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.MODEL.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.MODEL.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.RUN.workers * 2, pin_memory=True)

if args.RUN.gpus is not None:
    devices = [args.RUN.gpus[i] for i in range(args.RUN.workers)]
else:
    devices = list(range(args.RUN.workers))

net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)


def move_video_clip(frame_folder_path, video_root, pred, save_clip_folder):# set a new file to save clip video
    frame_folder_path = frame_folder_path[0]
    new_video_folder = os.path.join(save_clip_folder,str(pred))
    if not os.path.exists(new_video_folder):
        os.makedirs(new_video_folder)

    video_name = frame_folder_path.split('/')[-1]
    old_video_path = os.path.join(video_root, video_name)
    print('moving video {} to {}'.format(video_name,new_video_folder))
    shutil.copy(old_video_path,new_video_folder)


def eval_video(video_data):
    i, path, data, label = video_data

    if args.DATA.modality == 'RGB':
        length = 3
    elif args.DATA.modality == 'Flow':
        length = 10
    elif args.DATA.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.DATA.modality)

    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)
    rst = net(input_var).data.cpu().numpy().copy()
    scores = np.array(rst.mean(axis=0))

    return i, scores, label[0]

proc_start_time = time.time()
max_num = args.TEST.max_num if args.TEST.max_num > 0 else len(data_loader.dataset)

for i, (path, data, label) in data_gen:
    if i >= max_num:
        break
    _, scores, label = eval_video((i, path, data, label))
    print(scores)
    pred = np.argmax(scores)
    move_video_clip(path, args.TEST.clip_root, pred, args.TEST.save_clip_folder )

    for idx,a_score in enumerate(scores):
        if a_score > args.TEST.save_clip_thresh and idx != pred:
            move_video_clip(path, args.TEST.clip_root, idx,args.TEST.save_clip_folder )

    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1))) # no cf matrix