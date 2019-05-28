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
parser.add_argument('num_class', type=int)
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--save_clip_folder', type=str, default=None)
parser.add_argument('--clip_root', type=str, default=None)
parser.add_argument('--save_clip_thresh', type=float, default=0.3,
                    help='class score > thresh, then save clip as this class')
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


net = TSN(args.num_class, 1, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout,before_softmax=False)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.test_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="image_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)


def move_video_clip(frame_folder_path, video_root, pred, save_clip_folder):
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

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)
    rst = net(input_var).data.cpu().numpy().copy()
    scores = np.array(rst.mean(axis=0))

    return i, scores, label[0]

proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, (path, data, label) in data_gen:
    if i >= max_num:
        break
    _, scores, label = eval_video((i, path, data, label))
    print(scores)
    pred = np.argmax(scores)
    move_video_clip(path, args.clip_root, pred, args.save_clip_folder)

    for idx,a_score in enumerate(scores):
        if a_score > args.save_clip_thresh and idx != pred:
            move_video_clip(path, args.clip_root, idx,args.save_clip_folder)

    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))
