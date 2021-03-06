# -*- coding:utf-8 -*-
import argparse
import time

from PIL import Image
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from models import TSN
from transforms import *

from multiprocessing import set_start_method
try:
    set_start_method('forkserver',force=True)
except RuntimeError:
    pass

if __name__ == '__main__':
    # options
    parser = argparse.ArgumentParser(
        description="Standard video-level testing")
    parser.add_argument('num_class', type=int)
    parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('test_list', type=str)
    parser.add_argument('weights', type=str)
    parser.add_argument('--input_type', type=str, default='Frames', choices=['Image','Frames','Video'])
    parser.add_argument('--video_handler',type=str, default='nvvl', choices=['nvvl','cv2'])
    parser.add_argument('--arch', type=str, default="resnet101")
    parser.add_argument('--save_scores', type=str, default=None)
    parser.add_argument('--num_segments', type=int, default=25)
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
              dropout=args.dropout)

    scale_size = net.scale_size #256,这个指的是需要resize或者scale的大小；主要作用是这个；
    input_size = net.input_size # 这个参数指的是输入网络的大小;主要作用是在resize后的img上进行tranfrom，比如crop等等，直接处理成224*224的方块，目的是输入到conv中；
    input_mean = net.input_mean
    input_std = net.input_std

    net= torch.nn.DataParallel(net, device_ids=args.gpus).cuda()

    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    net.load_state_dict(checkpoint['state_dict'])

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            #GroupScale(scale_size),
            #GroupRandomSizedCrop(input_size),
            #GroupCenterCrop(input_size),
            GroupfixCropandCenterCrop(input_size),
            
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

    test_transform = torchvision.transforms.Compose([
                               cropping,
                               Stack(roll=args.arch == 'BNInception'),
                               ToTorchFormatTensor(div=args.arch != 'BNInception'),
                               GroupNormalize(input_mean, input_std)])


    if args.input_type == 'Frames':
        from dataset_frames import TSNDataSetFrames
        data_loader = torch.utils.data.DataLoader(
            TSNDataSetFrames(args.test_list, num_segments=args.num_segments,
                       new_length=1 if args.modality == "RGB" else 5,
                       modality=args.modality, test_mode=True,
                       image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                       transform=test_transform),
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.input_type == 'Image':
        from dataset_image import TSNDataSetImage
        data_loader = torch.utils.data.DataLoader(
            TSNDataSetImage(args.test_list,
                            transform=test_transform),
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.input_type == 'Video':
        if args.video_handler == 'nvvl':
            from dataset_video_nvvl import TSNDataSetVideoNVVL
            test_data_set = TSNDataSetVideoNVVL(args.test_list, num_segments=args.num_segments,
                                            new_length=1,
                                            modality=args.modality,
                                            transform=test_transform)
            data_loader = torch.utils.data.DataLoader(
                test_data_set,
                batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True, worker_init_fn=test_data_set.set_video_reader)
        elif args.video_handler == 'cv2':
            from dataset_video_cv2 import TSNDataSetVideoCV2
            data_loader = torch.utils.data.DataLoader(
                TSNDataSetVideoCV2(args.test_list, num_segments=args.num_segments,
                                   new_length=1,
                                   modality=args.modality,
                                   test_mode=True,
                                   transform=test_transform),
                batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True)

    net.eval()
    data_gen = enumerate(data_loader)
    total_num = len(data_loader.dataset)

    video_scores = []
    video_labels = []

    def eval_video(video_data):
        i, data, label = video_data

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
        #print(rst)
        scores = np.array(rst.mean(axis=0))
        print('video #{}, pred:{}, label:{}'.format(i,np.argmax(scores),label[0]))
        return i, scores, label[0]


    proc_start_time = time.time()
    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
    print('max_num is :',max_num)
    print('scale_size is: ',scale_size)
    print('input_size is: ',input_size)

    for i, (data, label) in data_gen:
        print(i) # i =0
        d1 = data.numpy() #将tensor的4维图像(1,30,224,224)转为array
        num_frame = d1.shape[1]/30 #取出第二维，算出有多少帧
        all_frame = np.split(d1,num_frame,axis=1) # 按照channel维度进行数组的拆分；拆分个数为帧数；返回一个list，包含每一帧的array。

        for frame in all_frame:
            #print(frame)
            tensor_frame = torch.from_numpy(frame) # 将每一帧转为tensor；shape:(1,30,224,224).时刻检查shape很关键。
            #print(tensor_frame)
            rst = eval_video((i, tensor_frame, label))
            #print(rst[1])
            video_scores.append(rst[1])
            #print(video_scores)
            video_labels.append(rst[2])
            cnt_time = time.time() - proc_start_time
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                        total_num,
                                                                        float(cnt_time) / (i+1)))
            
        if i >= max_num:
            break
        

    video_pred = [np.argmax(x) for x in video_scores]
    import collections

    
    d = dict() # for save predict per frame
    d = collections.OrderedDict()
    #label_name = ['BK_Fighting','Fighting_Sports','Others']
    #label_name = ['BK_Fighting','Others'] # for V0.5
    label_name = ['BK_March_Banne','Normal_Banner','People_Gathering','BK_Fighting','Fighting_Sports','Others'] #for num_label=5
    print(video_scores)
    for a in video_scores:
        k=np.argmax(a)
        print(k)
        value = a[k]
        print(value)
        d.update({value:label_name[k]})

    print(d)

    import pickle
    with open("./predict_per_frame_zhaji01.file",'wb') as f:
        pickle.dump(d,f)
        print("Save predict per frame done!")

    


    video_hit = np.sum(np.array(video_pred) == np.array(video_labels))
    video_acc = video_hit / float(len(video_labels))

    print('Video-level Accuracy {:.02f}%'.format(video_acc*100))


    cf = confusion_matrix(video_labels, video_pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    recog_as_cls_cnt = cf.sum(axis=0)
    cls_hit = np.diag(cf)


    cls_recall = cls_hit / cls_cnt
    print('Label-level Recall {:.02f}%'.format(np.mean(cls_recall) * 100))
    print(cls_recall)
    cls_acc = cls_hit / (recog_as_cls_cnt+1e-9)
    print('Label-level Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    print(cls_acc)

    print('Confusion Matrix:')
    print(cf)

    if args.save_scores is not None:
        np.savez(args.save_scores, scores=video_scores, labels=video_labels)




