import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from model.models import TSN
from transforms.tansforms import *

from multiprocessing import set_start_method
try:
    set_start_method('forkserver',force=True)
except RuntimeError:
    pass

if __name__ == '__main__':

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
              dropout=args.OPT.dropout)

    scale_size = net.scale_size
    input_size = net.input_size
    input_mean = net.input_mean
    input_std = net.input_std

    net= torch.nn.DataParallel(net, device_ids=args.RUN.gpus).cuda()

    checkpoint = torch.load(args.TEST.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    net.load_state_dict(checkpoint['state_dict'])

    if args.TEST.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.TEST.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.TEST.test_crops))

    test_transform = torchvision.transforms.Compose([
                               cropping,
                               Stack(roll=args.MODEL.arch == 'BNInception'),
                               ToTorchFormatTensor(div=args.MODEL.arch != 'BNInception'),
                               GroupNormalize(input_mean, input_std)])
    if args.DATA.input_type == 'Frames':
        from dataset_frames import TSNDataSetFrames
        data_loader = torch.utils.data.DataLoader(
            TSNDataSetFrames(args.DATA.test_list, num_segments=args.num_segments,
                       new_length=1 if args.DATA.modality == "RGB" else 5,
                       modality=args.DATA.modality, test_mode=True,
                       image_tmpl="image_{:05d}.jpg" if args.DATA.modality in ["RGB", "RGBDiff"] else args.RUN.flow_prefix+"{}_{:05d}.jpg",
                       transform=test_transform),
            batch_size=1, shuffle=False,
            num_workers=args.RUN.workers, pin_memory=True)
    elif args.DATA.input_type == 'Image':
        from dataset_image import TSNDataSetImage
        data_loader = torch.utils.data.DataLoader(
            TSNDataSetImage(args.DATA.test_list,
                            transform=test_transform),
            batch_size=1, shuffle=False,
            num_workers=args.RUN.workers, pin_memory=True)
    elif args.DATA.input_type == 'Video':
        if args.DATA.video_handler == 'nvvl':
            from dataset_video_nvvl import TSNDataSetVideoNVVL
            test_data_set = TSNDataSetVideoNVVL(args.DATA.test_list, num_segments=args.MODEL.num_segments,
                                            new_length=1,
                                            modality=args.DATA.modality,
                                            transform=test_transform)
            data_loader = torch.utils.data.DataLoader(
                test_data_set,
                batch_size=1, shuffle=False,
                num_workers=args.RUN.workers, pin_memory=True, worker_init_fn=test_data_set.set_video_reader)
        elif args.DATA.video_handler == 'cv2':
            from dataset_video_cv2 import TSNDataSetVideoCV2
            data_loader = torch.utils.data.DataLoader(
                TSNDataSetVideoCV2(args.DATA.test_list, num_segments=args.MODEL.num_segments,
                                   new_length=1,
                                   modality=args.DATA.modality,
                                   transform=test_transform),
                batch_size=1, shuffle=False,
                num_workers=args.RUN.workers, pin_memory=True)

    net.eval()
    data_gen = enumerate(data_loader)
    total_num = len(data_loader.dataset)

    video_scores = []
    video_labels = []

    def eval_video(video_data):
        i, data, label = video_data

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
        print('video #{}, pred:{}, label:{}'.format(i,np.argmax(scores),label[0]))
        return i, scores, label[0]


    proc_start_time = time.time()
    max_num = args.TEST.max_num if args.TEST.max_num > 0 else len(data_loader.dataset)

    for i, (data, label) in data_gen:
        if i >= max_num:
            break
        rst = eval_video((i, data, label))
        video_scores.append(rst[1])
        video_labels.append(rst[2])
        cnt_time = time.time() - proc_start_time
        print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                        total_num,
                                                                        float(cnt_time) / (i+1)))

    video_pred = [np.argmax(x) for x in video_scores]
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

    if args.TEST.save_scores is not None:
        np.savez(args.TEST.save_scores, scores=video_scores, labels=video_labels)



