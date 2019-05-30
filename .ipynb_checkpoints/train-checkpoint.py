# --------------------------------------------------------
# Copyright (c) 2019 
# Written by Kang Haidong
# --------------------------------------------------------
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from models import TSN
from transforms import *
from opts import parser
import numpy as np
from tensorboardX import SummaryWriter
import torch.onnx
from multiprocessing import set_start_method
try:
    set_start_method('forkserver',force=True)
except RuntimeError:
    pass

best_prec1 = 0


def main():
    # options
    parser = argparse.ArgumentParser(
        description="Standard video-level train")
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file',
                    default='./cfgs/imagenet/res18.yml', type=str)
    from config import cfg, cfg_from_file
    global args, best_prec1
    args = parser.parse_args()
    if args.cfg_from_file is not None:
        cfg_from_file(args.cfg_file)
        


    model = TSN(args.DATA.num_class, args.MODEL.num_segments, args.DATA.modality,
                base_model=args.MODEL.arch,
                consensus_type=args.MODEL.consensus_type, dropout=args.OPT.dropout, partial_bn=args.MODEL.use_partialbn, use_GN=args.OPT.use_gn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(vgg_style=False)

    model = torch.nn.DataParallel(model, device_ids=args.RUN.gpus).cuda()

    if args.RUN.gpus is None:
        GPU_num = torch.cuda.device_count()
    else:
        GPU_num = len(args.RUN.gpus)

    if args.MODEL.resume:
        if os.path.isfile(args.MODEL.resume):
            print(("=> loading checkpoint '{}'".format(args.MODEL.resume)))
            checkpoint = torch.load(args.MODEL.resume)
            if args.MODEL.kinetics_pretrained:
                pretrained_state = checkpoint['state_dict']
                del pretrained_state['module.new_fc.weight']
                del pretrained_state['module.new_fc.bias']
                model_state = model.state_dict()
                model_state.update(pretrained_state)
                model.load_state_dict(model_state)
            else:
                args.MODEL.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                .format(args.MODEL.resume, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.MODEL.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.DATA.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()
    train_transform = torchvision.transforms.Compose([
        train_augmentation,
        Stack(roll=args.MODEL.arch == 'BNInception'),
        ToTorchFormatTensor(div=args.MODEL.arch != 'BNInception'),
        normalize,
    ])
    val_transform = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(roll=args.MODEL.arch == 'BNInception'),
        ToTorchFormatTensor(div=args.MODEL.arch != 'BNInception'),
        normalize,
    ])

    if args.DATA.modality == 'RGB':
        data_length = 1
    elif args.DATA.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if args.DATA.input_type == 'Frames':
        from dataset_frames import TSNDataSetFrames
        train_loader = torch.utils.data.DataLoader(
            TSNDataSetFrames(args.DATA.train_list, num_segments=args.MODEL.num_segments,
                       new_length=data_length,
                       modality=args.DATA.modality,
                       image_tmpl="image_{:05d}.jpg" if args.DATA.modality in ["RGB", "RGBDiff"] else args.RUN.flow_prefix+"{}_{:05d}.jpg",
                       transform=train_transform),
            batch_size=args.MODEL.batch_size, shuffle=True,
            num_workers=args.RUN.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            TSNDataSetFrames(args.DATA.val_list, num_segments=args.MODEL.num_segments,
                             new_length=data_length,
                             modality=args.DATA.modality,
                             image_tmpl="image_{:05d}.jpg" if args.DATA.modality in ["RGB",
                                                                                "RGBDiff"] else args.RUN.flow_prefix + "{}_{:05d}.jpg",
                             random_shift=False,
                             transform=val_transform),
            batch_size=int(args.MODEL.batch_size/GPU_num), shuffle=False,
            num_workers=args.RUN.workers, pin_memory=True)
    elif args.DATA.input_type == 'Image':
        from dataset_image import TSNDataSetImage
        train_loader = torch.utils.data.DataLoader(
            TSNDataSetImage(args.DATA.train_list,
                            transform=train_transform),
            batch_size=args.MODEL.batch_size * args.MODEL.num_segments, shuffle=True,
            num_workers=args.RUN.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            TSNDataSetImage(args.DATA.val_list,
                            transform=val_transform),
            batch_size=int(args.MODEL.batch_size * args.MODEL.num_segments/GPU_num), shuffle=True,
            num_workers=args.RUN.workers, pin_memory=True)
    elif args.DATA.input_type == 'Video':
        if args.DATA.video_handler == 'nvvl':
            from dataset_video_nvvl import TSNDataSetVideoNVVL
            train_data_set = TSNDataSetVideoNVVL(args.DATA.train_list, num_segments=args.MODEL.num_segments,
                                            new_length=data_length,
                                            modality=args.DATA.modality,
                                            transform=train_transform)
            train_loader = torch.utils.data.DataLoader(
                train_data_set,
                batch_size=args.MODEL.batch_size, shuffle=True, timeout=300,
                num_workers=args.RUN.workers, pin_memory=True, worker_init_fn=train_data_set.set_video_reader)
            val_data_set = TSNDataSetVideoNVVL(args.DATA.val_list, num_segments=args.MODEL.num_segments,
                                          new_length=data_length,
                                          modality=args.DATA.modality,
                                          random_shift=False,
                                          transform=val_transform)
            val_loader = torch.utils.data.DataLoader(
                val_data_set,
                batch_size=int(args.MODEL.batch_size/GPU_num), shuffle=False, timeout=300,
                num_workers=args.RUN.workers, pin_memory=True, worker_init_fn=val_data_set.set_video_reader)
        elif args.DATA.video_handler == 'cv2':# cv2,dataload
            from dataset_video_cv2 import TSNDataSetVideoCV2
            train_loader = torch.utils.data.DataLoader(
                TSNDataSetVideoCV2(args.DATA.train_list, num_segments=args.num_segments,#args.DATA.train_list
                                   new_length=data_length,
                                   modality=args.DATA.modality,
                                   transform=train_transform),
                batch_size=args.MODEL.batch_size, shuffle=True,
                num_workers=args.RUN.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                TSNDataSetVideoCV2(args.DATA.val_list, num_segments=args.MODEL.num_segments,
                                   new_length=data_length,
                                   modality=args.DATA.modality,
                                   random_shift=False,
                                   transform=val_transform),
                batch_size=int(args.MODEL.batch_size/GPU_num), shuffle=False,
                num_workers=args.RUN.workers, pin_memory=True)


    # define loss function (criterion) and optimizer
    if args.MODEL.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.OPT.lr,
                                momentum=args.OPT.momentum,
                                weight_decay=args.OPT.weight_decay)

    if args.RUN.evaluate:
        validate(val_loader, model, criterion, 0)
        return
    #writer = SummaryWriter(log_dir = '/workspace/mnt/group/video/kanghaidong/haidong/mixup-cifar10-master/logs')

    for epoch in range(args.MODEL.start_epoch, args.MODEL.epochs):
        adjust_learning_rate(optimizer, epoch, args.OPT.lr_steps)

        # train for one epoch
        #train_loss,train_top1,train_top5 = train(train_loader, model, criterion, optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)
        #writer.add_scalar('train_loss',train_loss,epoch)
        #writer.add_scalar('train_top1',train_top1,epoch)
        #writer.add_scalar('train_top5',train_top5,epoch)

        # evaluate on validation set
        if (epoch + 1) % args.MC.eval_freq == 0 or epoch == args.MODEL.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))
        

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            print('best_prec1',best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.MODEL.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
        #writer.add_scalar('val_loss',loss,epoch)
        #writer.add_scalar('val_top1',top1,epoch)
        #writer.add_scalar('val_top5',top5,epoch)
        #writer.add_scalar('val_prec1',best_prec1,epoch)
    #writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.MODEL.use_partialbn:
        model.module.partialBN(True)
    else:
        model.module.partialBN(False)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()

        input, target_a, target_b, lam = mixup_data(input, target, args.OPT.mixup)
        input_var, target_a_var, target_b_var = map(torch.autograd.Variable, (input,
                                                      target_a, target_b))


        # compute output
        #batch=128,and then out of memory
        try:
            output = model(input_var)
            #loss = criterion(output, target_var)
            loss = mixup_criterion(criterion, output, target_a_var, target_b_var, lam)
            #train_loss += loss.item(),writer.scalar() error ,like is tensor or numpy with tensorboardX
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda,'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise e

        # measure accuracy and record loss
        prec1, prec5 = mixup_accuracy(output.data, target_a, target_b, lam, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.OPT.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.OPT.clip_gradient)
            if total_norm > args.OPT.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.OPT.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.MC.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    #return (losses,top1,top5)


def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.MC.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.RUN.snapshot_pref, args.DATA.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.RUN.snapshot_pref, args.DATA.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.OPT.lr * decay
    decay = args.OPT.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mixup_accuracy(output, target_a, target_b, lam, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target_a.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct_a = pred.eq(target_a.view(1, -1).expand_as(pred))
    correct_b = (1 - lam) * pred.eq(target_b.view(1, -1).expand_as(pred))
    correct_a = correct_a.float()
    correct_b = correct_b.float()
    correct = lam * correct_a + (1 - lam) * correct_b

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':
    main()