# --------------------------------------------------------
# Copyright (c) 2019 
# Written by Kang Haidong
# --------------------------------------------------------

import os
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import numpy as np

cur_pth = os.getcwd()

__C = edict()
cfg = __C

#
# Global options
#
#__C.workers = 4  # number of data loading workers

#
# Data options
#
__C.DATA = edict()
__C.DATA.num_class = 6  # number of classes
__C.DATA.modality = 'RGB'  # modality of input, ect. choices=['RGB', 'Flow', 'RGBDiff']
__C.DATA.train_list = 'train'  # train list of your datasets
__C.DATA.val_list = 'val'  # val folder
__C.DATA.test_list = 'test' # test folder
__C.DATA.input_type = 'Frames'  # the type of input, etc. choices=['Image','Frames','Video']
__C.DATA.video_handler = 'cv2'  # the type of reading video, etc. choices=['nvvl','cv2']
#
# Model options
#
__C.MODEL = edict()
__C.MODEL.arch = 'resnet50' # model architecture
__C.MODEL.num_segments = '3' # number of segments
__C.MODEL.consenus_type = 'avg' # the type of consenus segment, etc. choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn']
__C.MODEL.k = '3' # number of segments
__C.MODEL.epochs = 80 # number of total epochs to run
__C.MODEL.start_epoch = 0 # manual epoch number (useful on resume)
__C.MODEL.resume = '' # path to latest checkpoint(default: 4)
__C.MODEL.kinetics_pretrained = 'True' # use kinetics model as pretrained model, use with resume
__C.MODEL.use_partialbn = False # partialbn
__C.MODEL.batch_size = 125 # mini-batch size (default: 256)
__C.MODEL.ckpt = 'ckpt/V1.0/rensnet50/' # path to save checkpoint
__C.MODEL.loss_type = 'nll' # the type of loss, etc. choices=['nll']
#
# Optimizer options
#
__C.OPT = edict()
__C.OPT.optim_type = 'SGD'  # optimizer type: ASGD, LBFGS, RMSprop, Rprop, SGD, Adadelta, Adagrad, Adam, SparseAdam, Adamax
__C.OPT.gamma = 0.1  # base_lr is multiplied by gamma on lr_schedule
__C.OPT.momentum = 0.9  # momentum
__C.OPT.weight_decay = 5e-4  # weight_decay (default: 5e-4)
__C.OPT.dropout = 0.5 # dropout (default: 0.8)
__C.OPT.clip_gradient = 'None' # gradient norm clipping (default: disabled)
__C.OPT.lr = 0.01  # base learning rate
__C.OPT.lr_step = [30, 60]  # decrease learning rate at these epochs, epochs to decay learning rate by 10
__C.OPT.use_gn = False # use group norm clipping (default: False), action='store_true'
__C.OPT.mixup = 0  # add mixup, with alpta value. if 0, no mixup
#
#Monitor configs
__C.MC = edict()
__C.MC.print_freq = 20 # print frequency (default: 10)
__C.MC.eval_freq = 5 # evaluation frequency (default: 5)
#
# Runtime configs
#
__C.RUN = edict()
__C.RUN.e = ''  # evaluate model on validation set
__C.RUN.snapshot_pref = ""
__C.RUN.gpus = 'None'
__C.RUN.flow_prefix = ""
__C.RUN.workers = 4  # number of data loading workers
__C.RUN.evaluate = 'True' # evaluate model on validation set
#
# logger options
#
__C.LOG = edict()
__C.LOG.log_path = ''  # path for logger
#
#  test options
#
__C.TEST = edict()
__C.TEST.weights = 'path/pth.tar' # path of your model 
__C.TEST.max_num = '-1' # the length of dataset
__C.TEST.test_crops = '10' # crop op on test model
__C.TEST.input_size = '224' # size of input, etc. the size of frame.
__C.TEST.save_scores = 'None' # save scores
__C.TEST.test_segments = 25 # number of test segments
__C.TEST.clip_root = 'None' # the toot of clip file
__C.TEST.save_clip_thresh = 0.3 # class score > thresh, then save clip as this class
__C.TEST.save_clip_folder = 'None' # folder of clip file

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])

        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif old_type is type(None):
                pass
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value


