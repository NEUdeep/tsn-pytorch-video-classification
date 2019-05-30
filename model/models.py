import torch
from torch import nn
from model.consensus_ops import ConsensusModule, Identity
from transforms.transforms import *
from torch.nn.init import normal, constant
from transforms.group_norm import GroupNorm2d

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=False, use_GN=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self._enable_pbn = partial_bn
        self._enable_gn = use_GN
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
    use_partial_bn:     {}
    use_group_norm:     {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout,
                   self._enable_pbn, self._enable_gn)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._init_with_rgb_model(self.base_model)

            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

    def _init_with_rgb_model(self, base_model):
        rgb_base_model = ''
        checkpoint = torch.load(rgb_base_model)
        pretrained_state = checkpoint['state_dict']
        del pretrained_state['module.new_fc.weight']
        del pretrained_state['module.new_fc.bias']
        new_pretrained_state = dict()
        for k,v in pretrained_state.items():
            new_k = k.replace('module.base_model.','')
            new_pretrained_state[new_k] = v
        model_state = base_model.state_dict()
        model_state.update(new_pretrained_state)
        base_model.load_state_dict(model_state)
        return base_model

    def _prepare_tsn(self, num_class):
        if self.base_model.__class__.__name__ == 'DPN':
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_channels
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)

        return feature_dim


    def _prepare_base_model(self, base_model):
        if 'vgg' in base_model:
            from my_model.vgg import vgg19_bn as MY_VGG
            self.base_model = MY_VGG(False)
            model_dict = self.base_model.state_dict()
            pretrained_model = getattr(torchvision.models, base_model)(True)
            pretrained_dict = pretrained_model.state_dict()
            pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.base_model.load_state_dict(model_dict)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif 'resnet50' in base_model or 'resnet101' in base_model or 'resnet152' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif 'inceptionv3' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        elif 'inceptionv4' in base_model or 'inceptionresnetv2' in base_model:
            import pytorch_model_zoo
            self.base_model = getattr(pytorch_model_zoo, base_model)()
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        elif 'nasnetalarge' in base_model:
            import pytorch_model_zoo
            self.base_model = getattr(pytorch_model_zoo, base_model)(pretrained='imagenet+background')
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 331
            self.input_mean = [0.5]
            self.input_std = [0.5]
        elif 'dpn' in base_model:
            import pytorch_model_zoo
            self.base_model = getattr(pytorch_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classifier' #attention:last layer is a conv
            self.input_size = 224
            self.input_mean =  [124. / 255, 117. / 255, 104. / 255]
            self.input_std = [1 / (.0167 * 255)] * 3
        elif 'resnext101' in base_model:
            from poseidon.models import resnext
            self.base_model = getattr(resnext, base_model)()
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        elif 'xception' in base_model:
            import poseidon.models
            self.base_model = getattr(poseidon.models, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

        if self._enable_gn:
            self._replace_bn_with_gn(self.base_model)
            print("enable GN:",self.base_model)


    def _replace_bn_with_gn(self, model):
        for layer_name, layer_type in model.named_children():
            if isinstance(layer_type, nn.BatchNorm2d):
                channel_num = layer_type.num_features
                setattr(model, layer_name, GroupNorm2d(channel_num))
            elif isinstance(layer_type, nn.Sequential):
                for idx, block in enumerate(layer_type):
                    if isinstance(block, nn.BatchNorm2d):
                        channel_num = block.num_features
                        layer_type.__setitem__(idx, GroupNorm2d(channel_num))  # this method is new in pytorch0.4.0
                    else:
                        self._replace_bn_with_gn(block)


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        gn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, GroupNorm2d):
                gn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        if self._enable_gn:
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                 'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                 'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                 'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                 'name': "normal_bias"},
                {'params': gn, 'lr_mult': 1, 'decay_mult': 0.2,
                 'name': "GN scale/shift"},
            ]
        else:
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                 'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                 'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                 'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                 'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                 'name': "BN scale/shift"},
            ]


    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)
        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self,vgg_style=False):
        if self.modality == 'RGB' and vgg_style:
            resize_range_min = self.input_size * 256 // 224
            resize_range_max = self.input_size * 320 // 224
            return torchvision.transforms.Compose([GroupRandomResizeCrop([resize_range_min,resize_range_max], self.input_size),
                                                   GroupRandomHorizontalFlip(is_flow=False),
                                                   GroupColorJitter(brightness=0.05,contrast=0.05,saturation=0.05,hue=0.05)])
        elif self.modality == 'RGB' and not vgg_style:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
