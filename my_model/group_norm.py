import torch
import torch.nn as nn

__all__ = [
    'GroupNorm2d',
]

class GroupNorm2d(nn.Module):
    def __init__(self, channel_num, group_num = 32, eps = 1e-10, zero_gamma=False):
        super(GroupNorm2d,self).__init__()
        self.group_num = group_num
        if zero_gamma:
            self.gamma = nn.Parameter(torch.zeros(channel_num, 1, 1))
        else:
            self.gamma = nn.Parameter(torch.ones(channel_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(channel_num, 1, 1))
        self.eps = eps

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        N, C, H, W = input.size()
        if C % self.group_num != 0:
            raise ValueError('expected channel num {} can be devided by group num {}'
                             .format(C,self.group_num))

    def forward(self, input):
        N, C, H, W = input.size()

        input = input.view(N, self.group_num, -1)

        mean = input.mean(dim = 2, keepdim = True)
        std = input.std(dim = 2, keepdim = True)

        input = (input - mean) / (std+self.eps)
        input = input.view(N, C, H, W)

        return input * self.gamma + self.beta


if __name__ == '__main__':
    import torch.nn as nn
    import torchvision

    def _replace_bn_with_gn(model, pattern):
        for layer_name, layer_type in model.named_children():
            if isinstance(layer_type, nn.BatchNorm2d):
                channel_num = layer_type.num_features
                setattr(model, layer_name, GroupNorm2d(channel_num))
            elif isinstance(layer_type, nn.Sequential):
                for idx, block in enumerate(layer_type):
                    if isinstance(block, nn.BatchNorm2d):
                        channel_num = block.num_features
                        layer_type.__setitem__(idx, GroupNorm2d(channel_num))  # this method is new in pytorch0.4.0
                        print(block)
                    else:
                        _replace_bn_with_gn(block)


    base_model = torchvision.models.resnet50()
    _replace_bn_with_gn(base_model)
    print(base_model)