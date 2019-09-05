import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numpy as np
import numbers
import math
import torch


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


#define a new fix transfroms for monitor video Fighting testing. it boosted acc from 1% to 90% for actual monitor video in lingang.
#by Kang Haidong
class GroupfixCropandCenterCrop(object):
    def __init__(self,size):
        if isinstance(size,numbers.Number):
            self.size = (int(size),int(size))
        else:
            self.size = size
    def __call__(self,img_group):
        w,h = img_group[0].size
        y1 = h
        x1 = w*3/4
        th,tw = self.size
        out_images = list()
        from torchvision import transforms
        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            img1 = img.crop((0,0,x1,y1))
            crop = transforms.CenterCrop(self.size)
            out_images.append(crop(img1)) #size is crop_size is 224*224

        return out_images


#define a new dynamic transfroms for monitor video Fighting testing.
#by Kang Haidong
class GroupDynamicCropandCenterCrop(object):
    def __init__(self,size):
        if isinstance(size,numbers.number):
            self.size = (int(size),int(size))
        else:
            self.size = size 

    def __call__(self,img_group):
        w,h = img_group[0].size 
        zx = w*3/4
        zy = h*4/3
        y1 = random.randint(0,zy-h)
        x1 = random.randint(0,w-zx)
        out_images = list()
        from torchvision import transforms
        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            img1 = img.crop((x1,y1,x1+zx,y1+zy))
            crop = transforms.CenterCrop(self.size)
            out_images.append(crop(img1))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size) #scale_size ；是一个方阵.

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupColorJitter(object):
    def __init__(self, brightness=0,contrast=0,saturation=0,hue=0):
        self.worker = torchvision.transforms.ColorJitter(brightness,contrast,saturation,hue)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScaleCV2(object):
    """ Using Opencv to rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    using opencv, then image should be transfer to PIL again
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        if not isinstance(size, int):
            raise TypeError('Got inappropriate size arg: {}'.format(size))
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        cv2_img_group = [cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) for pil_image in img_group]
        resized_cv2_img_group =[self._resize_cv2(img) for img in cv2_img_group]
        pil_img_group = [Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) for img in resized_cv2_img_group]

        return pil_img_group

    def _resize_cv2(self, img):
        h, w = img.shape[:-1]
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return cv2.resize(img, (ow, oh), self.interpolation)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return cv2.resize(img, (ow, oh), self.interpolation)



class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    lib: using opencv or PIL to resize, if using opencv, then image should be transfer to PIL again
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object): # 5张crop，5张crop加翻转）,corner crop，而且是4个corner和1个center
    def __init__(self, crop_size, scale_size=None):
        """
        :param crop_size: crop image to 224(e.g.)
        :param scale_size: scale image to shorter size 256(e.g.)
        """
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None


    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group
"""
首先__init__中的GroupScale类也是在transforms.py中定义的，其实是对输入的n张图像都做torchvision.transforms.Scale操作，也就是resize到指定尺寸。GroupMultiScaleCrop.fill_fix_offset返回的offsets是一个长度为5的列表，每个值都是一个tuple，其中前4个是四个点坐标，最后一个是中心点坐标，目的是以这5个点为左上角坐标时可以在原图的四个角和中心部分crop出指定尺寸的图，后面有例子介绍。crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))是按照crop_w*crop_h的大小去crop原图像，这里采用的是224*224。flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)是对crop得到的图像做左右翻转。最后把未翻转的和翻转后的列表合并，这样一张输入图像就可以得到10张输出了（5张crop，5张crop加翻转）。举个例子，假设image_w=340，image_h=256，crop_w=224，crop_h=224，那么offsets就是[(0,0),(116,0),(0,32),(116,32),(58,16)]，因此第一个crop的结果就是原图上左上角坐标为(0,0)，右下角坐标为(224,224)的图，这也就是原图的左上角部分图；第二个crop的结果就是原图上左上角坐标为(116,0)，右下角坐标为(340,224)的图，这也就是原图的右上角部分图，其他依次类推分别是原图的左下角部分图和右下角部分图，最后一个是原图正中央crop出来的224*224图。这就是论文中说的corner crop，而且是4个corner和1个center。
"""
class GroupRandomResizeCrop(object):
    """
    random resize image to shorter size = [256,320] (e.g.),
    and random crop image to 224[e.g.]
    p.s.: if input size > 224, resize_range should be enlarged in equal proportion
    """
    def __init__(self, resize_range, input_size, interpolation=Image.BILINEAR):
        self.resize_range = resize_range
        self.crop_worker = GroupRandomCrop(input_size)
        self.interpolation = interpolation

    def __call__(self, img_group):
        resize_size = random.randint(self.resize_range[0],self.resize_range[1])
        resize_worker = GroupScale(resize_size)
        resized_img_group = resize_worker(img_group)
        crop_img_group = self.crop_worker(resized_img_group)

        return crop_img_group



class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


if __name__ == "__main__":
    trans = torchvision.transforms.Compose([
        GroupScale(256),
        GroupRandomCrop(224),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )]
    )

    im = Image.open('../tensorflow-model-zoo.torch/lena_299.png')

    color_group = [im] * 3
    rst = trans(color_group)

    gray_group = [im.convert('L')] * 9
    gray_rst = trans(gray_group)

    trans2 = torchvision.transforms.Compose([
        GroupRandomSizedCrop(256),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225])
    ])
    print(trans2(color_group))