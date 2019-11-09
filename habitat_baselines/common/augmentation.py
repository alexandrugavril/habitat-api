""" credits to
    https://github.com/amdegroot/ssd.pytorch/edit/master/utils/augmentations.py
"""
import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from torch.nn import functional as F


def rgb_to_hsv(rgb):
    rgb = rgb.float()
    input_shape = rgb.size()
    rgb2 = rgb.view(-1, 3)
    r, g, b = rgb2[:, 0], rgb2[:, 1], rgb2[:, 2]

    maxc = torch.max(rgb2, dim=1)[0] #np.maximum(np.maximum(r, g), b)
    minc = torch.min(rgb2, dim=1)[0] #np.minimum(np.minimum(r, g), b)
    v = maxc

    deltac = maxc - minc
    s = deltac / maxc
    deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = 4.0 + gc - rc
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0

    h *= 179.
    s *= 255.
    res = torch.stack([h, s, v.float()], dim=1) #np.dstack([h, s, v])

    return res.view(input_shape)


def hsv_to_rgb(hsv):
    input_shape = hsv.size()

    hsv2 = hsv.view(-1, 3)
    h, s, v = hsv2[:, 0], hsv2[:, 1], hsv2[:, 2]
    h /= 179.
    s /= 255.

    i = (h * 6.0).int()
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = torch.zeros_like(hsv2)
    v, t, p, q = v.view(-1, 1), t.view(-1, 1), p.view(-1, 1), q.view(-1, 1)
    rgb[i == 0] = torch.cat([v, t, p], dim=1)[i == 0]
    rgb[i == 1] = torch.cat([q, v, p], dim=1)[i == 1]
    rgb[i == 2] = torch.cat([p, v, t], dim=1)[i == 2]
    rgb[i == 3] = torch.cat([p, q, v], dim=1)[i == 3]
    rgb[i == 4] = torch.cat([t, p, v], dim=1)[i == 4]
    rgb[i == 5] = torch.cat([v, p, q], dim=1)[i == 5]
    rgb[s == 0.0] = torch.cat([v, v, v], dim=1)[s == 0.0]

    return rgb.view(input_shape)


# img = cv2.imread("1.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# hsv = rgb_to_hsv(torch.from_numpy(img))
# rgb = hsv_to_rgb(hsv)
#
# bgr = cv2.cvtColor(rgb.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
# cv2.imwrite("test.jpg", bgr)


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class ConvertColor(object):
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = rgb_to_hsv(image)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = hsv_to_rgb(image)
        else:
            raise NotImplementedError
        return image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class Expand(object):
    def __init__(self, min_scale, max_scale, mode="bilinear"):
        self.min = min_scale
        self.max = max_scale
        self.mode = mode

    def __call__(self, image,):
        if random.randint(2):
            return image

        ratio = random.uniform(self.min, self.max)

        image = image.permute(2, 0, 1).unsqueeze(0)

        image = F.upsample(image, scale_factor=ratio, mode=self.mode)
        image = image.squeeze(0).permute(1, 2, 0)
        return image


class RandomMirror(object):
    def __call__(self, image):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
        return image


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image):
        im = image.clone()
        im = self.rand_brightness(im)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        return self.rand_light_noise(im)


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        pass

    def __call__(self, image, cwidth, cheight):
        height, width, _ = image.shape
        cheight, cwidth = cheight, cwidth

        left = random.uniform(width - cwidth)
        top = random.uniform(height - cheight)

        # convert to integer rect x1,y1,x2,y2
        rect = np.array([int(left), int(top), int(left+cwidth),
                         int(top+cheight)])

        current_image = image[rect[1]:rect[3], rect[0]:rect[2], :]
        return current_image


class RandomMove(object):
    def __init__(self, min_scale, max_scale, mode="bilinear"):
        self.crop = RandomSampleCrop()
        self.expand = Expand(min_scale, max_scale, mode=mode)

    def __call__(self, image):
        if random.randint(2):
            height, width, _ = image.shape
            image = self.expand(image)
            image = self.crop(image, width, height)
        return image


class AIMASAug(object):
    def __init__(self, width, height, min_scale, max_scale):
        self.augment = Compose([
            PhotometricDistort(),
            RandomMove(min_scale, max_scale),
            # RandomSampleCrop(),
            # RandomMirror(),
            # ToPercentCoords(),
            # Resize(self.size),
            # SubtractMeans(self.mean)
        ])

    def __call__(self, img):
        return self.augment(img)


if __name__ == "__main__":

    img = cv2.imread("1.jpg")
    height, width, _ = img.shape
    aug = AIMASAug(width, height, 1., 1.2)

    cv2.imshow("ORIGINAL", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb = torch.from_numpy(img).float()

    while True:
        new_img = aug(rgb)
        new_img.clamp_(0, 255)
        bgr = cv2.cvtColor(new_img.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        print("New image")
        cv2.imshow("test", bgr)
        cv2.waitKey(0)



