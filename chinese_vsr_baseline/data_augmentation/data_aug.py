import numpy as np
import torch
import cv2

from torchvision import transforms

cv2.setNumThreads(0)


def CenterCrop(imgs: np.ndarray, crop_size: tuple):
    assert imgs.ndim == 4
    N, C, H, W = imgs.shape
    th, tw = crop_size
    assert H >= th and W >= tw, f"Invalid crop_size: {crop_size}"
    x1 = int(round(W - tw) / 2.)
    y1 = int(round(H - th) / 2.)
    return imgs[:, :, y1: y1 + th, x1: x1 + tw]


def RandomCrop(imgs: np.ndarray, crop_size: tuple):
    assert imgs.ndim == 4
    th, tw = crop_size
    N, C, H, W = imgs.shape
    max_x, max_y = W - tw, H - th
    if max_x == 0 or max_y == 0:
        return imgs
    assert max_x > 0 and max_y > 0, f"Invalid crop_size: {crop_size}"
    x1, y1 = torch.randint(0, max_x, (1,)).item(), np.random.randint(0, max_y, (1,)).item()
    return imgs[:, :, y1: y1 + th, x1: x1 + th]


def DropFrame(imgs: np.ndarray, drop_p: float):
    """
    Drop some frames. The dropped frames will be filled with the mean image.

    Args:
        imgs:
        drop_p:

    Returns:

    """
    assert imgs.ndim == 4
    assert 0. <= drop_p <= 1.
    N, C, H, W = imgs.shape
    mean_img = imgs.mean(0)

    new_imgs = []
    idxs_left = []
    for i in range(N):
        p = np.random.uniform(0., 1.)
        if p > drop_p:
            new_imgs.append(imgs[i])
            idxs_left.append(i)
        elif p <= drop_p:
            new_imgs.append(mean_img)

    return np.array(new_imgs, dtype=np.float32)


def HorizontalFlip(imgs: np.ndarray, p: float):
    assert imgs.ndim == 4
    return imgs[:, :, :, ::-1] if np.random.rand() < p else imgs


def ColorNormalize(imgs: np.ndarray):
    imgs = imgs.astype(np.float32)
    imgs /= 255.
    m, d = 0.421, 0.165
    return (imgs - m) / d


def bgr2gray(imgs: np.ndarray):
    assert imgs.ndim == 4
    N, C, H, W = imgs.shape                 
    imgs = imgs.transpose(0, 2, 3, 1)      
    ret_imgs = np.zeros((N, H, W), dtype=np.float32)
    for i in range(N):
        ret_imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
    return ret_imgs


def DataAugumentation(imgs, is_training, crop_size, p, drop_frm, Tmask_p, resize=(96, 96)):
    """

    Args:
        imgs:
        is_training:
        crop_size:
        p:
        drop_frm:
        Tmask_p:
        resize:

    Returns:

    """
    Crop = RandomCrop if is_training else CenterCrop
    Flip = HorizontalFlip if is_training else lambda x, p: x     

    # 112 -> 96 
    n, c, h, w = imgs.shape
    hh, ww = resize
    imgs = CenterCrop(imgs, resize)

    if drop_frm:
        imgs = DropFrame(imgs, drop_p=0.1)

    if is_training and Tmask_p > 0.:
        imgs = TemporalMasking(imgs)

    
    imgs = Crop(imgs, crop_size)

    imgs = Flip(imgs, p)
    
        
    imgs = bgr2gray(imgs) if imgs.shape[1] == 3 else imgs[:, [0]]   # gray or rgb
    imgs = ColorNormalize(imgs)
    
    imgs = imgs.astype(np.float32)

    return imgs
