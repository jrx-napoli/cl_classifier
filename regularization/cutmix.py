import torch
import numpy as np


def cutmix_images(x, y, alpha=1.0):
    """
    Sourced from https://github.com/drimpossible/GDumb/blob/master/src/utils.py
    """

    assert (alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_repr(x, y, alpha=1.0):
    """
    Cutmix implementation for latent samples
    """

    assert (alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]

    bbx1, bbx2 = rand_vec(x.size(), lam)
    x[:, bbx1:bbx2] = x[index, bbx1:bbx2]

    # adjust lambda to exactly match cutout ratio
    lam = 1 - ((bbx2 - bbx1) / (x.size()[-1]))
    return x, y_a, y_b, lam

def rand_vec(size, lam):
    W = size[1]
    cut_rat = 1. - lam
    cut_w = np.int(W * cut_rat)

    cx = np.random.randint(W)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)

    return bbx1, bbx2
