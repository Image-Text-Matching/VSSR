# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    # norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    # X = torch.div(X, norm)
    X = X / (torch.norm(X,p=1,dim=dim,keepdim=True)+eps)
    return X

# calculate the mask according to the given lens
def get_mask(lens):
    """
    :param lens: length of the sequence
    :return: 
    """
    batch = lens.shape[0]
    max_l = int(lens.max())
    mask = torch.arange(max_l).expand(batch, max_l).to(lens.device)
    mask = (mask<lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(lens.device)
    return mask

# calculate the fine-grained similarity according to the given images and captions
def get_fgsims(imgs, caps):
    bi, n_r, embi = imgs.shape
    bc, n_w, embc = caps.shape
    imgs = imgs.reshape(bi*n_r, embi)
    caps = caps.reshape(bc*n_w, embc).t()
    sims = torch.matmul(imgs,caps)
    sims = sims.reshape(bi, n_r, bc, n_w).permute(0,2,1,3)
    return sims


# calculate the mask of fine-grained similarity according to the given images length and captions length
def get_fgmask(img_lens, cap_lens):
    bi = img_lens.shape[0]
    bc = cap_lens.shape[0]
    max_r = int(img_lens.max())
    max_w = int(cap_lens.max())

    mask_i = torch.arange(max_r).expand(bi, max_r).to(img_lens.device)
    mask_i = (mask_i<img_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(img_lens.device)
    mask_i = mask_i.reshape(bi*max_r,1)

    mask_c = torch.arange(max_w).expand(bc,max_w).to(cap_lens.device)
    mask_c = (mask_c<cap_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(cap_lens.device)
    mask_c = mask_c.reshape(bc*max_w,1).t()

    mask = torch.matmul(mask_i,mask_c).reshape(bi, max_r, bc, max_w).permute(0,2,1,3)
    return mask
