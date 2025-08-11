import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def pos_neg_mask(labels):
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    return pos_mask, neg_mask


def pos_neg_mask_xy(labels_col, labels_row):
    pos_mask = (labels_row.unsqueeze(0) == labels_col.unsqueeze(1))
    neg_mask = (labels_row.unsqueeze(0) != labels_col.unsqueeze(1))

    return pos_mask, neg_mask


def loss_select(opt, loss_type='vse', margin=0.2):
    if loss_type == 'vse':
        # the default loss
        criterion = ContrastiveLoss(opt=opt, margin=0.05, max_violation=opt.max_violation)
    elif loss_type == 'trip':
        # Triplet loss with the distance-weight sampling
        criterion = TripletLoss(opt=opt, margin=margin)
    else:
        raise ValueError('Invalid loss {}'.format(loss_type))

    return criterion


# class ContrastiveLoss(nn.Module):
#
#     def __init__(self, opt, margin=0.2, max_violation=False):
#         super(ContrastiveLoss, self).__init__()
#         self.opt = opt
#         self.margin = margin
#         self.max_violation = max_violation
#         self.mask_repeat = opt.mask_repeat
#
#         self.false_hard = []
#
#     def max_violation_on(self):
#         self.max_violation = True
#         # print('Use VSE++ objective.')
#
#     def max_violation_off(self):
#         self.max_violation = False
#         # print('Use VSE0 objective.')
#
#     # def forward(self, im, s, img_ids=None):
#     def forward(self, scores, img_ids=None):
#
#         # compute image-sentence score matrix
#         # scores = get_sim(im, s)
#
#         diagonal = scores.diag().view(scores.size(0), 1)
#         d1 = diagonal.expand_as(scores)
#         d2 = diagonal.t().expand_as(scores)
#
#         # compare every diagonal score to scores in its column
#         # caption retrieval, i->t
#         cost_s = (self.margin + scores - d1).clamp(min=0)
#
#         # compare every diagonal score to scores in its row
#         # image retrieval t->i
#         cost_im = (self.margin + scores - d2).clamp(min=0)
#
#         # clear diagonals
#         if not self.mask_repeat:
#             mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
#         else:
#             img_ids = img_ids
#             mask = (img_ids.unsqueeze(1) == img_ids.unsqueeze(0))
#
#         cost_s = cost_s.masked_fill_(mask, 0)
#         cost_im = cost_im.masked_fill_(mask, 0)
#
#         # keep the maximum violating negative for each query
#         if self.max_violation:
#             cost_s, idx_s = cost_s.max(1)
#             cost_im, idx_im = cost_im.max(0)
#
#         loss = cost_s.sum() + cost_im.sum()
#
#         return loss


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt=None, margin=0.05, max_violation=True):
        super(ContrastiveLoss, self).__init__()
        if opt is not None:
            self.opt = opt
            self.margin = margin
            self.max_violation =max_violation
        else:
            self.margin = margin
            self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s,sims=None):
        # compute image-sentence score matrix
        # sims = get_sim(im, s)
        if sims==None:
            sims = get_sim(im, s)
        else:
            sims=sims
        diagonal = sims.diag().view(sims.size(0), 1)
        d1 = diagonal.expand_as(sims)
        d2 = diagonal.t().expand_as(sims)

        # compare every diagonal score to sims in its column
        # caption retrieval
        cost_s = (self.margin + sims - d1).clamp(min=0)
        # compare every diagonal score to sims in its row
        # image retrieval
        cost_im = (self.margin + sims - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(sims.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


# Triplet loss + DistanceWeight Miner
# Sampling Matters in Deep Embedding Learning, ICCV, 2017
# more information refer to https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer
class TripletLoss(nn.Module):

    def __init__(self, opt=None, margin=0.2):
        super().__init__()

        self.opt = opt
        self.margin = margin

        self.cut_off = 0.5
        self.d = 512

        if opt.dataset == 'coco':
            self.nonzero_loss_cutoff = 1.9
        else:
            self.nonzero_loss_cutoff = 1.7

    def forward(self, im, s, img_ids):

        sim_mat = get_sim(im, s)
        img_ids = img_ids.cuda()

        if im.size(0) == s.size(0):
            pos_mask, neg_mask = pos_neg_mask(img_ids)
        else:
            pos_mask, neg_mask = pos_neg_mask_xy(torch.unique(img_ids), img_ids)

        loss_im = self.loss_forward(sim_mat, pos_mask, neg_mask)
        loss_s = self.loss_forward(sim_mat.t(), pos_mask.t(), neg_mask.t())

        loss = loss_im + loss_s

        return loss

    def loss_forward(self, sim_mat, pos_mask, neg_mask):

        pos_pair_idx = pos_mask.nonzero(as_tuple=False)
        anchor_idx = pos_pair_idx[:, 0]
        pos_idx = pos_pair_idx[:, 1]

        dist = (2 - 2 * sim_mat).sqrt()
        dist = dist.clamp(min=self.cut_off)

        log_weight = (2.0 - self.d) * dist.log() - ((self.d - 3.0) / 2.0) * (1.0 - 0.25 * (dist * dist)).log()
        inf_or_nan = torch.isinf(log_weight) | torch.isnan(log_weight)

        log_weight = log_weight * neg_mask
        log_weight[inf_or_nan] = 0.

        weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
        weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float()

        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
        weight = weight[anchor_idx]

        # maybe not exist
        try:
            neg_idx = torch.multinomial(weight, 1).squeeze(1)
        except Exception:
            return torch.zeros([], requires_grad=True, device=sim_mat.device)

        s_ap = sim_mat[anchor_idx, pos_idx]
        s_an = sim_mat[anchor_idx, neg_idx]

        loss = F.relu(self.margin + s_an - s_ap)
        loss = loss.sum()

        return loss

    # Source based on: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    import torch

def supcon_joint_caption_entity_loss(
        img_embeds, cap_embeds, ent_embeds, ent_img_ids, temperature=0.05,
        img_ent_pos_mask=None,
        ent_mask=None,
        img_indices=None,
):
    """Computes supervised contrastive loss of image against caption+entity (and vice versa).

    Here we know the number of images and captions are the same.
    Args:
    * img_embeds: image embeddings (n_images, d)
    * cap_embeds: caption embeddings (n_captions, d)
    * ent_embeds: image embeddings (n_entities, d)
    * ent_img_ids: image id of each entity (n_entities,) - to show which image an entity belongs to.
    * temperature: softmax temperature.
    * img_ent_pos_mask: to show which (image, entity) are positive pairs.
    """
    n_images = img_embeds.shape[0]
    n_entities = ent_embeds.shape[0]

    cap_and_ent_embeds = torch.cat((cap_embeds, ent_embeds), dim=0)
    scores = img_embeds @ cap_and_ent_embeds.t() / temperature  # (n_images, n_captions + n_entities)

    # create mask to describe which elements in the scores matrix represent positive pairs
    # we start with an identity matrix
    cap_mask = torch.eye(n_images, device=img_embeds.device)

    # find (image, caption) positive pairs
    if img_indices is not None:
        # There's a chance that an image is sampled twice (or more) in the batch, so we use image index to find
        # these positive pairs.
        same_img_mask = (img_indices.unsqueeze(0) == img_indices.unsqueeze(1)).to(img_embeds.device)
        cap_mask = cap_mask.bool() | same_img_mask

    # find (image, entity) positive pairs
    if ent_mask is None:
        ent_mask = torch.zeros((n_entities, n_images), device=img_embeds.device).bool()
        ent_mask[torch.arange(n_entities), ent_img_ids] = True
        ent_mask = ent_mask.t()  # (n_images, n_entities)
    if img_ent_pos_mask is not None:
        ent_mask = ent_mask | img_ent_pos_mask
    mask = torch.cat((cap_mask, ent_mask), dim=1)  # (n_images, n_captions + n_entities)

    # minus max value for numerical stability
    logits_max, _ = torch.max(scores, dim=1, keepdim=True)
    logits = scores - logits_max.detach()
    exp_logits = torch.exp(logits)

    # Mask away positive pairs. (SupCon paper does this but their PyTorch implementation doesn't).
    # Source: https://github.com/HobbitLong/SupContrast/issues/64.
    exp_logits_masked_positive = exp_logits * (1 - mask.float())
    exp_logits_masked_positive_sum = exp_logits_masked_positive.sum(1, keepdim=True)  # (n_images, 1)

    # this is the denominator in the contrastive loss
    exp_logits_masked_positive_sum = exp_logits_masked_positive_sum + exp_logits + 1e-9

    # compute log likelihood
    log_prob = logits - torch.log(exp_logits_masked_positive_sum)

    # we aggregate log prob of all positive pairs to align the positive pair embeddings close together
    mask_log_prob = mask * log_prob

    mask_sum = mask.sum(1)  # number of positive pairs per image
    mean_log_prob_pos = mask_log_prob.sum(1) / mask_sum  # mean loss per image
    loss1 = -mean_log_prob_pos.mean()  # mean image-to-caption&entity loss

    # Next, break down image to caption loss and image to entity loss (for loss plotting and understanding purpose).
    # image to caption
    image_to_caption_loss = -(mask_log_prob[:, :n_images].sum(1) / mask[:, :n_images].sum(1)).mean()
    # image to entity
    image_to_entity_loss = mask_log_prob[:, n_images:].sum(1)
    num_entities_per_image = mask[:, n_images:].sum(1)
    non_empty_entities_mask = num_entities_per_image > 0
    image_to_entity_loss = -(
            image_to_entity_loss[non_empty_entities_mask] / (num_entities_per_image[non_empty_entities_mask] + 1e-9)
    ).mean()

    out = {
        'image_to_caption_loss': image_to_caption_loss,
        'image_to_entity_loss': image_to_entity_loss,
    }

    # do the same as above, but now this is image retrieval using caption and entities as inputs
    scores = scores.t()
    mask = mask.t()
    logits_max, _ = torch.max(scores, dim=1, keepdim=True)
    logits = scores - logits_max.detach()
    exp_logits = torch.exp(logits)

    exp_logits_masked_positive = exp_logits * (1 - mask.float())
    exp_logits_masked_positive_sum = exp_logits_masked_positive.sum(1, keepdim=True)  # (n_captions + n_entities, 1)
    exp_logits_masked_positive_sum = exp_logits_masked_positive_sum + exp_logits + 1e-6
    log_prob = logits - torch.log(exp_logits_masked_positive_sum)

    mask_log_prob = mask * log_prob

    mask_sum = mask.sum(1)  # number of positive images per caption and entity
    mean_log_prob_pos = mask_log_prob.sum(1) / mask_sum
    loss2 = -mean_log_prob_pos.mean()

    caption_to_image_loss = -mean_log_prob_pos[:n_images].mean()
    entity_to_image_loss = -mean_log_prob_pos[n_images:].mean()

    out.update({
        'total_loss': loss1 + loss2,
        'caption_to_image_loss': caption_to_image_loss,
        'entity_to_image_loss': entity_to_image_loss
    })

    return out


def triplet_specificity_loss(img, cap, ent, ent_img_ids, margin=0.2):
    """Computs triplet loss to enfore specificity between image and caption+entity.

    We want to encourage the model to produce higher score for caption than for entity, since caption is more specific.

    Args:
    * img: image embeddings (n_images, d)
    * cap: caption embeddings (n_captions, d)
    * ent: image embeddings (n_entities, d)
    * ent_img_ids: image id of each entity (n_entities,) - to show which image an entity belongs to.
    """
    n_images = img.shape[0]
    n_entities = ent.shape[0]

    scores_img_to_ent = img @ ent.t()  # (n_images, n_entities)
    scores_img_to_cap = (img * cap).sum(1, keepdim=True)  # (n, 1)

    # mask to describe which elements in scores_img_to_ent are positive pairs
    gt_entity_mask = torch.zeros((n_entities, n_images), device=img.device).bool()
    gt_entity_mask[torch.arange(n_entities), ent_img_ids] = True
    gt_entity_mask = gt_entity_mask.t()  # (n_images, n_entities)

    loss = (scores_img_to_ent + margin - scores_img_to_cap).clamp(min=0)
    loss = loss[gt_entity_mask].mean()

    return loss






if __name__ == '__main__':
    pass
