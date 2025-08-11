import arguments
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder, SimsEncoder, l2norm, get_entities_encoder
from lib.loss import *

logger = logging.getLogger(__name__)
PAD_TOKEN = "[PAD]"


class VSEModel(object):
    def __init__(self, opt, eval=False):

        self.opt = opt
        self.grad_clip = opt.grad_clip

        self.img_enc = get_image_encoder(opt, opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt, opt.embed_size, no_txtnorm=opt.no_txtnorm)
        self.entities_enc = get_entities_encoder(n_layers=1, embed_size=1024)

        self.sim_enc = SimsEncoder(coding_type=opt.coding_type, pooling_type=opt.pooling_type, opt=opt)
        self.criterion = loss_select(opt, loss_type=opt.loss_type)
        self.e_criterion = loss_select(opt, 'vse', margin=0.15)

        self.ett_criterion = loss_select(opt, 'vse')  # 普通的三元组损失（CHAN）
        # self.proto_criterion = ProtoContrastiveLoss(opt)

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4

        self.params = list(self.txt_enc.parameters()) + list(self.img_enc.parameters()) + list(
            self.sim_enc.parameters()) + list(self.entities_enc.parameters())

        all_text_params = list(self.txt_enc.parameters())
        bert_params = list(self.txt_enc.bert.parameters())
        sim_enc_params = list(self.sim_enc.parameters())
        entities_enc_params = list(self.entities_enc.parameters())

        # Tensor.data_ptr() → int, Returns the address of the first element
        bert_params_ptr = [p.data_ptr() for p in bert_params]
        text_params_no_bert = list()

        # select other parameters except BERT
        for p in all_text_params:
            if p.data_ptr() not in bert_params_ptr:
                text_params_no_bert.append(p)

        self.optimizer = torch.optim.AdamW([
            {'params': text_params_no_bert, 'lr': opt.learning_rate},
            {'params': bert_params, 'lr': opt.learning_rate * 0.05},
            {'params': entities_enc_params, 'lr': opt.entities_learing_rate * 0.1},
            {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
        ],
            lr=opt.learning_rate, weight_decay=decay_factor)

        # iteration
        self.Eiters = 0
        self.data_parallel = False

        # use the gpu
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.entities_enc.cuda()
            torch.backends.cudnn.benchmark = True

    def set_max_violation(self, max_violation=True):
        if self.opt.loss_type == 'vse':
            if max_violation:
                self.criterion.max_violation_on()
            else:
                self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [
            self.img_enc.state_dict(),
            self.txt_enc.state_dict(),
            self.entities_enc.state_dict(),
            self.sim_enc.state_dict(),
        ]
        return state_dict

    def load_state_dict(self, state_dict, ):
        # strict=True, ensure keys match
        self.img_enc.load_state_dict(state_dict[0], strict=True)

        # Unexpected key(s) in state_dict: "bert.embeddings.position_ids". 
        # incompatible problem of transformers package version 
        self.txt_enc.load_state_dict(state_dict[1], strict=False)
        self.entities_enc.load_state_dict(state_dict[2], strict=False)
        self.sim_enc.load_state_dict(state_dict[3], strict=False)

    def train_start(self):
        self.img_enc.train()
        self.txt_enc.train()
        self.entities_enc.train()
        self.sim_enc.train()

    def val_start(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.entities_enc.eval()
        self.sim_enc.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.entities_enc = nn.DataParallel(self.entities_enc)
        self.data_parallel = True
        logger.info('Image/Text encoder is data paralleled (use multi GPUs).')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    # Compute the image and caption embeddings
    def forward_emb(self, images, captions, image_cap_embeddings, bge_cap_embs,
                    padded_text_entities,text_entity_lengths, padded_text_relations, lengths, image_lengths=None):

        # padded_img_cap_entities = padded_img_cap_entities.cuda()
        padded_text_entities = padded_text_entities.cuda()
        # img_cap_entity_lengths = img_cap_entity_lengths.cuda()
        text_entity_lengths = text_entity_lengths.cuda()


        # padded_img_cap_entities = self.entities_enc(padded_img_cap_entities, padded_img_cap_relations)
        padded_text_entities = self.entities_enc(padded_text_entities, padded_text_relations)

        # img_emb:128,1024
        images = images.cuda()
        image_lengths = image_lengths.cuda()
        image_cap_embeddings = image_cap_embeddings.cuda()
        img_emb = self.img_enc(images, image_cap_embeddings, image_lengths)

        # cap_emb:128,1024
        captions = captions.cuda()
        lengths = lengths.cuda()
        bge_cap_embs = bge_cap_embs.cuda()
        cap_emb, text_entities_emb = self.txt_enc(captions, bge_cap_embs, padded_text_entities,
                                                  lengths, text_entity_lengths)

        return img_emb, cap_emb,  text_entities_emb

    def forward_sim(self, img_emb, cap_emb, img_len, cap_len):
        sims = self.sim_enc(img_emb, cap_emb, img_len, cap_len)
        return sims

    def forward_loss(self, img_emb, cap_emb, img_len, cap_len):
        """Compute the loss given pairs of image and caption embeddings
        """
        sims = self.forward_sim(img_emb, cap_emb, img_len, cap_len)
        loss = self.ett_criterion(sims)
        # self.logger.update('Le', loss.data.item(), sims.size(0))
        return loss

    # One training step given images and captions
    def train_emb(self, images, captions, image_cap_embeddings, bge_cap_embs,
                  padded_text_entities,text_entity_lengths, padded_text_relations, opt, lengths, image_lengths=None,
                  img_ids=None):

        self.Eiters += 1
        self.logger.update('Iter', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # padded_img_cap_entities = padded_img_cap_entities.cuda()
        padded_text_entities = padded_text_entities.cuda()
        # img_cap_entity_lengths = img_cap_entity_lengths.cuda()
        text_entity_lengths = text_entity_lengths.cuda()

        # padded_img_cap_entities = self.entities_enc(padded_img_cap_entities, padded_img_cap_relations)
        padded_text_entities = self.entities_enc(padded_text_entities, padded_text_relations)

        # img_emb:128,1024
        images = images.cuda()
        image_lengths = image_lengths.cuda()
        image_cap_embeddings = image_cap_embeddings.cuda()
        img_emb = self.img_enc(images, image_cap_embeddings, image_lengths,  graph=True)

        # cap_emb:128,1024
        captions = captions.cuda()
        lengths = lengths.cuda()
        bge_cap_embs = bge_cap_embs.cuda()
        cap_emb, text_entities_emb = self.txt_enc(captions, bge_cap_embs, padded_text_entities,
                                                  lengths, text_entity_lengths, graph=True)


        bs, n_entities, d = padded_text_entities.shape

        # 创建掩码
        entity_valid_mask = torch.zeros((bs, n_entities), dtype=torch.bool, device=padded_text_entities.device)
        for i in range(bs):
            entity_valid_mask[i, :text_entity_lengths[i]] = True

        # 展平实体编码
        flat_entities = padded_text_entities.view(-1, d)
        # 展平掩码
        flat_mask = entity_valid_mask.view(-1)
        # 只保留有效的实体编码
        flat_entities = flat_entities[flat_mask]

        # 创建图像索引
        ent_img_ids = torch.arange(bs, device=img_emb.device).unsqueeze(1).expand(-1, n_entities)
        # 展平并只保留有效实体对应的图像索引
        ent_img_ids = ent_img_ids.reshape(-1)[flat_mask]

        # compute loss
        self.optimizer.zero_grad()
        loss_entities = self.e_criterion(img_emb, text_entities_emb)
        loss_caption_entities = supcon_joint_caption_entity_loss(img_emb, cap_emb, flat_entities, ent_img_ids,temperature=0.01)
        loss_specificity = triplet_specificity_loss(img_emb, cap_emb, flat_entities, ent_img_ids,margin=0.15)
        loss = self.criterion(img_emb, cap_emb, img_ids=img_ids) + loss_entities + loss_caption_entities['total_loss']+ loss_specificity

        print('Loss_entities', loss_entities.item())
        print('loss_caption_entities', loss_caption_entities['total_loss'].item())
        print('loss_specificity', loss_specificity.item())
        print('Loss_g', loss.item() - loss_entities.item() - loss_caption_entities['total_loss'].item() -loss_specificity.item())
        print('Loss', loss.item())
        # self.logger.update('Loss_entities', loss_entities.item(), self.opt.batch_size)
        self.logger.update('Loss', loss.item(), self.opt.batch_size)
        # self.logger.update('loss_entities', loss_entities.item(), self.opt.batch_size)

        # compute gradient and update
        if torch.isnan(loss):
            logger.error("We have NaN numbers, ")
            return 0.

        loss.backward()
        # loss_entities.backward()

        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()


if __name__ == '__main__':
    pass
