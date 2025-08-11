import torch
import torch.utils.data as data
import os
import numpy as np
import random
import logging
import json
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

MASK = -1
EPS = 1e-8
PAD_TOKEN = "[PAD]"


def concat_entity_attribute(name, attributes):
    if not attributes or (len(attributes) == 1 and attributes[0] == ""):
        return name
    attr_text = " ".join([attr for attr in attributes if attr])
    return f"{name} is {attr_text}" if attr_text else name


class PrecompRegionDataset(data.Dataset):
    def __init__(self, data_path, data_split, tokenizer, opt, train):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train

        if 'coco' in opt.dataset:
            data_base = os.path.join(data_path, 'coco')
        else:
            data_base = os.path.join(data_path, 'f30k')
        loc = os.path.join(data_base,"precomp")
        loc_1 = os.path.join(loc, "factual")
        loc_mapping = os.path.join(data_base, 'id_mapping.json')


        # Raw captions
        self.captions = []
        with open(os.path.join(loc, '%s_caps.txt' % data_split), 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())
        #图像生成文本的bge编码特征
        self.image_cap_embeddings = []
        if data_split == 'train':
            # train_caps_synthesis_florence_bge_det.npy
            embeddings = np.load(os.path.join(loc_1, f'{data_split}_caps_synthesis_florence_bge_det.npy'))
            self.image_cap_embeddings = embeddings
        else:
            # test_caps_synthesis_florence_bge_det.npy
            embeddings = np.load(os.path.join(loc_1, f'{data_split}_caps_synthesis_florence_bge_det.npy'))
            self.image_cap_embeddings = embeddings
        #原始文本的bge编码特征
        self.cap_bge_embs = []
        if data_split == 'train':
            # train_caps_bge_det.npy
            embeddings = np.load(os.path.join(loc_1, f'{data_split}_caps_bge_det.npy'))
            self.cap_bge_embs = embeddings
        else:
            # test_caps_bge_det.npy
            embeddings = np.load(os.path.join(loc_1, f'{data_split}_caps_bge_det.npy'))
            self.cap_bge_embs = embeddings
        #图像标签的entities bge编码
        # self.img_cap_entity_embeddings = []
        # if data_split == 'train':
        #     # train_caps_synthesis_florence_entities_cls_bge.npy
        #     embeddings = np.load(os.path.join(loc_1, f'{data_split}_caps_synthesis_florence_entities_bge.npy'))
        #     self.img_cap_entity_embeddings = embeddings
        # else:
        #     # test_caps_synthesis_florence_entities_cls_bge.npy
        #     embeddings = np.load(os.path.join(loc_1, f'{data_split}_caps_synthesis_florence_entities_bge.npy'))
        #     self.img_cap_entity_embeddings = embeddings
        #原始文本的entities bge编码
        self.text_entity_embeddings = []
        if data_split == 'train':
            # train_caps_entities_bge.npy
            embeddings = np.load(os.path.join(loc_1, f'{data_split}_caps_entities_bge.npy'))
            self.text_entity_embeddings = embeddings
        else:
            # test_caps_entities_bge.npy
            embeddings = np.load(os.path.join(loc_1, f'{data_split}_caps_entities_bge.npy'))
            self.text_entity_embeddings = embeddings

        #图像标签生成场景图的关系矩阵
        # if data_split == 'train':
        #     # train_caps_synthesis_florence_relation_matrices_cls_bge.npy
        #     self.img_cap_relation_matrices = np.load(
        #         os.path.join(loc_1,  f'{data_split}_caps_synthesis_florence_relation_matrices_cls_bge.npy'))
        # else:
        #     # test_caps_synthesis_florence_relation_matrices_cls_bge.npy
        #     self.img_cap_relation_matrices = np.load(
        #         os.path.join(loc_1,  f'{data_split}_caps_synthesis_florence_relation_matrices_cls_bge.npy'))

        #原始文本生成场景图的关系矩阵
        if data_split == 'train':
            # train_caps_relation_matrices_cls_bge.npy
            self.text_relation_matrices = np.load(os.path.join(loc_1,  f'{data_split}_caps_relation_matrices_cls_bge.npy'))
        else:
            #test_caps_relation_matrices_cls_bge.npy
            self.text_relation_matrices = np.load(os.path.join(loc_1,  f'{data_split}_caps_relation_matrices_cls_bge.npy'))

        # Region features
        self.images = np.load(os.path.join(loc, '%s_ims.npy' % data_split))

        # num_captions
        self.length = len(self.captions)
        self.num_images = len(self.images)

        if self.num_images != self.length:
            # one images to five captions (train set)
            self.im_div = 5
        else:
            # one images to one captions (test set)
            self.im_div = 1

        if data_split == 'dev':
            self.length = 5000


    def __getitem__(self, index):

        # handle the image redundancy
        # index for captions, img_index for images
        img_index = index // self.im_div
        caption = self.captions[index]
        image_cap_embeddings = self.image_cap_embeddings[index]
        bge_cap_embs = self.cap_bge_embs[index]
        # 13，1024
        # img_cap_entity_embeddings = self.img_cap_entity_embeddings[index]
        # 16，1024
        text_entity_embeddings = self.text_entity_embeddings[index]

        # img_cap_relation_matrix = self.img_cap_relation_matrices[index]  # 图像标签的关系矩阵
        text_relation_matrix = self.text_relation_matrices[index]

        image_cap_embeddings = torch.Tensor(image_cap_embeddings)
        bge_cap_embs = torch.Tensor(bge_cap_embs)
        # img_cap_entity_embeddings = torch.Tensor(img_cap_entity_embeddings)
        text_entity_embeddings = torch.Tensor(text_entity_embeddings)

        # img_cap_relation_matrix = torch.Tensor(img_cap_relation_matrix)
        text_relation_matrix = torch.Tensor(text_relation_matrix)

        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        target = process_caption_bert(self.tokenizer, caption_tokens, self.train)
        image = torch.Tensor(self.images[img_index])

        if self.train and self.opt.size_augment:
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > 0.20)]

        # target:文本caption经过分词以后的
        return (image, target, image_cap_embeddings, bge_cap_embs, text_entity_embeddings, text_relation_matrix, index, img_index)

    def __len__(self):
        return self.length


def process_caption_bert(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        # text -> token (basic_tokenizer.tokenize) -> sub_token (wordpiece_tokenizer.tokenize)
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)

        prob = random.random()

        # first, 20% probability use the augmenation operations
        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token from the BERT-vocab
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> 40% delete the token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    # record the index of sub_token
                    deleted_idx.append(len(output_tokens) - 1)
        # 80% probability keep the token
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    # and first and last notations for BERT model
    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']

    # Convert token to vocabulary indices, torch.float32
    target = tokenizer.convert_tokens_to_ids(output_tokens)

    # convert to the torch-tenfor 
    target = torch.Tensor(target)
    return target


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, image_cap_embeddings, bge_cap_embs, text_entity_embeddings,text_relation_matrix, ids, img_ids = zip(*data)

    # 处理基本的image_cap_embeddings（128，1024）
    image_cap_embeddings = torch.stack(image_cap_embeddings, 0)

    # 处理基本的bge_cap_embs（128，1024）
    bge_cap_embs = torch.stack(bge_cap_embs, 0)

    # 处理img_cap_entity_embeddings
    # batch_size = len(img_cap_entity_embeddings)
    # hidden_size = img_cap_entity_embeddings[0].size(1)

    # 获取每个样本中非零实体的实际数量
    # img_cap_entity_lengths = [len(torch.nonzero(emb.sum(dim=1))) for emb in img_cap_entity_embeddings]
    # max_img_cap_entities = max(img_cap_entity_lengths)
    #
    # # 创建新的tensor来存储填充后的实体embeddings
    # padded_img_cap_entities = torch.zeros(batch_size, max_img_cap_entities, hidden_size)
    # for i, emb in enumerate(img_cap_entity_embeddings):
    #     actual_entities = img_cap_entity_lengths[i]
    #     padded_img_cap_entities[i, :actual_entities] = emb[:actual_entities]

        # 处理img_cap_relation_matrix
        # 创建新的tensor来存储填充后的关系矩阵
    # padded_img_cap_relations = torch.zeros(batch_size, max_img_cap_entities, max_img_cap_entities)
    # for i, rel_matrix in enumerate(img_cap_relation_matrix):
    #     actual_entities = img_cap_entity_lengths[i]
    #     padded_img_cap_relations[i, :actual_entities, :actual_entities] = rel_matrix[:actual_entities, :actual_entities]

    # 处理img_cap_relation_matrix
    # padded_img_cap_relations = torch.zeros(batch_size, max_img_cap_entities, max_img_cap_entities)
    # for i, rel_matrix in enumerate(img_cap_relation_matrix):
    #     actual_entities = img_cap_entity_lengths[i]
    #     # 将非零值改为1
    #     rel_matrix_binary = (rel_matrix != 0).float()
    #     # 设置对角线为1，添加自环性
    #     for j in range(actual_entities):
    #         rel_matrix_binary[j, j] = 1.0
    #     padded_img_cap_relations[i, :actual_entities, :actual_entities] = rel_matrix_binary[:actual_entities, :actual_entities]


    # 处理text_entity_embeddings
    # 获取每个样本中非零文本实体的实际数量
    batch_size = len(text_entity_embeddings)
    hidden_size = text_entity_embeddings[0].size(1)
    text_entity_lengths = [len(torch.nonzero(emb.sum(dim=1))) for emb in text_entity_embeddings]
    max_text_entities = max(text_entity_lengths)

    # 创建新的tensor来存储填充后的文本实体embeddings
    padded_text_entities = torch.zeros(batch_size, max_text_entities, hidden_size)
    for i, emb in enumerate(text_entity_embeddings):
        actual_entities = text_entity_lengths[i]
        padded_text_entities[i, :actual_entities] = emb[:actual_entities]

    # 处理text_relation_matrix
    padded_text_relations = torch.zeros(batch_size, max_text_entities, max_text_entities)
    for i, rel_matrix in enumerate(text_relation_matrix):
        actual_entities = text_entity_lengths[i]
        # 将非零值改为1
        rel_matrix_binary = (rel_matrix != 0).float()
        for j in range(actual_entities):
            rel_matrix_binary[j, j] = 1.0
        padded_text_relations[i, :actual_entities, :actual_entities] = rel_matrix_binary[:actual_entities, :actual_entities]

    img_ids = torch.tensor(img_ids)
    ids = torch.tensor(ids)

    repeat = len(img_ids) - len(torch.unique(img_ids))

    # 处理图像特征
    img_lengths = [len(image) for image in images]
    all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
    for i, image in enumerate(images):
        end = img_lengths[i]
        all_images[i, :end] = image[:end]

    img_lengths = torch.tensor(img_lengths)

    # 处理文本caption
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = torch.tensor(lengths)

    # img_cap_entity_lengths = torch.tensor(img_cap_entity_lengths)
    text_entity_lengths = torch.tensor(text_entity_lengths)

    # all_images: Batch_size * max_img_lengths * 2048 (the dimension of region-features)
    # targets:  Batch_size * max_cap_lengths
    return (all_images, img_lengths, targets, image_cap_embeddings, bge_cap_embs, padded_text_entities,
             text_entity_lengths, padded_text_relations, lengths, ids, img_ids, repeat)


def get_loader(data_path, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    drop_last = True if train else False

    dset = PrecompRegionDataset(data_path, data_split, tokenizer, opt, train)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              drop_last=drop_last)

    return data_loader


def get_train_loader(data_path, tokenizer, batch_size, workers, opt):
    train_loader = get_loader(data_path, 'train', tokenizer, opt, batch_size, True, workers, train=True)

    return train_loader


def get_test_loader(data_path, split_name, tokenizer, batch_size, workers, opt):
    test_loader = get_loader(data_path, split_name, tokenizer, opt, batch_size, False, workers,
                             train=False)

    return test_loader


if __name__ == '__main__':
    pass
