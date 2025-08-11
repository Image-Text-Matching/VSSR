from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Tuple
from torch.cuda.amp import autocast
import gc

MASK = -1
EPS = 1e-8


def get_mask(lens):
    max_len = int(lens.max())
    mask = torch.arange(max_len, device=lens.device).expand(lens.size(0), max_len)
    mask = (mask < lens.long().unsqueeze(dim=1)).float()
    return mask.unsqueeze(-1)


def get_pooling(pooling_type, **args):
    belta = args["opt"].belta
    if pooling_type == "MeanPooling":
        return MeanPooling()
    elif pooling_type == "LSEPooling":
        return LSEPooling(belta)
    if pooling_type == "MaxPooling":
        return MaxPooling()
    else:
        raise ValueError("Unknown pooling type: {}".format(pooling_type))


def get_coding(coding_type, **args):
    alpha = args["opt"].alpha
    if coding_type == "VHACoding":
        return VHACoding()
    elif coding_type == "THACoding":
        return THACoding()
    elif coding_type == "VSACoding":
        return VSACoding(alpha)
    else:
        raise ValueError("Unknown coding type: {}".format(coding_type))


class LSEPooling(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        sims[sims == -1] = -torch.inf
        sims = torch.logsumexp(sims / self.temperature, dim=-1)
        return sims

class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        sims = sims.max(dim=-1)[0]
        return sims
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape) == 3
        lens = (sims != MASK).sum(dim=-1)
        sims[sims == MASK] = 0
        sims = sims.sum(dim=-1) / lens
        return sims


class VHACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r, max_w = int(img_lens.max()), int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:, :, :max_r, :max_w]
        mask = get_fgmask(img_lens, cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-2)[0]
        return sims


class THACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r, max_w = int(img_lens.max()), int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:, :, :max_r, :max_w]
        mask = get_fgmask(img_lens, cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-1)[0]
        return sims


class VSACoding(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        max_r, max_w = int(img_lens.max()), int(cap_lens.max())
        imgs = imgs[:, :max_r, :]
        caps = caps[:, :max_w, :]
        sims = get_fgsims(imgs, caps)[:, :, :max_r, :max_w]
        mask = get_fgmask(img_lens, cap_lens)

        sims = sims / self.temperature
        sims = torch.softmax(sims.masked_fill(mask == 0, -torch.inf), dim=-1)
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims, caps)
        sims = torch.mul(sims.permute(1, 0, 2, 3), imgs).permute(1, 0, 2, 3).sum(dim=-1) \
               / (torch.norm(sims, p=2, dim=-1, keepdim=False) + EPS)

        mask = get_mask(img_lens).permute(0, 2, 1).repeat(1, cap_lens.size(0), 1)
        sims = sims.masked_fill(mask == 0, -1)
        return sims


def concat_entity_attribute(name, attributes):
    if not attributes or (len(attributes) == 1 and attributes[0] == ""):
        return name
    attr_text = " ".join([attr for attr in attributes if attr])
    return f"{name} is {attr_text}" if attr_text else name


class EntityEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./bge-large-en-v1.5")
        self.model = AutoModel.from_pretrained("./bge-large-en-v1.5")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def batch_encode(self, texts):
        with autocast():
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            outputs = self.model(**encoded_input)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)

            return embeddings


def get_batch_data(data: List, batch_size: int, start_idx: int) -> Tuple[List, List[int]]:
    end_idx = min(start_idx + batch_size, len(data))
    batch_data = data[start_idx:end_idx]

    batch_entities = []
    entity_lengths = []

    for item in batch_data:
        entity_texts = []
        for entity in item['entities']:
            entity_text = concat_entity_attribute(entity['name'], entity['attributes'])
            entity_texts.append(entity_text)
        batch_entities.append(entity_texts)
        entity_lengths.append(len(entity_texts))

    max_entities_in_batch = max(entity_lengths)

    padded_batch_entities = []
    for entities in batch_entities:
        padded = entities + ['[PAD]'] * (max_entities_in_batch - len(entities))
        padded_batch_entities.append(padded)

    if len(padded_batch_entities) < batch_size:
        padding_needed = batch_size - len(padded_batch_entities)
        empty_entity = ['[PAD]'] * max_entities_in_batch
        padded_batch_entities.extend([empty_entity] * padding_needed)
        entity_lengths.extend([0] * padding_needed)

    return padded_batch_entities, entity_lengths


def process_batch(encoder: EntityEncoder, batch_entities: List[List[str]], batch_lengths: List[int],
                  device: torch.device) -> torch.Tensor:
    batch_size = len(batch_entities)
    max_entities = len(batch_entities[0])

    batch_embeddings = torch.zeros((batch_size, max_entities, encoder.embed_dim), device=device)

    for i, (entities, length) in enumerate(zip(batch_entities, batch_lengths)):
        if length > 0:
            valid_entities = entities[:length]
            embeddings = encoder.batch_encode(valid_entities)
            batch_embeddings[i, :length] = embeddings

    return batch_embeddings


@torch.no_grad()
def get_fgsims(imgs: torch.Tensor, caps: torch.Tensor) -> torch.Tensor:
    bi, n_r, embi = imgs.shape
    bc, n_w, embc = caps.shape

    imgs = imgs.reshape(bi * n_r, embi)
    caps = caps.reshape(bc * n_w, embc).t()

    sims = torch.matmul(imgs, caps)
    sims = sims.reshape(bi, n_r, bc, n_w).permute(0, 2, 1, 3)

    return sims


def get_fgmask(img_lens: torch.Tensor, cap_lens: torch.Tensor) -> torch.Tensor:
    bi = img_lens.shape[0]
    bc = cap_lens.shape[0]
    max_r = int(img_lens.max())
    max_w = int(cap_lens.max())

    mask_i = torch.arange(max_r, device=img_lens.device).expand(bi, max_r)
    mask_i = (mask_i < img_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1)
    mask_i = mask_i.reshape(bi * max_r, 1)

    mask_c = torch.arange(max_w, device=cap_lens.device).expand(bc, max_w)
    mask_c = (mask_c < cap_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1)
    mask_c = mask_c.reshape(bc * max_w, 1).t()

    mask = torch.matmul(mask_i, mask_c).reshape(bi, max_r, bc, max_w).permute(0, 2, 1, 3)
    return mask


@torch.no_grad()
def compute_similarity_matrix(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
                              image_lengths: List[int], text_lengths: List[int],
                              pooling_type: str = 'mean', coding_type: str = 'VHACoding',
                              temperature: float = 0.1, device: torch.device = None) -> torch.Tensor:
    with autocast():
        img_lens = torch.tensor(image_lengths, device=device)
        cap_lens = torch.tensor(text_lengths, device=device)

        # 使用选择的编码方式
        if coding_type == 'VHACoding':
            coding = VHACoding()
        elif coding_type == 'THACoding':
            coding = THACoding()
        elif coding_type == 'VSACoding':
            coding = VSACoding(temperature)

        sims = coding(image_embeddings, text_embeddings, img_lens, cap_lens)

        # 使用选择的池化方式
        if pooling_type == 'mean':
            pooling = MeanPooling()
        elif pooling_type == 'LSE':
            pooling = LSEPooling(temperature)
        else:
            pooling = MaxPooling()

        similarity = pooling(sims)

        return similarity

def main():
    batch_size = 128
    synthesis_file = "D:\study\FACTUAL\scene_graph/test_caps_synthesis_florence_det_extract_ea.json"
    extract_file = "D:\study\FACTUAL\scene_graph/test_caps_extract_ea.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取数据
    with open(synthesis_file, 'r', encoding='utf-8') as f:
        synthesis_data = json.load(f)
    with open(extract_file, 'r', encoding='utf-8') as f:
        extract_data = json.load(f)

    total_samples = len(synthesis_data)
    encoder = EntityEncoder()

    # 初始化最终的相似度矩阵
    full_similarity_matrix = torch.zeros((total_samples, total_samples), device=device)

    # 批处理循环
    for i in range(0, total_samples, batch_size):
        print(f"Processing batch {i // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}")

        # 获取synthesis批次数据
        synthesis_batch, synthesis_lengths = get_batch_data(synthesis_data, batch_size, i)
        synthesis_embeddings = process_batch(encoder, synthesis_batch, synthesis_lengths, device)

        # 对于每个synthesis批次，我们需要与所有extract批次计算相似度
        for j in range(0, total_samples, batch_size):
            extract_batch, extract_lengths = get_batch_data(extract_data, batch_size, j)
            extract_embeddings = process_batch(encoder, extract_batch, extract_lengths, device)

            # 计算当前批次的相似度
            similarity = compute_similarity_matrix(
                synthesis_embeddings,
                extract_embeddings,
                synthesis_lengths,
                extract_lengths,
                pooling_type='MaxPooling',
                coding_type='VHACoding',
                device=device
            )
        #最好的：120-LSEPooling /VHACoding
            # 将结果填入完整矩阵的对应位置
            rows = min(batch_size, total_samples - i)
            cols = min(batch_size, total_samples - j)
            full_similarity_matrix[i:i + rows, j:j + cols] = similarity[:rows, :cols]

            # 清理GPU缓存
            torch.cuda.empty_cache()
            gc.collect()

    # 保存结果
    np_filename = "similarity_matrix_5000x5000_MaxPooling_VHACoding.npy"
    np.save(np_filename, full_similarity_matrix.cpu().numpy())
    print(f"Similarity matrix saved to {np_filename}")
    print(f"Matrix shape: {full_similarity_matrix.shape}")
    print(f"Sample of similarity matrix:")
    print(full_similarity_matrix[:5, :5].cpu().numpy())  # 打印一个5x5的样本


if __name__ == "__main__":
    main()

