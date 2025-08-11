import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import json
import numpy as np


def concat_entity_attribute(name, attributes):
    if not attributes or (len(attributes) == 1 and attributes[0] == ""):
        return name
    attr_text = " ".join([attr for attr in attributes if attr])
    return f"{name} is {attr_text}" if attr_text else name


def load_scene_graphs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_entities(scene_graphs):
    entities_list = []
    for sg in scene_graphs:
        entities = []
        for entity in sg['entities']:
            entity_text = concat_entity_attribute(entity['name'], entity['attributes'])
            if entity_text:  # 只有当entity_text非空时才添加
                entities.append(entity_text)
        entities_list.append(entities)
    return entities_list


def process_relation_matrices(scene_graphs):
    """处理每个场景图中实体间的关系矩阵，构建对称的关系矩阵"""
    relation_matrices_list = []

    for sg in scene_graphs:
        # 创建原始索引到新索引的映射
        valid_entity_mapping = {}
        new_idx = 0
        for orig_idx, entity in enumerate(sg['entities']):
            if entity['name']:  # 只处理有效实体
                valid_entity_mapping[orig_idx] = new_idx
                new_idx += 1

        # 获取有效实体数量
        num_entities = len(valid_entity_mapping)

        # 创建关系标签矩阵，初始值为0（表示无关系）
        relation_matrix = np.zeros((num_entities, num_entities), dtype=int)

        # 处理关系
        if 'relations' in sg:
            for idx, rel in enumerate(sg['relations'], 1):  # 从1开始编号
                subject_orig_idx = rel['subject']
                object_orig_idx = rel['object']

                # 只处理有效实体之间的关系
                if (subject_orig_idx in valid_entity_mapping and
                        object_orig_idx in valid_entity_mapping):
                    # 使用映射后的新索引
                    subject_new_idx = valid_entity_mapping[subject_orig_idx]
                    object_new_idx = valid_entity_mapping[object_orig_idx]

                    # 将关系标记为对应的编号，同时标记反向关系
                    relation_matrix[subject_new_idx][object_new_idx] = idx
                    relation_matrix[object_new_idx][subject_new_idx] = idx  # 添加反向关系

        relation_matrices_list.append(relation_matrix)

    return relation_matrices_list


def pad_entities(entities_list, PAD_TOKEN="[PAD]"):
    max_entities = max(len(entities) for entities in entities_list)
    padded_entities = []
    entity_lengths = []

    for entities in entities_list:
        length = len(entities)
        entity_lengths.append(length)
        padded = entities + [PAD_TOKEN] * (max_entities - length)
        padded_entities.append(padded)

    return padded_entities, entity_lengths, max_entities


def pad_relation_matrices(relation_matrices_list):
    """对关系矩阵进行填充"""
    max_entities = max(matrix.shape[0] for matrix in relation_matrices_list)
    padded_matrices = []

    for matrix in relation_matrices_list:
        current_size = matrix.shape[0]
        if current_size < max_entities:
            # 创建填充后的矩阵
            padded_matrix = np.zeros((max_entities, max_entities), dtype=int)
            # 将原始矩阵复制到左上角
            padded_matrix[:current_size, :current_size] = matrix
            padded_matrices.append(padded_matrix)
        else:
            padded_matrices.append(matrix)

    return padded_matrices


def pooling(token_embeddings, attention_mask, pooling_method='cls'):
    if pooling_method == 'cls':
        return token_embeddings[:, 0]
    elif pooling_method == 'mean':
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    else:
        raise ValueError("Pooling method must be either 'cls' or 'mean'")


def encode_entity_batch(entities_batch, model, tokenizer, device, PAD_TOKEN="[PAD]"):
    embeddings = []
    for entities in entities_batch:
        valid_entities = [e for e in entities if e != PAD_TOKEN]
        if valid_entities:
            encoded_input = tokenizer(valid_entities, padding=True, truncation=True,
                                    return_tensors='pt').to(device)

            with torch.no_grad():
                model_output = model(**encoded_input)
                entity_embeddings = model_output[0][:, 0, :]
                entity_embeddings = entity_embeddings.cpu()

            padding_size = len(entities) - len(valid_entities)
            if padding_size > 0:
                pad_embeddings = torch.zeros(padding_size, entity_embeddings.size(1))
                entity_embeddings = torch.cat([entity_embeddings, pad_embeddings], dim=0)
        else:
            embed_dim = model.config.hidden_size
            entity_embeddings = torch.zeros(len(entities), embed_dim)

        embeddings.append(entity_embeddings.numpy())

    return embeddings


def process_file(input_file, output_file, model, tokenizer, device, batch_size=100, PAD_TOKEN="[PAD]"):
    print(f"Processing {input_file}...")
    output_dir = os.path.dirname(output_file)
    temp_dir = os.path.join(output_dir, 'temp_embeddings')
    os.makedirs(temp_dir, exist_ok=True)

    # 加载场景图并处理实体和关系矩阵
    scene_graphs = load_scene_graphs(input_file)
    entities_list = process_entities(scene_graphs)
    relation_matrices_list = process_relation_matrices(scene_graphs)

    # 填充实体和关系矩阵
    padded_entities, entity_lengths, max_entities = pad_entities(entities_list, PAD_TOKEN)
    padded_relation_matrices = pad_relation_matrices(relation_matrices_list)

    total_samples = len(padded_entities)
    print(f"Total samples to process: {total_samples}")

    # 处理和保存
    print("Processing and saving in batches...")
    temp_files_entities = []

    for i in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, total_samples)

        # 处理实体
        batch_entities = padded_entities[i:batch_end]
        batch_embeddings_entities = encode_entity_batch(batch_entities, model, tokenizer, device, PAD_TOKEN)

        # 保存当前批次
        batch_array_entities = np.stack(batch_embeddings_entities)
        temp_file_entities = os.path.join(temp_dir, f'batch_entities_{i}.npy')
        np.save(temp_file_entities, batch_array_entities)
        temp_files_entities.append(temp_file_entities)

        # 清理内存
        del batch_embeddings_entities, batch_array_entities
        torch.cuda.empty_cache()

    # 合并所有临时文件
    print("Merging temporary files...")
    final_embeddings_entities = []

    for temp_file in tqdm(temp_files_entities, desc="Merging"):
        batch_data_entities = np.load(temp_file)
        final_embeddings_entities.append(batch_data_entities)

    combined_embeddings_entities = np.concatenate(final_embeddings_entities, axis=0)

    # 将关系矩阵转换为numpy数组
    relation_matrices_array = np.stack(padded_relation_matrices)

    # 保存最终结果
    entities_output_file = output_file
    relations_output_file = output_file.replace('entities', 'relation_matrices')

    print(f"Saving final embeddings and relation matrices...")
    np.save(entities_output_file, combined_embeddings_entities)
    np.save(relations_output_file, relation_matrices_array)

    # 清理临时文件
    print("Cleaning up temporary files...")
    for temp_file in temp_files_entities:
        os.remove(temp_file)
    os.rmdir(temp_dir)

    # 验证最终文件
    print("Verifying saved files...")
    loaded_embeddings_entities = np.load(entities_output_file, mmap_mode='r')
    loaded_relation_matrices = np.load(relations_output_file, mmap_mode='r')
    print(f"Shape of loaded entities embeddings: {loaded_embeddings_entities.shape}")
    print(f"Shape of loaded relation matrices: {loaded_relation_matrices.shape}")
    print("Verification successful!")


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LOCAL_MODEL_PATH = './bge-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(device)
model.eval()

# 文件列表
files = [
    ('D:\study\FACTUAL/f30k/train_caps.json',
     'D:\study\FACTUAL/f30k/train_caps_entities_cls_bge.npy'),
    ('D:\study\FACTUAL/f30k/test_caps.json',
     'D:\study\FACTUAL/f30k/test_caps_entities_cls_bge.npy'),
    ('D:\study\FACTUAL/f30k/train_caps_synthesis_florence_det.json',
     'D:\study\FACTUAL/f30k/train_caps_synthesis_florence_entities_cls_bge.npy'),
    ('D:\study\FACTUAL/f30k/test_caps_synthesis_florence_det.json',
     'D:\study\FACTUAL/f30k/test_caps_synthesis_florence_entities_cls_bge.npy'),
]

# 处理所有文件
for input_file, output_file in files:
    process_file(input_file, output_file, model, tokenizer, device)

print("All files processed successfully!")