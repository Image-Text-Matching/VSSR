import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LOCAL_MODEL_PATH = './bge-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(device)
model.eval()


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


def encode_sentences(sentences, batch_size=32):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
            token_embeddings = model_output[0]

        sentence_embeddings = pooling(token_embeddings, encoded_input['attention_mask'], pooling_method='mean')
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        embeddings.append(sentence_embeddings)

    return torch.cat(embeddings, dim=0)


def process_file(input_file, output_file, encoding_batch_size=1):
    print(f"Processing {input_file}...")
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [s.strip() for s in sentences]

    all_embeddings = []

    for i in tqdm(range(0, len(sentences), encoding_batch_size), desc="Encoding sentences"):
        batch_sentences = sentences[i:i + encoding_batch_size]
        embeddings = encode_sentences(batch_sentences, batch_size=encoding_batch_size)
        all_embeddings.append(embeddings.cpu().numpy())

    # Combine all embeddings into a single numpy array
    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Shape of combined embeddings: {combined_embeddings.shape}")

    # Save the combined embeddings
    np.save(output_file, combined_embeddings)
    print(f"Saved embeddings to {output_file}")

    # Verify the saved file
    loaded_embeddings = np.load(output_file)
    print(f"Shape of loaded embeddings: {loaded_embeddings.shape}")

    print(f"Finished processing {input_file}")


# 文件列表
files = [
        # ('D:\study\FACTUAL/f30k/train_caps_synthesis_florence_det.txt',
        #  'D:\study\FACTUAL/f30k/train_caps_synthesis_florence_bge_det.npy'),
        # ('D:\study\FACTUAL/f30k/test_caps_synthesis_florence_det.txt',
        #  'D:\study\FACTUAL/f30k/test_caps_synthesis_florence_bge_det.npy'),
        ('D:\study\FACTUAL/f30k/test_caps.txt',
         'D:\study\FACTUAL/f30k/test_caps_bge_det.npy'),
        ('D:\study\FACTUAL/f30k/train_caps.txt',
        'D:\study\FACTUAL/f30k/train_caps_bge_det.npy'),
]

# 处理所有文件
for input_file, output_file in files:
    process_file(input_file, output_file)

print("All files processed successfully!")
