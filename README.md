# VLMs bridging-enhanced Scene Semantic Reasoning Framework for Image-Text Matching

![Static Badge](https://img.shields.io/badge/Pytorch-EE4C2C)
![License: MIT](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)

The codes for our paper "VLMs bridging-enhanced Scene Semantic Reasoning Framework for Image-Text Matching(VSSR)", ,which is accepted by the ICMR2025. We referred to the implementations of [GPO](https://github.com/woodfrog/vse_infty) , [HREM](https://github.com/CrossmodalGroup/HREM)and [FACTUAL](https://github.com/zhuang-li/FactualSceneGraph) to build up our codes. We express our gratitude for these outstanding works.

## Introduction

The main challenge in image-text matching lies in bridging the gap between visual and linguistic modalities for accurate cross-modal semantic alignment. While current mainstream methods enhance local feature interactions through region-word attention mechanisms, their isolated object modeling paradigm fails to capture deep semantic relationships, limiting fine-grained cross-modal reasoning capabilities. Research has explored structured relationship models like scene graphs; however, the visual modality faces inherent limitations: unlike text, which can build relationship graphs through lexical logic, visual scenes lack clear contextual semantics, resulting in issues such as ambiguous entity boundaries and distorted relationships that impede effective cross-modal alignment. This paper proposes a VLMs bridging-enhanced Scene Semantic Reasoning framework (VSSR). Based on modal characteristic differences, we construct a dual-path scene parsing framework: On the visual modeling, leveraging the strong semantic understanding capabilities of Vision-Language Models (VLMs) to generate dense scene semantic labels; On the textual modeling, fully exploiting the structured advantages of language by designing a graph attention network guided by relationship inductive bias to deeply mine implicit semantic associations between textual entities. To further bridge the modality gap, we create a multimodal collaborative representation space, using scene semantic labels as anchors to bridge the two modalities and achieve cross-modal knowledge transfer through joint semantic projection. While maintaining linear computational complexity, this architecture realizes fine-grained matching from scene-level (image-caption) to entity-level (image-entity) through relation-aware semantic modeling. Experiments on the Flickr30K and MS-COCO benchmark datasets demonstrate that VSSR outperforms existing state-of-the-art approaches in retrieval performance.

![overview](https://github.com/Image-Text-Matching/VSSR/blob/main/overview.png)


## Training

Train MSCOCO and Flickr30K from scratch:

```
python train.py \
    --batch_size 128 \
    --data_path <path_to_your_dataset> \
    --dataset f30k \
    --loss_type trip \
    --coding_type VHACoding \
    --pooling_type MaxPooling \
    --logger_name <path_to_save_logs> \
    --num_epochs 25
```

```
python train.py \
    --batch_size 256 \ 
    --data_path <path_to_your_dataset> \
    --dataset coco \
    --loss_type trip \
    --coding_type VHACoding \
    --pooling_type MaxPooling \
    --logger_name <path_to_save_logs> \
    --num_epochs 25
```

## Evaluation

Modify the corresponding parameters in eval.py to test the Flickr30K or MSCOCO data set:

```
python eval.py  --dataset f30k  --data_path "path/to/dataset"
```

```
python eval.py  --dataset coco --data_path "path/to/dataset"
```

##  Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and a citation 📝:

```
@inproceedings{gao2025VSSR,
  title={VLMs bridging-enhanced Scene Semantic Reasoning Framework for Image-Text Matching},
  author={Gao, Yihua and Chen, Junyu and Li, Mingyong},
  booktitle={Proceedings of the 2025 International Conference on Multimedia Retrieval},
  pages={330--339},
  year={2025}
}
```
