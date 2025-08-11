import os

import torch
import argparse
import logging
from lib import evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='D:\study\FACTUAL', type=str,
                        help='path to datasets')
    parser.add_argument('--dataset', default='f30k', type=str, help='the dataset choice, coco or f30k')
    parser.add_argument('--save_results', type=int, default=1, help='if save the similarity matrix for ensemble')
    parser.add_argument('--coco_ann_path', default='E:/universal_Retrieval/datasets/dataset/coco/captions_val2014.json',
                        type=str, help='path to captions_val2014.json')
    parser.add_argument('--gpu-id', type=int, default=0, help='the gpu-id for evaluation')

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)

    if opt.dataset == 'coco':
        weights_bases = [
            'runs/coco_test_5'
        ]
    else:
        weights_bases = [
            # 'runs/f30k_test'
            'runs/f30k_test_5',
            # 'runs/f30k_test_5',

        ]

    for base in weights_bases:

        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')

        # Save the final similarity matrix
        if opt.save_results:
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            # Evaluate COCO 5K
            evaluation.evalrank(model_path, split='testall', fold5=True, save_path=save_path, data_path=opt.data_path)

        else:
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, split='test', fold5=False, save_path=save_path, data_path=opt.data_path)


if __name__ == '__main__':
    main()

