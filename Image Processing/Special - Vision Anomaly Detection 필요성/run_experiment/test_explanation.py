
# GLASS 모델은 exp1 에서 Overlay 출력으로 이미 설명 능력 테스트가 완료된 셈이므로,
# TinyViT 에 대해서만 설명 능력 테스트 실시


import warnings
warnings.filterwarnings('ignore')

try:
    from common import (get_datasets,
                        set_fixed_seed,
                        load_tinyvit_trained_model,
                        run_tinyvit_explanation)
except:
    from run_experiment.common import (get_datasets,
                                       set_fixed_seed,
                                       load_tinyvit_trained_model,
                                       run_tinyvit_explanation)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.pytorch_grad_cam import get_xai_model

import torch
from torch.utils.data import DataLoader


TEST_BATCH_SIZE = 4

# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device for training : {device}')


if __name__ == '__main__':
    category_list = ['bottle', 'hazelnut', 'carpet', 'grid']

    # run TinyViT explanation experiment
    for category_name in category_list:
        set_fixed_seed(2025)

        # load trained TinyViT model
        tinyvit_with_softmax = load_tinyvit_trained_model('exp2', category_name)
        target_layers = [tinyvit_with_softmax.tinyvit_model.stages[3].downsample.conv3.conv]

        # load test dataset
        _, _, test_dataset_tinyvit = get_datasets(category_name,
                                                  dataset_dir_name='mvtec_dataset_exp1_classify',
                                                  img_size=512,
                                                  model_name='TinyViT',
                                                  experiment_no=2)

        xai_model = get_xai_model(tinyvit_with_softmax, target_layers)

        # run TinyViT explanation
        test_loader = DataLoader(test_dataset_tinyvit, batch_size=TEST_BATCH_SIZE, shuffle=False)
        run_tinyvit_explanation(xai_model, test_loader, category_name, experiment_no=2)
