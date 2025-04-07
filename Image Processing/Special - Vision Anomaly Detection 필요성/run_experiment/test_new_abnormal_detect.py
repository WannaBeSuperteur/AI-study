
import warnings
warnings.filterwarnings('ignore')

try:
    from common import (get_datasets,
                        run_train_tinyvit,
                        run_test_tinyvit,
                        set_fixed_seed,
                        load_tinyvit_trained_model,
                        run_tinyvit_explanation)
except:
    from run_experiment.common import (get_datasets,
                                       run_train_tinyvit,
                                       run_test_tinyvit,
                                       set_fixed_seed,
                                       load_tinyvit_trained_model,
                                       run_tinyvit_explanation)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.tinyvit import get_model as get_tinyvit_model
from models.pytorch_grad_cam import get_xai_model

from torch.utils.data import DataLoader


PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
TEST_BATCH_SIZE = 4


# Anomaly Detection 데이터셋 분류 모델 (정상 vs. Sample 이 가장 많은 종류의 Anomaly) 학습 및 테스트
# Create Date : 2025.04.07
# Last Update Date : -

# args :
# - category_name (str) : 카테고리 이름

# returns :
# - 정상 vs. Sample 이 가장 많은 종류의 Anomaly 를 구분하는 Classification Model 학습 및 저장

def train_and_test_tinyvit(category_name):
    tinyvit_model = get_tinyvit_model()
    print('model load finished')

    train_dataset_tinyvit, valid_dataset_tinyvit, test_dataset_tinyvit = (
        get_datasets(category_name,
                     dataset_dir_name='mvtec_dataset_exp3_classify',
                     img_size=512,
                     model_name='TinyViT',
                     experiment_no=3))

    val_accuracy_list, loss_tinyvit, val_auroc_list = run_train_tinyvit(model=tinyvit_model,
                                                                        train_dataset=train_dataset_tinyvit,
                                                                        valid_dataset=valid_dataset_tinyvit,
                                                                        category=category_name,
                                                                        experiment_no=3)

    test_result_tinyvit, confusion_matrix_tinyvit = run_test_tinyvit(test_dataset=test_dataset_tinyvit,
                                                                     category=category_name,
                                                                     experiment_no=3)

    print(f'\n==== TRAIN RESULT ({category_name}) of TinyViT ====')
    for idx, loss in enumerate(loss_tinyvit):
        print(f'epoch {idx + 1} : loss = {loss}')

    print(f'\n==== TEST RESULT ({category_name}) of TinyViT ====')
    print(test_result_tinyvit)
    print(confusion_matrix_tinyvit)


# 학습한 Anomaly Detection 데이터셋 분류 모델 (정상 vs. Sample 이 가장 많은 종류의 Anomaly) 에 대한 XAI 실시
# Create Date : 2025.04.07
# Last Update Date : -

# args :
# - category_name (str) : 카테고리 이름

# returns :
# - 정상 vs. Sample 이 가장 많은 종류의 Anomaly 를 구분하는 Classification Model 에
#   XAI (PyTorch Grad-CAM) 를 실행했을 때의 Overlay Image 결과 저장

def run_xai_on_trained_tinyvit(category_name):

    # load trained TinyViT model
    tinyvit_with_softmax = load_tinyvit_trained_model('exp3', category_name)

    layer_name_map = {
        'stage2_conv1': tinyvit_with_softmax.tinyvit_model.stages[1].downsample.conv1.conv,
        'stage2_conv2': tinyvit_with_softmax.tinyvit_model.stages[1].downsample.conv2.conv,
        'stage2_conv3': tinyvit_with_softmax.tinyvit_model.stages[1].downsample.conv3.conv,
        'stage3_conv1': tinyvit_with_softmax.tinyvit_model.stages[2].downsample.conv1.conv,
        'stage3_conv2': tinyvit_with_softmax.tinyvit_model.stages[2].downsample.conv2.conv,
        'stage3_conv3': tinyvit_with_softmax.tinyvit_model.stages[2].downsample.conv3.conv,
        'stage4_conv1': tinyvit_with_softmax.tinyvit_model.stages[3].downsample.conv1.conv,
        'stage4_conv2': tinyvit_with_softmax.tinyvit_model.stages[3].downsample.conv2.conv,
        'stage4_conv3': tinyvit_with_softmax.tinyvit_model.stages[3].downsample.conv3.conv
    }

    for layer_name, target_layer in layer_name_map.items():
        target_layers = [target_layer]

        # load test dataset
        _, valid_dataset_tinyvit, test_dataset_tinyvit = get_datasets(category_name,
                                                                      dataset_dir_name='mvtec_dataset_exp3_classify',
                                                                      img_size=512,
                                                                      model_name='TinyViT',
                                                                      experiment_no=3)

        xai_model = get_xai_model(tinyvit_with_softmax, target_layers)

        # run TinyViT explanation on VALID dataset
        valid_loader = DataLoader(valid_dataset_tinyvit, batch_size=TEST_BATCH_SIZE, shuffle=False)

        run_tinyvit_explanation(xai_model,
                                valid_loader,
                                category_name,
                                layer_name,
                                experiment_no=3,
                                overlay_img_dir_name='overlay_valid')

        # run TinyViT explanation on TEST dataset
        test_loader = DataLoader(test_dataset_tinyvit, batch_size=TEST_BATCH_SIZE, shuffle=False)

        run_tinyvit_explanation(xai_model,
                                test_loader,
                                category_name,
                                layer_name,
                                experiment_no=3,
                                overlay_img_dir_name='overlay_test')


if __name__ == '__main__':
    category_list = ['bottle', 'hazelnut', 'carpet', 'grid']  # get_category_list()

    for category_name in category_list:
        set_fixed_seed(2025)

        # 정상 vs. Sample 이 가장 많은 종류의 Anomaly 를 구분하는 Classification Model 학습 및 저장
        train_and_test_tinyvit(category_name)

        # 학습된 Classification Model 에 대해 XAI (PyTorch Grad-CAM) 적용
        run_xai_on_trained_tinyvit(category_name)
