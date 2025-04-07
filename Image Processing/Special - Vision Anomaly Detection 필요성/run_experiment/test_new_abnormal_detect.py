
import warnings
warnings.filterwarnings('ignore')

try:
    from common import (get_datasets,
                        run_train_tinyvit,
                        run_test_tinyvit,
                        set_fixed_seed)
except:
    from run_experiment.common import (get_datasets,
                                       run_train_tinyvit,
                                       run_test_tinyvit,
                                       set_fixed_seed)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.tinyvit import get_model as get_tinyvit_model

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


if __name__ == '__main__':
    category_list = ['bottle', 'hazelnut', 'carpet', 'grid']  # get_category_list()

    for category_name in category_list:
        set_fixed_seed(2025)

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