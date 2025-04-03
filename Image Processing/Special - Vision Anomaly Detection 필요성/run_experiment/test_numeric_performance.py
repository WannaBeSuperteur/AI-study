
try:
    from common import get_datasets, run_train_glass, run_train_tinyvit, run_test_glass, run_test_tinyvit
except:
    from run_experiment.common import get_datasets, run_train_glass, run_train_tinyvit, run_test_glass, run_test_tinyvit

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from handle_dataset.main import get_category_list
from models.glass import get_model as get_glass_model
from models.tinyvit import get_model as get_tinyvit_model

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


if __name__ == '__main__':
    category_list = ['bottle', 'hazelnut', 'carpet', 'grid']  # get_category_list()

    for category_name in category_list:
        glass_model = get_glass_model()
        tinyvit_model = get_tinyvit_model()
        print('model load finished')

        train_dataset_glass, valid_dataset_glass, test_dataset_glass = (
            get_datasets(category_name,
                         dataset_dir_name='mvtec_dataset_exp1_anomaly',
                         img_size=256,
                         model_name='GLASS',
                         experiment_no=1))

        train_dataset_tinyvit, valid_dataset_tinyvit, test_dataset_tinyvit = (
            get_datasets(category_name,
                         dataset_dir_name='mvtec_dataset_exp1_classify',
                         img_size=512,
                         model_name='TinyViT',
                         experiment_no=1))

        """
        loss_glass = run_train_glass(model=glass_model,
                                     train_dataset=train_dataset_glass,
                                     valid_dataset=valid_dataset_glass,
                                     category=category_name,
                                     experiment_no=1)

        loss_tinyvit = run_train_tinyvit(model=tinyvit_model,
                                         train_dataset=train_dataset_tinyvit,
                                         valid_dataset=valid_dataset_tinyvit,
                                         category=category_name,
                                         experiment_no=1)
        """

        test_result_glass, confusion_matrix_glass = run_test_glass(test_dataset=test_dataset_glass,
                                                                   category=category_name,
                                                                   experiment_no=1)

        test_result_tinyvit, confusion_matrix_tinyvit = run_test_tinyvit(test_dataset=test_dataset_tinyvit,
                                                                         category=category_name,
                                                                         experiment_no=1)

        print(f'\n==== TRAIN RESULT ({category_name}) of GLASS ====')
        for idx, loss in enumerate(loss_glass):
            print(f'epoch {idx + 1} : loss = {loss}')

        print(f'\n==== TRAIN RESULT ({category_name}) of TinyViT ====')
        for idx, loss in enumerate(loss_tinyvit):
            print(f'epoch {idx + 1} : loss = {loss}')

        print(f'\n==== TEST RESULT ({category_name}) of GLASS ====')
        print(test_result_glass)
        print(confusion_matrix_glass)

        print(f'\n==== TEST RESULT ({category_name}) of TinyViT ====')
        print(test_result_tinyvit)
        print(confusion_matrix_tinyvit)
