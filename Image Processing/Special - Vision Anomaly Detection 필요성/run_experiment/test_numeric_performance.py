
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


if __name__ == '__main__':
    category_list = get_category_list()

    for category_name in category_list:
        glass_model = get_glass_model()
        tinyvit_model = get_tinyvit_model()
        print('model load finished')

        train_dataset, valid_dataset, test_dataset = get_datasets(category_name)

        run_train_glass(glass_model, train_dataset, valid_dataset)
        run_train_tinyvit(tinyvit_model, train_dataset, valid_dataset)

        test_result_glass, confusion_matrix_glass = run_test_glass(glass_model, test_dataset)
        test_result_tinyvit, confusion_matrix_tinyvit = run_test_tinyvit(tinyvit_model, test_dataset)

        print(f'\n==== TEST RESULT ({category_name}) of GLASS ====')
        print(test_result_glass)
        print(confusion_matrix_glass)

        print(f'\n==== TEST RESULT ({category_name}) of TinyViT ====')
        print(test_result_tinyvit)
        print(confusion_matrix_tinyvit)
