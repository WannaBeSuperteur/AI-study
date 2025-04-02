
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from handle_dataset.main import get_category_list


# GLASS 모델 가져오기
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - model (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델

def get_glass_model():
    raise NotImplementedError


# 학습, 검증 및 테스트 데이터셋 정의
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - category_name (str) : 카테고리 이름

# Returns:
# - train_dataset (Dataset) : 해당 카테고리의 학습 데이터셋
# - valid_dataset (Dataset) : 해당 카테고리의 검증 데이터셋
# - test_dataset  (Dataset) : 해당 카테고리의 테스트 데이터셋

def get_datasets(category_name):
    raise NotImplementedError


# 모델 학습 실시
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - model         (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델
# - train_dataset (Dataset)   : 학습 데이터셋 (카테고리 별)
# - valid_dataset (Dataset)   : 검증 데이터셋 (카테고리 별)

# Returns:
# - val_loss_list (list) : Valid Loss 기록

def run_train(model, train_dataset, valid_dataset):
    raise NotImplementedError


# 모델 테스트 실시
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - model        (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델
# - test_dataset (Dataset)   : 테스트 데이터셋 (카테고리 별)

# Returns:
# - test_result      (dict)             : 테스트 성능 평가 결과
#                                         {'accuracy': float, 'recall': float, 'precision': float, 'f1_score': float}
# - confusion_matrix (Pandas DataFrame) : 테스트 성능 평가 시 생성된 Confusion Matrix

def run_test(model, test_dataset):
    raise NotImplementedError


if __name__ == '__main__':
    glass_model = get_glass_model()
    category_list = get_category_list()

    for category_name in category_list:
        train_dataset, valid_dataset, test_dataset = get_datasets(category_name)
        run_train(glass_model, train_dataset, valid_dataset)

        test_result, confusion_matrix = run_test(glass_model, test_dataset)

        print(f'\n==== TEST RESULT ({category_name}) ====')
        print(test_result)
        print(confusion_matrix)