try:
    from common import save_model_graph_image
    from tinyvit import get_model as get_tinyvit_model
    from gradcam_original_code.grad_cam import GradCAM

except:
    from models.common import save_model_graph_image
    from models.tinyvit import get_model as get_tinyvit_model
    from models.gradcam_original_code.grad_cam import GradCAM


# XAI 모델인 pytorch-grad-cam 모델 가져오기
# Create Date : 2025.04.02
# Last Update Date : 2025.04.07
# - Original Model 이 아닌 GradCAM 을 반환하도록 수정

# Arguments:
# - original_model (nn.Module)       : XAI 의 대상이 되는 원본 모델
# - target_layers  (list(nn.Module)) : XAI 의 대상이 되는 레이어 목록

# Returns:
# - xai_model (nn.Module) : pytorch-grad-cam 모델

def get_xai_model(original_model, target_layers):
    return GradCAM(model=original_model, target_layers=target_layers)


if __name__ == '__main__':
    tinyvit_classifier_model = get_tinyvit_model()
    target_layers = [tinyvit_classifier_model.stages[3].downsample.conv3.conv]

    xai_model = get_xai_model(tinyvit_classifier_model, target_layers)
    print(xai_model)
