# model implementation reference:
# - https://huggingface.co/timm/tiny_vit_21m_512.dist_in22k_ft_in1k


import timm

try:
    from common import save_model_graph_image
except:
    from models.common import save_model_graph_image


# TinyViT-21M-512-distill 모델 가져오기
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - model (nn.Module) : TinyViT-21M-512-distill 모델

def get_model():
    model = timm.create_model('tiny_vit_21m_512.dist_in22k_ft_in1k', pretrained=True, num_classes=2)
    return model


if __name__ == '__main__':
    tinyvit_model = get_model()
    save_model_graph_image(tinyvit_model, model_name='tinyvit', img_size=512)

