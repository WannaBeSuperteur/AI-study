# model implementation reference:
# - https://huggingface.co/timm/tiny_vit_21m_512.dist_in22k_ft_in1k


import timm
import os
from torchview import draw_graph


# 모델 그래프를 이미지로 그리기
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - model (nn.Module) : TinyViT-21M-512-distill 모델

# Returns:
# - 해당 TinyViT 모델의 그래프 이미지를 tinyvit.png 로 저장

def save_model_graph_image(model):
    model_graph = draw_graph(model, input_size=(16, 3, 512, 512), depth=3)
    visual_graph = model_graph.visual_graph

    # Model Graph 이미지 저장
    visual_graph.render(format='png')

    # Model Graph 이미지 이름 변경
    file_path = os.path.abspath(os.path.dirname(__file__))
    graph_img_name = f'{file_path}/model.gv.png'
    dest_name = f'{file_path}/GLASS.png'

    os.rename(graph_img_name, dest_name)


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