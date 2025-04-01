import os
from torchview import draw_graph


# 모델 그래프를 이미지로 그리기
# Create Date : 2025.04.01
# Last Update Date : 2025.04.01
# - 모델 그래프 format 을 PNG -> PDF 로 변경

# Arguments:
# - model      (nn.Module) : 모델 (TinyViT-21M-512-distill 또는 GLASS)
# - model_name (str)       : 모델 이름
# - img_size   (int)       : 입력 이미지의 가로/세로 길이

# Returns:
# - 해당 TinyViT 모델의 그래프 이미지를 tinyvit.png 로 저장

def save_model_graph_image(model, model_name, img_size=512):
    model_graph = draw_graph(model, input_size=(16, 3, img_size, img_size), depth=3)
    visual_graph = model_graph.visual_graph

    # Model Graph 이미지 저장
    file_path = os.path.abspath(os.path.dirname(__file__))
    dest_name = f'{file_path}/{model_name}.pdf'

    visual_graph.render(format='pdf', outfile=dest_name)

