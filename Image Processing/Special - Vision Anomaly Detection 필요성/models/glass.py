from common import save_model_graph_image


# GLASS 모델 가져오기
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - model (nn.Module) : GLASS 모델

def get_model():
    # TODO implement
    return model


if __name__ == '__main__':
    GLASS_model = get_model()
    save_model_graph_image(GLASS_model, model_name='GLASS', img_size=512)
