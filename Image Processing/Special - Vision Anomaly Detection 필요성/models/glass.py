try:
    from common import save_model_graph_image
    import glass_original_code.utils as utils
    import glass_original_code.glass as glass
except:
    from models.common import save_model_graph_image
    import models.glass_original_code.utils as utils
    import models.glass_original_code.glass as glass

from torchinfo import summary
from torchvision import models

GLASS_IMG_SIZE = 256  # to reduce GPU memory (12GB)


# GLASS 모델 가져오기
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - model (nn.Module) : GLASS 모델

# Reference:
# - https://github.com/cqylunlun/GLASS/blob/main/main.py (Original Code)

def get_model():
    gpu = ['0']
    device = utils.set_torch_device(gpu)
    GLASS_backbone = models.wide_resnet50_2(pretrained=False).cuda()

    model = glass.GLASS(device)
    model.load(
        backbone=GLASS_backbone,
        layers_to_extract_from=['layer2', 'layer3'],
        device=device,
        input_shape=(3, GLASS_IMG_SIZE, GLASS_IMG_SIZE),
        pretrain_embed_dimension=384,                    # originally 1536
        target_embed_dimension=384,                      # originally 1536
        patchsize=3,
        meta_epochs=5,
        eval_epochs=1,
        dsc_layers=2,
        dsc_hidden=256,                                  # originally 1024
        dsc_margin=0.5,
        train_backbone=False,
        pre_proj=1,
        mining=1,
        noise=0.015,
        radius=0.75,
        p=0.5,
        lr=0.0001,
        svd=0,
        step=20,
        limit=392
    )

    return model


if __name__ == '__main__':
    GLASS_model = get_model()

    summary(GLASS_model, input_size=(16, 3, GLASS_IMG_SIZE, GLASS_IMG_SIZE))
    save_model_graph_image(GLASS_model, model_name='GLASS', img_size=GLASS_IMG_SIZE)
