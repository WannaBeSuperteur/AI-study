
from torchvision.utils import save_image
import os


def save_tensor_as_image(tensor, dir_name, img_idx):
    transformed_tensor = 0.5 * tensor + 0.25
    os.makedirs(dir_name, exist_ok=True)

    img_path = f'{dir_name}/image_{img_idx:04d}.png'
    save_image(transformed_tensor, img_path)
