
from run_extract_letters import extract_letters
from run_train_letter_classify_model import image_transform, load_pretrained_model, LETTERS

import torch
import numpy as np

from common import save_tensor_as_image


if __name__ == '__main__':
    pretrained_model_path = 'models/letter_classify_model.pth'

    pretrained_model = load_pretrained_model()
    state_dict = torch.load(pretrained_model_path, map_location=pretrained_model.device)
    pretrained_model.load_state_dict(state_dict, strict=True)
    pretrained_model.eval()

    extracted_letters = extract_letters('test_black_white.png')
    result = ''

    for idx, letter in enumerate(extracted_letters):
        transformed_letter = image_transform(letter)
        save_tensor_as_image(transformed_letter, dir_name='final_result', img_idx=idx)

        prediction_result = pretrained_model(transformed_letter.unsqueeze(0).cuda())
        prediction_result = prediction_result.detach().cpu().numpy()
        prediction_max_idx = np.argmax(prediction_result)

        result += LETTERS[prediction_max_idx]

    print(f'final result:\n{result}')
