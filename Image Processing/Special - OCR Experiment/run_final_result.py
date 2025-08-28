
from run_extract_letters import extract_letters
from run_train_letter_classify_model import image_transform, load_pretrained_model, LETTERS

import numpy as np


if __name__ == '__main__':
    extracted_letters = extract_letters('test_black_white.png')
    pretrained_model = load_pretrained_model()
    result = ''

    for idx, letter in enumerate(extracted_letters):
        transformed_letter = image_transform(letter)
        prediction_result = pretrained_model(transformed_letter.unsqueeze(0).cuda())
        prediction_result = prediction_result.detach().cpu().numpy()
        prediction_max_idx = np.argmax(prediction_result)

        result += LETTERS[prediction_max_idx]

    print(f'final result:\n{result}')
