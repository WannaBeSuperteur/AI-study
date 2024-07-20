from model_decide_train_data import Classify_Male_Or_Female_CNN_Model, load_training_data, predict_male_or_female_for_all_images
from model_utils import train_cnn_model
import tensorflow as tf
import os


def train_male_or_female_model(model_dir):
    if os.path.exists(model_dir):
        print('male/female model already exists')
        model = tf.keras.models.load_model(model_dir)
        return model

    print('loading training data ...')
    train_input, train_output = load_training_data('resized')

    print('loading cnn model ...')
    model = train_cnn_model(train_input, train_output, Classify_Male_Or_Female_CNN_Model)
    model.save(model_dir)

    return model


def predict_male_or_female(model):
    print('prediction start ...')
    all_data_dir_list = ['augmented/10k-images', 'augmented/female', 'augmented/male',
                         'resized/10k-images', 'resized/female', 'resized/male']

    predict_male_or_female_for_all_images(model, all_data_dir_list)


if __name__ == '__main__':
    model = train_male_or_female_model(model_dir='models/decide_train_data')
    predict_male_or_female(model)

