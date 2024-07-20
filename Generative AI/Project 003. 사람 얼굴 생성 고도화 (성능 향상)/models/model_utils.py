# from GAI-P2

import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# 모델 반환 (with early stopping + lr_reduced)
def define_model(model_class):
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)

    model = model_class()
    return model, optimizer, early_stopping, lr_reduced


def train_cnn_model(train_input, train_output, model_class, epochs=15, validation_split=0.1):

    """Training CNN model

    Args:
        train_input      (np.array)               : training input images
        train_output     (np.array)               : training output images
        model_class      (TensorFlow Model Class) : python class for model
        epochs           (int)                    : epochs for model training
        validation_split (float)                  : validation data split ratio for model training

    Outputs:
        cnn_model (TensorFlow Model) : trained CNN model
    """

    cnn_model, optimizer, early_stopping, lr_reduced = define_model(model_class)
    cnn_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(f'train input shape : {np.shape(train_input)}')
    print(f'train output shape : {np.shape(train_output)}')

    cnn_model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=epochs,
        validation_split=validation_split
    )

    cnn_model.summary()
    return cnn_model
