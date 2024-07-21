import tensorflow as tf
from test_cvae import test_all_cases

# TODO bugfix for 'self.model.decoder'

class ModelTestCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pass
#        test_all_cases(self.model.decoder, epoch_no=epoch)
