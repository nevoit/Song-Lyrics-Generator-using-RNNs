"""
This function manages the general RNN architecture
"""
import os
import random

import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard

# Environment settings
IS_COLAB = (os.name == 'posix')
LOAD_DATA = not (os.name == 'posix')

if IS_COLAB:
    from datetime import datetime
    from packaging import version

    # Define the Keras TensorBoard callback.
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)


class RecurrentNeuralNetwork(object):
    def __init__(self, seed):
        """
        Seed - The seed used to initialize the weights
        width, height, cells - used for defining the tensors used for the input images
        loss, metrics, optimizer, dropout_rate - settings used for compiling the siamese model (e.g., 'Accuracy' and 'ADAM)
        """
        K.clear_session()
        self.seed = seed
        self.initialize_seed()
        self.model = None

    def initialize_seed(self):
        """
        Initialize seed all for environment
        """
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _load_weights(self, weights_file):
        """
        A function that attempts to load pre-existing weight files for the siamese model. If it succeeds then returns
        True and updates the weights, otherwise False.
        :return True if the file is already exists
        """
        self.model.summary()
        self.load_file = weights_file
        if os.path.exists(weights_file):  # if the file is already exists, load and return true
            print('Loading pre-existed weights file')
            self.model.load_weights(weights_file)
            return True
        return False

    def fit(self, weights_file, batch_size, epochs, patience, min_delta, x_train, y_train, x_val, y_val):
        """
        Function for fitting the model. If the weights already exist, just return the summary of the model. Otherwise,
        perform a whole train/validation/test split and train the model with the given parameters.
        """
        # Create callbacks
        if not self._load_weights(weights_file=weights_file):
            print('No such pre-existed weights file')
            print('Beginning to fit the model')
            if IS_COLAB:
                callbacks = [
                    tensorboard_callback,
                    EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta)
                ]
            else:
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta)
                ]
            self.model.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           callbacks=callbacks,
                           validation_data=(x_val, y_val))
            self.model.save_weights(self.load_file)
        # evaluate on the validation set
        loss, accuracy = self.model.evaluate(x_val, y_val, batch_size=batch_size)
        print(f'Loss on Validation set: {loss}')
        print(f'Accuracy on Validation set: {accuracy}')

    def evaluate(self, x_test, y_test, batch_size):
        """
        Function for evaluating the final model after training.
        test_file - file path to the test file.
        batch_size - the batch size used in training.

        Returns the loss and accuracy results.
        """
        print(f'Available Metrics: {self.model.metrics_names}')
        y_test = np.array(y_test, dtype='float64')
        x_test[0] = np.array(x_test[0], dtype='float64')
        x_test[1] = np.array(x_test[1], dtype='float64')
        # evaluate on the test set
        loss, accuracy = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        return loss, accuracy

    def predict(self, data):
        return self.model.predict(data)


print('Loaded Successfully')
