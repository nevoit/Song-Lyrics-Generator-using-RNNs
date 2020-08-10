"""
This function manages the LSTM model with lyrics only
"""
import os

from keras import Input, Model
from keras import backend as K
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, Masking
from keras.optimizers import Adam

# Environment settings
IS_COLAB = (os.name == 'posix')
LOAD_DATA = not (os.name == 'posix')

if not IS_COLAB:
    from rnn import RecurrentNeuralNetwork


class LSTMLyrics(RecurrentNeuralNetwork):
    def __init__(self, seed, loss, metrics, optimizer, learning_rate, total_words, seq_length, vector_size,
                 word2vec_matrix, units):
        """
        Seed - The seed used to initialize the weights
        width, height, cells - used for defining the tensors used for the input images
        loss, metrics, optimizer, dropout_rate - settings used for compiling the siamese model (e.g., 'Accuracy' and 'ADAM)
        :return Nothing
        """
        super().__init__(seed)
        K.clear_session()
        self.seed = seed
        self.initialize_seed()
        self.initialize_model(learning_rate, loss, metrics, optimizer, seq_length, total_words, units, vector_size,
                              word2vec_matrix)

    def initialize_model(self, learning_rate, loss, metrics, optimizer, seq_length, total_words, units, vector_size,
                         word2vec_matrix):
        """
        This function initializes the architecture and builds the model
        :param learning_rate: a tuning parameter in an optimization algorithm that determines the step size
        :param loss: the loss function we want to use
        :param metrics: the metrics we want to use, such as Loss
        :param optimizer: the optimizer function, such as Adam
        :param seq_length: the length of the sequence (the sentence in this case)
        :param total_words: total number of words we have (used for the output dense)
        :param units: number of LSTM units
        :param vector_size: the size of the embedding vector
        :param word2vec_matrix: the embedding matrix
        :return: Nothing
        """
        lyrics_features_input = Input((seq_length,))

        embedding_layer = Embedding(input_dim=total_words,  # the size of the vocabulary in the text data
                                    input_length=seq_length,  # the length of input sequences
                                    output_dim=vector_size,
                                    # the size of the vector space in which words will be embedded
                                    weights=[word2vec_matrix],
                                    trainable=False,
                                    # the model must be informed that some part of
                                    # the data is actually padding and should be ignored.
                                    mask_zero=True,
                                    name='MelodiesLyrics')(lyrics_features_input)

        masking_layer = Masking(mask_value=0.)(embedding_layer)
        # Bidirectional Recurrent layer
        b_rnn_layer = Bidirectional(LSTM(units=units, activation='relu'))(masking_layer)
        dropout_layer = Dropout(0.6)(b_rnn_layer)

        output_dense = Dense(units=total_words, activation='softmax')(dropout_layer)

        self.model = Model(inputs=lyrics_features_input, outputs=output_dense)
        if optimizer == 'adam':
            optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


print("Loaded Successfully")
