"""
This file manages the data preparation
"""
import numpy as np


def get_word2vec_matrix(total_words, index2word, word2vec, vector_size):
    """
    This function creates a matrix where the rows are the words and the columns represents the embedding vector.
    We will use this matrix in the embedding layer
    :param total_words: Number of words in our word2vec dictionary.
    :param index2word: dictionary maps between index and word
    :param word2vec: dictionary maps between a word and a vector
    :param vector_size: the size of the embedding vector size
    :return: embedding layer
    """
    word2vec_matrix = np.zeros((total_words, vector_size))
    for index_word, word in index2word.items():
        if word not in word2vec:
            print(f'Can not find the word "{word}" in the word2vec dictionary')
            continue
        else:
            vec = word2vec[word]
            word2vec_matrix[index_word] = vec
    return word2vec_matrix


def _create_sequences(encoded_lyrics_list, total_words, seq_length):
    """
    This function creates sequences from the lyrics
    :param encoded_lyrics_list: A list representing all the songs in the dataset (615 songs). Each cell contains a list
    of ints, where each int corresponds to the lyrics in that song. "I'm a barbie girl" --> [23, 52, 189, 792] etc.
    :param total_words: Number of words in our word2vec dictionary.
    :param seq_length: Number of words predating the word to be predicted.
    :return: (1) A numpy array containing all the sequences seen, concatenated.
             (2) A 2d numpy array where each row represents a word and the columns are the possible words in the
             vocabulary. There is a '1' in the corresponding word (e.g, word number '20,392' in the dataset is word
              number '39' in the vocab.
    """
    input_sequences = []
    next_words = []
    for song_sequence in encoded_lyrics_list:  # iterate over songs
        for i in range(seq_length, len(song_sequence)):  # iterate from minimal sequence length (number of words) to
            start_index = i - seq_length  # number of words in the song
            end_index = i
            # Slice the list into the desired sequence length
            sequence = song_sequence[start_index:end_index]
            input_sequences.append(sequence)
            next_word = song_sequence[end_index]
            next_words.append(next_word)
    input_sequences = np.array(input_sequences)
    one_hot_encoding_next_words = convert_to_one_hot_encoding(input_sequences, next_words, total_words)
    return input_sequences, one_hot_encoding_next_words


def convert_to_one_hot_encoding(input_sequences, next_words, total_words):
    """
    This function converts input to one hot encoding
    """
    one_hot_encoding_next_words = np.zeros((len(input_sequences), total_words), dtype=np.int8)
    for word_index, word in enumerate(next_words):
        one_hot_encoding_next_words[word_index, word] = 1
    return one_hot_encoding_next_words


def create_sets(train_encoded_lyrics_list, test_encoded_lyrics_list, total_words, seq_length, validation_set_size,
                seed):
    """
    This function splits training set to smaller training set and new validation set
    :param train_encoded_lyrics_list: list of sequences in the training set
    :param test_encoded_lyrics_list: list of sequences in the testing set
    :param total_words: total words in the lyrics
    :param seq_length: length of the sequence
    :param validation_set_size: percentage of the validation set
    :param seed: random state for the split
    :return: training/testing/validation set values and labels
    """
    x_train, y_train = _create_sequences(encoded_lyrics_list=train_encoded_lyrics_list,
                                         total_words=total_words, seq_length=seq_length)

    x_train, x_val = create_validation_set(data_to_split=x_train,
                                           val_data_percentage=validation_set_size,
                                           seed=seed)
    y_train, y_val = create_validation_set(data_to_split=y_train,
                                           val_data_percentage=validation_set_size,
                                           seed=seed)

    x_test, y_test = _create_sequences(encoded_lyrics_list=test_encoded_lyrics_list,
                                       total_words=total_words, seq_length=seq_length)

    return {'train': (x_train, y_train), 'validation': (x_val, y_val), 'test': (x_test, y_test)}


def create_validation_set(data_to_split, val_data_percentage, seed):
    """
    This function splits to training and validation set
    :param data_to_split: matrix where the rows are the sequences and the columns are the word indices
    :param val_data_percentage: percentage of the validation set
    :param seed: random state for the split
    :return: training and validation set
    """
    np.random.seed(seed=seed)
    np.random.shuffle(data_to_split)

    validation_ending_index = int(len(data_to_split) * val_data_percentage)
    validation_set = data_to_split[:validation_ending_index]
    data_to_split = data_to_split[validation_ending_index:]

    return data_to_split, validation_set


print('Loaded Successfully')
