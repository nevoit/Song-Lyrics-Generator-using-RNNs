"""
This file manages the experiments, see the main function for changing the settings
"""
import os
import random
import time

import pandas as pd
from gtts import gTTS
from keras_preprocessing.text import Tokenizer


def main():
    """
    This function runs the process of the experiments. Iterates over the parameters and output the results.
    :return: Nothing
    """
    # Some settings for the files we will use
    saved_file_type = 'pkl'
    midi_pickle = os.path.join(PICKLES_FOLDER, f"midi.{saved_file_type}")
    midi_folder = os.path.join(DATA_PATH, "midi_files")

    # Read a pre-trained word2vec dictionary
    word2vec_path = os.path.join(PICKLES_FOLDER, f"{WORD2VEC_FILENAME}.{saved_file_type}")
    pre_trained = os.path.join(INPUT_FOLDER, f"{GLOVE_FILE_NAME}.txt")

    # Get the embedding dictionary that maps between word to a vector
    word2vec = get_word2vec(word2vec_path=word2vec_path,
                            pre_trained=pre_trained,
                            vector_size=VECTOR_SIZE,
                            encoding=ENCODING)

    # load the training and testing set that provided by the course staff
    train_pickle_path = os.path.join(PICKLES_FOLDER, f'{TRAIN_NAME}.{saved_file_type}')
    input_train_path = os.path.join(INPUT_FOLDER, INPUT_TRAINING_SET)
    training_set = get_input_sets(input_file=input_train_path,
                                  pickle_path=train_pickle_path,
                                  word2vec=word2vec,
                                  midi_folder=midi_folder)
    test_pickle_path = os.path.join(PICKLES_FOLDER, f'{TEST_NAME}.{saved_file_type}')
    input_test_path = os.path.join(INPUT_FOLDER, INPUT_TESTING_SET)
    testing_set = get_input_sets(input_file=input_test_path,
                                 pickle_path=test_pickle_path,
                                 word2vec=word2vec,
                                 midi_folder=midi_folder)

    artists = training_set['artists'] + testing_set['artists']
    songs_names = training_set['names'] + testing_set['names']
    lyrics = training_set['lyrics'] + testing_set['lyrics']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lyrics)
    total_words = len(tokenizer.word_index) + 1

    encoded_lyrics_list = tokenizer.texts_to_sequences(lyrics)
    index2word = tokenizer.index_word

    melodies = get_midi_files(midi_folder=midi_folder,
                              midi_pickle=midi_pickle,
                              artists=artists,
                              names=songs_names)

    train_encoded_lyrics_list = encoded_lyrics_list[:len(training_set['lyrics'])]
    test_encoded_lyrics_list = encoded_lyrics_list[len(training_set['lyrics']):]
    melody_pickle = os.path.join(PICKLES_FOLDER, "melody_data." + saved_file_type)

    comb_dict = {'seed': [], 'seq_length': [], 'learning_rate': [], 'batch_size': [], 'epochs': [],
                 'patience': [], 'min_delta': [], 'melody_method': [], 'model_names': [], 'cos_sim_1_gram': [],
                 'cos_sim_2_gram': [],
                 'cos_sim_3_gram': [], 'cos_sim_5_gram': [], 'cos_sim_max_gram': [], 'polarity_diff': [],
                 'subjectivity_diff': [], 'loss_val': [], 'accuracy': []}

    word2vec_matrix = get_word2vec_matrix(total_words=total_words,
                                          index2word=index2word,
                                          word2vec=word2vec,
                                          vector_size=VECTOR_SIZE)

    for seed in seeds_list:
        for sl in seq_length_list:
            sets_dict = create_sets(
                train_encoded_lyrics_list=train_encoded_lyrics_list,
                test_encoded_lyrics_list=test_encoded_lyrics_list,
                total_words=total_words,
                seq_length=sl,
                validation_set_size=VALIDATION_SET_SIZE,
                seed=seed)
            training_sequences = sets_dict['train'][1].shape[0] + sets_dict['validation'][1].shape[0]
            for melody_method in melody_extraction:
                m_train, m_val, m_test = get_melody_data_sets(
                    train_num=training_sequences,
                    val_size=VALIDATION_SET_SIZE,
                    melodies_list=melodies,
                    sequence_length=sl,
                    encoded_lyrics_matrix=encoded_lyrics_list,
                    pkl_file_path=melody_pickle,
                    seed=seed,
                    feature_method=melody_method)
                melody_feature_vector_size = m_train.shape[2]
                for l in learning_rate_list:
                    for bs in batch_size_list:
                        for ep in epochs_list:
                            for pa in patience_list:
                                for md in min_delta_list:
                                    for u in units_list:
                                        for m_name in model_names_list:
                                            run_combination(comb_dict, sl, bs, ep, index2word, l, md, pa, seed,
                                                            testing_set['artists'], melody_method,
                                                            testing_set['lyrics'], testing_set['names'], total_words, u,
                                                            word2vec,
                                                            word2vec_matrix, tokenizer, sets_dict['train'][0],
                                                            sets_dict['validation'][0], sets_dict['test'][0], m_train,
                                                            m_val, m_test, sets_dict['train'][1],
                                                            sets_dict['validation'][1], sets_dict['test'][1], m_name,
                                                            melody_feature_vector_size)
                                            if m_name == 'lyrics':
                                                break
    # Here we save all the results to a csv file
    comb_df = pd.DataFrame.from_dict(comb_dict)
    comb_df.to_csv(COMB_PATH, index=False)


def run_combination(comb_dict, seq_length, batch_size, epochs, index2word, learning_rate, min_delta, patience, seed,
                    test_artists, melody_extraction_method,
                    test_lyrics, test_names, total_words, units, word2vec, word2vec_matrix, tokenizer, x_train,
                    x_val, x_test, m_train, m_val, m_test, y_train, y_val, y_test, model_name, melody_num_features):
    """
    This function runs a combination with a specific settings and training or testing set
    :param melody_extraction_method: The method used to extract melody features (naive or with meta data)
    :param comb_dict: dictionary of all the results
    :param seq_length: this is the input sequence length we used for the LSTM model
    :param batch_size: the batch size for the model
    :param epochs: number of epochs for the model
    :param index2word: a dictionary maps between index and words.
    :param learning_rate: learning rate for the model
    :param min_delta: minimum delta for early stopping of the model
    :param patience: patience fo the early stopping of the model
    :param seed: for the random state
    :param test_artists: list of artist in the training set
    :param test_lyrics: list of lyrics in the training set
    :param test_names: list of songs name in the training set
    :param total_words: total size of the vocabulary
    :param units: number of LSTM units
    :param word2vec: dictionary maps between a word and a vector
    :param word2vec_matrix: a matrix of words (rows) and vectors (columns) of the word2vec
    :param tokenizer: Tokenizer object
    :param x_train: lyrics training set
    :param x_val: lyrics validation set
    :param x_test: lyrics testing xet
    :param m_train: melody training set
    :param m_val: melody validation set
    :param m_test: melody testing set
    :param y_train: training output words
    :param y_val: validation output words
    :param y_test: testing output words
    :param model_name: the name of the model we want to use in this function
    :param melody_num_features: size of the melody vector
    :return: Nothing
    """
    model_save_type = 'h5'  # file type
    initialize_seed(seed)  # files paths
    parameters_name = f'seq_lens_{seq_length}_seed_{seed}_u_{units}_lr_{learning_rate}_bs_{batch_size}_ep_{epochs}_' \
                      f'val_{VALIDATION_SET_SIZE}_pa_{patience}_md_{min_delta}_mn_{model_name}'
    if not model_name == 'lyrics':
        parameters_name += f'_fm_{melody_extraction_method}'
    # A path for the weights
    load_weights_path = os.path.join(WEIGHTS_FOLDER, f'weights_{parameters_name}.{model_save_type}')
    model = None
    if model_name == 'lyrics':
        model = LSTMLyrics(seed=seed,
                           loss=LOSS,
                           metrics=METRICS,
                           optimizer=OPTIMIZER,
                           learning_rate=learning_rate,
                           total_words=total_words,
                           seq_length=seq_length,
                           vector_size=VECTOR_SIZE,
                           word2vec_matrix=word2vec_matrix,
                           units=units)
    elif model_name == 'melodies_lyrics':
        x_train = [x_train, m_train]
        x_val = [x_val, m_val]
        x_test = [x_test, m_test]
        model = LSTMLyricsMelodies(seed=seed,
                                   loss=LOSS,
                                   metrics=METRICS,
                                   optimizer=OPTIMIZER,
                                   learning_rate=learning_rate,
                                   total_words=total_words,
                                   seq_length=seq_length,
                                   vector_size=VECTOR_SIZE,
                                   word2vec_matrix=word2vec_matrix,
                                   units=units,
                                   melody_num_features=melody_num_features)
    model.fit(weights_file=load_weights_path,
              batch_size=batch_size,
              epochs=epochs,
              patience=patience,
              min_delta=min_delta,
              x_train=x_train,
              y_train=y_train,
              x_val=x_val,
              y_val=y_val)
    loss_val, accuracy = model.evaluate(x_test=x_test, y_test=y_test, batch_size=batch_size)
    print(f'Loss on Testing set: {loss_val}')
    print(f'Accuracy on Testing set: {accuracy}')
    all_original_lyrics, all_generated_lyrics = generate_lyrics(
        model_name=model_name,
        word_index=index2word,
        seq_length=seq_length,
        model=model,
        tokenizer=tokenizer,
        artists=test_artists,
        lyrics=test_lyrics,
        names=test_names,
        word2vec=word2vec,
        melodies=m_test
    )
    cos_sim_1_gram = calculate_cosine_similarity_n_gram(all_generated_lyrics=all_generated_lyrics,
                                                        all_original_lyrics=all_original_lyrics,
                                                        n=1,
                                                        word2vec=word2vec)
    print(f'Mean Cosine Similarity (1-gram): {cos_sim_1_gram}')
    cos_sim_2_gram = calculate_cosine_similarity_n_gram(all_generated_lyrics=all_generated_lyrics,
                                                        all_original_lyrics=all_original_lyrics,
                                                        n=2,
                                                        word2vec=word2vec)
    print(f'Mean Cosine Similarity (2-gram): {cos_sim_2_gram}')
    cos_sim_3_gram = calculate_cosine_similarity_n_gram(all_generated_lyrics=all_generated_lyrics,
                                                        all_original_lyrics=all_original_lyrics,
                                                        n=3,
                                                        word2vec=word2vec)
    print(f'Mean Cosine Similarity (3-gram): {cos_sim_3_gram}')
    cos_sim_5_gram = calculate_cosine_similarity_n_gram(all_generated_lyrics=all_generated_lyrics,
                                                        all_original_lyrics=all_original_lyrics,
                                                        n=5,
                                                        word2vec=word2vec)
    print(f'Mean Cosine Similarity (5-gram): {cos_sim_5_gram}')
    cos_sim = calculate_cosine_similarity(all_generated_lyrics=all_generated_lyrics,
                                          all_original_lyrics=all_original_lyrics,
                                          word2vec=word2vec)
    print(f'Mean Cosine Similarity (Max-gram): {cos_sim}')
    pol_dif = get_polarity_diff(all_generated_lyrics=all_generated_lyrics, all_original_lyrics=all_original_lyrics)
    print(f'Mean Polarity Difference: {pol_dif}')
    subj_dif = get_subjectivity_diff(all_generated_lyrics=all_generated_lyrics, all_original_lyrics=all_original_lyrics)
    print(f'Mean Subjectivity Difference: {subj_dif}')
    update_comb_dict(batch_size, comb_dict, cos_sim, cos_sim_1_gram, cos_sim_2_gram, cos_sim_3_gram, cos_sim_5_gram,
                     epochs, learning_rate, min_delta, model_name, patience, pol_dif, seed, seq_length, subj_dif,
                     melody_extraction_method, loss_val, accuracy)


def update_comb_dict(batch_size, comb_dict, cos_sim, cos_sim_1_gram, cos_sim_2_gram, cos_sim_3_gram, cos_sim_5_gram,
                     epochs, learning_rate, min_delta, model_name, patience, pol_dif, seed, seq_length, subj_dif,
                     melody_extraction_method, loss_val, accuracy):
    """
    This function update the combination dictionary to write to csv
    :param accuracy: accuracy on the testing set
    :param loss_val: loss on the testing set
    :param batch_size: the batch size for the model
    :param comb_dict: the results dictionary
    :param cos_sim: the similarity score between the original and the generated sentence
    :param cos_sim_1_gram: the similarity score between each 1 gram of original and the generated sentence
    :param cos_sim_2_gram: the similarity score between each 2 gram of original and the generated sentence
    :param cos_sim_3_gram: the similarity score between each 3 gram of original and the generated sentence
    :param cos_sim_5_gram: the similarity score between each 5 gram of original and the generated sentence
    :param epochs: number of epochs for the model
    :param learning_rate: learning rate for the model
    :param min_delta: minimum delta for early stopping of the model
    :param model_name: The model name we want to test
    :param patience: patience fo the early stopping of the model
    :param pol_dif: the difference polarity score between the original and the generated sentence
    :param seed: for the random state
    :param seq_length: length of the given sequences
    :param subj_dif: the difference subjective score between the original and the generated sentence
    :param melody_extraction_method: The method used to extract melody features (naive or with meta data)
    :return: Nothing
    """
    comb_dict['seed'].append(seed)
    comb_dict['seq_length'].append(seq_length)
    comb_dict['learning_rate'].append(learning_rate)
    comb_dict['batch_size'].append(batch_size)
    comb_dict['epochs'].append(epochs)
    comb_dict['patience'].append(patience)
    comb_dict['min_delta'].append(min_delta)
    comb_dict['model_names'].append(model_name)
    comb_dict['cos_sim_1_gram'].append(cos_sim_1_gram)
    comb_dict['cos_sim_2_gram'].append(cos_sim_2_gram)
    comb_dict['cos_sim_3_gram'].append(cos_sim_3_gram)
    comb_dict['cos_sim_5_gram'].append(cos_sim_5_gram)
    comb_dict['cos_sim_max_gram'].append(cos_sim)
    comb_dict['polarity_diff'].append(pol_dif)
    comb_dict['subjectivity_diff'].append(subj_dif)
    comb_dict['melody_method'].append(melody_extraction_method)
    comb_dict['loss_val'].append(loss_val)
    comb_dict['accuracy'].append(accuracy)


def generate_song_given_sequence(model_name, model, tokenizer, seed_words, vector_of_indices, required_length, artist,
                                 name, index_value, melodies_song):
    """
    This function generates a new song
    :param model_name: model name
    :param melodies_song: a matrix contains the melodies of this song
    :param model:
    :param tokenizer:
    :param seed_words:
    :param vector_of_indices:
    :param required_length:
    :param artist:
    :param name:
    :param index_value:
    :return: Nothing
    """
    new_song_lyrics: list = [seed_words]
    for word_i in range(required_length):
        if model_name == 'lyrics':  # Different input for lyrics alone and lyrics and melodies.
            voc_prob = model.predict(vector_of_indices)
        else:
            melody_seq = np.expand_dims(a=melodies_song[word_i], axis=0)
            voc_prob = model.predict([vector_of_indices, melody_seq])
        voc_prob = voc_prob.T  # Transpose the array
        word_index_array = np.arange(voc_prob.size)
        # This line select a word based on the predicted probabilities
        index_of_selected_word = random.choices(word_index_array, k=1, weights=voc_prob)
        selected_word = find_word_by_index(word_index=index_of_selected_word[0], tokenizer=tokenizer)
        index_of_selected_word_array = np.array(np.array(index_of_selected_word).reshape(1, 1))
        vector_of_indices = np.append(vector_of_indices, index_of_selected_word_array, axis=1)
        remove_index = 0
        vector_of_indices = np.delete(vector_of_indices, remove_index, 1)
        new_song_lyrics.append(selected_word)
    final_text = ' '.join(new_song_lyrics)
    if WRITE_TO_MP3:
        lyrics_to_mp3 = gTTS(text=final_text, lang='en', slow=False)
        lyrics_to_mp3.save(os.path.join(OUTPUT_FOLDER, f"{artist}_{name}_{index_value}.mp3"))
    return final_text


def find_word_by_index(word_index, tokenizer):
    """
    This function returns the word given the index
    :param word_index: the index of the word we want to find
    :param tokenizer: object
    :return: the word at that index
    """
    for word, index in tokenizer.word_index.items():
        if index == word_index:
            return word


def generate_lyrics(model_name, word_index, seq_length, model, tokenizer, artists, lyrics, names,
                    word2vec, melodies) -> (list, list):
    """
    This function creates lyrics for each song in the testing set
    :param melodies: a 3D array that maps sequence and the melodies features (2D array (sequence size, melody vector)).
    :param model_name: The model name we want to test
    :param word_index: A dictionary maps between index to word
    :param seq_length: length of the given sequences
    :param model: the learned model
    :param tokenizer: the tokenizer object
    :param artists: list of artists in the testing set
    :param lyrics: list of lyrics in the testing set
    :param names: list of song names in the testing set
    :param word2vec: A dictionary maps between word to embedding vector
    :return: lists of original and generated songs and
    """
    all_original_lyrics = []
    all_generated_lyrics = []
    start_index_melody = 0
    for artist, name, lyrics in zip(artists, names, lyrics):
        print('-' * 100)
        print(f'Original lyrics for {artist} - {name} are: "{lyrics}"')
        relevant_words_in_song = []
        find_relevant_words(lyrics, relevant_words_in_song, word2vec)
        number_of_seq = len(relevant_words_in_song) - seq_length + 1
        end_index_melody = start_index_melody + number_of_seq
        melodies_song = melodies[start_index_melody:end_index_melody, :, :]
        required_length = len(relevant_words_in_song) - (seq_length * TESTING_SEED_TEXT_PER_SONG)
        for seed_index in range(TESTING_SEED_TEXT_PER_SONG):
            # We select three different word\sentence as seed for the new song
            starting_index = 0 + seed_index * seq_length
            ending_index = starting_index + seq_length
            song_first_word_in_word2vec = relevant_words_in_song[starting_index:ending_index]
            song_first_indices = []
            for word in song_first_word_in_word2vec:
                word_i = [k for k, v in word_index.items() if v == word][0]
                song_first_indices.append(word_i)
            encoded_test = np.asarray(song_first_indices).reshape((1, seq_length))
            seed_text = ' '.join(song_first_word_in_word2vec)
            generated_text = generate_song_given_sequence(model_name, model, tokenizer, seed_text, encoded_test,
                                                          required_length, artist, name, seed_index, melodies_song)
            gen_list = generated_text.split(' ')
            all_generated_lyrics.append(gen_list.copy()[seq_length:])
            original_starting_index = starting_index + seq_length
            original_ending_index = original_starting_index + required_length
            original_lyrics = relevant_words_in_song[original_starting_index:original_ending_index]
            all_original_lyrics.append(original_lyrics)
            gen_list.insert(seq_length, '\n')
            generated_text = ' '.join(gen_list)
            print(f'Seed text: {generated_text}, required {required_length} words')
        print('-' * 100)
        start_index_melody = end_index_melody + 1
    return all_original_lyrics, all_generated_lyrics


def find_relevant_words(lyrics, selected_words, word2vec):
    """
    This loop selects all the relevant words in the pre-defined word2vec
    :param lyrics:
    :param selected_words:
    :param word2vec:
    :return:
    """
    for word in lyrics.split():
        if word in word2vec and word not in selected_words:
            selected_words.append(word)


def initialize_seed(seed):
    """
    Initialize all relevant environments with the seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def folder_exists(path):
    """
    This function checks if folder path is exists, in case not, the function creates the folder.
    :param path: folder path
    """
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    # Environment settings
    IS_COLAB = (os.name == 'posix')
    LOAD_DATA = not (os.name == 'posix')
    path_separator = os.path.sep

    IS_EXPERIMENT = False
    WRITE_TO_MP3 = False
    if IS_COLAB:
        # the google drive folder we used
        DATA_PATH = os.path.sep + os.path.join('content', 'drive', 'My\ Drive', 'datasets', 'midi').replace('\\', '')
        IS_EXPERIMENT = True
    else:
        # locally
        from data_loader import get_word2vec
        from data_loader import get_input_sets
        from data_loader import get_midi_files
        from lstm_lyrics import LSTMLyrics
        from lstm_melodies_lyrics import LSTMLyricsMelodies
        from prepare_data import get_word2vec_matrix
        from prepare_data import create_sets
        from compute_score import calculate_cosine_similarity
        from compute_score import get_polarity_diff
        from compute_score import get_subjectivity_diff
        from compute_score import calculate_cosine_similarity_n_gram
        from extract_melodies_features import *

        DATA_PATH = os.path.join('.\\', 'midi')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # PATHS
    TRAIN_NAME = 'train'
    INPUT_TRAINING_SET = f"lyrics_{TRAIN_NAME}_set.csv"
    TEST_NAME = 'test'
    INPUT_TESTING_SET = f"lyrics_{TEST_NAME}_set.csv"
    OUTPUT_FOLDER = os.path.join(DATA_PATH, 'output_files')
    folder_exists(OUTPUT_FOLDER)
    INPUT_FOLDER = os.path.join(DATA_PATH, 'input_files')
    folder_exists(INPUT_FOLDER)
    PICKLES_FOLDER = os.path.join(DATA_PATH, 'pickles')
    folder_exists(PICKLES_FOLDER)
    WEIGHTS_FOLDER = os.path.join(DATA_PATH, 'weights')
    folder_exists(WEIGHTS_FOLDER)
    WORD2VEC_FILENAME = 'word2vec'
    RESULTS_FILE_NAME = 'results.csv'
    COMB_PATH = os.path.join(OUTPUT_FOLDER, RESULTS_FILE_NAME)
    GLOVE_FILE_NAME = 'glove.6B.300d'
    ENCODING = 'utf-8'

    LOSS = 'categorical_crossentropy'
    METRICS = ['accuracy']
    VECTOR_SIZE = 300
    VALIDATION_SET_SIZE = 0.2
    TESTING_SEED_TEXT_PER_SONG = 3
    OPTIMIZER = 'adam'

    if IS_EXPERIMENT:  # Experiments settings
        seeds_list = [0]
        learning_rate_list = [0.01]
        batch_size_list = [32, 64]
        epochs_list = [10]
        patience_list = [0]
        min_delta_list = [0.1]
        units_list = [256]
        seq_length_list = [1, 5, 20]
        model_names_list = ['melodies_lyrics', 'lyrics']
        melody_extraction = ['naive']
        # melody_extraction = ['naive', 'with_meta_features']
    else:  # Final settings
        seeds_list = [0]
        learning_rate_list = [0.01]
        batch_size_list = [32]
        epochs_list = [10]
        patience_list = [0]
        min_delta_list = [0.1]
        units_list = [256]
        seq_length_list = [1]
        model_names_list = ['melodies_lyrics']
        melody_extraction = ['naive']
        # model_names_list = ['melodies_lyrics', 'lyrics']
        # melody_extraction = ['naive', 'with_meta_features']

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
