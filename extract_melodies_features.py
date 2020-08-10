import os
import pickle

import numpy as np
from tqdm import tqdm

# Environment settings
IS_COLAB = (os.name == 'posix')
LOAD_DATA = not (os.name == 'posix')

if not IS_COLAB:
    from prepare_data import create_validation_set


def get_midi_file_instrument_data(word_idx, time_per_word, midi_file):
    """
    Extract data about the midi file in the given time period. We will extract number of beat changes, instruments used
    and velocity.
    :param word_idx: index of word in the song
    :param time_per_word: Average time per word in song
    :param midi_file: The midi file
    :return: An array where each cell contains some data about the pitch, velocity etc.
    """
    #  Features we want to extract:
    start_time = word_idx * time_per_word
    end_time = start_time + time_per_word
    avg_velocity, avg_pitch, num_of_instruments, num_of_notes, beat_changes, has_drums = 0, 0, 0, 0, 0, 0

    for beat in midi_file.get_beats():
        if start_time <= beat <= end_time:
            beat_changes += 1  # Count beats that are in the desired time frame
        elif beat > end_time:
            break  # We passed the final possible time
    for instrument in midi_file.instruments:
        in_range = False  # Will become true if the instrument contributed at least 1 note for this sequence.
        for note in instrument.notes:
            if start_time <= note.start:
                if note.end <= end_time:  # In required range
                    has_drums = 1 if instrument.is_drum else has_drums
                    in_range = True
                    num_of_notes += 1
                    avg_pitch += note.pitch
                    avg_velocity += note.velocity
                else:  # We passed the last relevant note
                    break
        if in_range:
            num_of_instruments += 1
    if num_of_notes > 0:  # If there was at least 1 note
        avg_velocity /= num_of_notes
        avg_pitch /= num_of_notes
    final_features = np.array([avg_velocity, avg_pitch, num_of_instruments, beat_changes, has_drums])
    return final_features


def extract_melody_features_1(melodies_list, sequence_length, encoded_song_lyrics):
    """
    First function for extracting features about the midi files. Using the instrument objects in each midi file we can
    see when each instrument was used and with what velocity. We can then calculate the average pitch and velocity for
    each word in the song.
    
    :param melodies_list: A list of midi files. Contains the training / validation / test set typically.
    :param sequence_length: Number of words per sequence.
    :param encoded_song_lyrics: A list where each cell represents a song. The cells contain a list of ints, where each
    cell corresponds to a word in the songs lyrics and the value is the index of the word in our word2vec vocabulary.
    :return: A 3d numpy array where the first axis is the number of sequences in the data, the 2nd is the sequence
    length and the third is the number of notes for that particular word in that sequence.
    """

    final_features = []
    print('Extracting melody features v1..')

    for idx, midi_file in tqdm(enumerate(melodies_list)):
        num_of_words_in_song = len(encoded_song_lyrics[idx])
        midi_file.remove_invalid_notes()
        time_per_word = midi_file.get_end_time() / num_of_words_in_song  # Average time per word in the lyrics
        number_of_sequences = num_of_words_in_song - sequence_length
        features_during_lyric = []
        for word_idx in range(num_of_words_in_song):  # Iterate over every word and get the features for it
            instrument_data = get_midi_file_instrument_data(word_idx, time_per_word, midi_file)
            features_during_lyric.append(instrument_data)

        for sequence_num in range(number_of_sequences):
            seq = features_during_lyric[sequence_num:sequence_num + sequence_length]  # Create a sequence from the notes
            final_features.append(seq)

    final_features = np.array(final_features)
    return final_features


def extract_melody_features_2(melodies_list, sequence_length, encoded_song_lyrics):
    """
    Using all midi files and lyrics, extract features for all sequences. This is the second method we'll try. Basically,
    we will take the piano roll matrix for each song. This is a matrix that displays which notes were played for every
    user defined time period and some number representing the velocity. In our case, we'll slice the song every 1/50
    seconds (20 miliseconds) and look at what notes were played during this time. This is in addition to the features
    used in v1.
    :param melodies_list: A list of midi files. Contains the training / validation / test set typically.
    :param total_dataset_size: Total length of the sequence array,
    :param sequence_length: Number of words per sequence.
    :param encoded_song_lyrics: A list where each cell represents a song. The cells contain a list of ints, where each cell
    corresponds to a word in the songs lyrics and the value is the index of the word in our word2vec vocabulary.
    :return: A 3d numpy array where the first axis is the number of sequences in the data, the 2nd is the sequence
    length and the third is the number of notes for that particular word in that sequence.
    """

    final_features = []
    print('Extracting melody features v2..')
    frequency_sample = 50
    for midi_idx, midi_file in tqdm(enumerate(melodies_list)):
        num_of_words_in_song = len(encoded_song_lyrics[midi_idx])
        midi_file.remove_invalid_notes()
        time_per_word = midi_file.get_end_time() / num_of_words_in_song  # Average time per word in the lyrics
        number_of_sequences = num_of_words_in_song - sequence_length
        piano_roll = midi_file.get_piano_roll(fs=frequency_sample)
        num_of_notes_per_word = int(piano_roll.shape[1] / num_of_words_in_song)  # Num of piano roll columns per word
        features_during_lyric = []
        for word_idx in range(num_of_words_in_song):   # Iterate over every word and get the features for it
            notes_features = extract_piano_roll_features(num_of_notes_per_word, piano_roll, word_idx)
            instrument_data = get_midi_file_instrument_data(word_idx, time_per_word, midi_file)
            features = np.append(notes_features, instrument_data, axis=0)  # Concatenate them
            features_during_lyric.append(features)

        for sequence_num in range(number_of_sequences):
            # Create the features per sequence
            sequence_features = features_during_lyric[sequence_num:sequence_num + sequence_length]
            final_features.append(sequence_features)

    final_features = np.array(final_features)
    return final_features


def extract_piano_roll_features(num_of_notes_per_word, piano_roll, word_idx):
    start_idx = word_idx * num_of_notes_per_word
    end_idx = start_idx + num_of_notes_per_word
    piano_roll_for_lyric = piano_roll[:, start_idx:end_idx].transpose()
    piano_roll_slice_sum = np.sum(piano_roll_for_lyric, axis=0)  # Sum each column into a single cell
    return piano_roll_slice_sum


def get_melody_data_sets(train_num, val_size, melodies_list, sequence_length, encoded_lyrics_matrix, seed,
                         pkl_file_path, feature_method):
    """
    Creates numpy arrays containing features of the melody for the training, validation and test sets.
    :param feature_method: Method of feature extraction to use. Either '1' or '2'.
    :param seed: Seed for splitting to train and test.
    :param pkl_file_path: the file path to the pickle file. Used for saving or loading.
    :param train_num: Number of words in the whole training set sequence (train + validation)
    :param val_size: Percentage of sequences used for validation set
    :param melodies_list: All of the training + validation set midi files
    :param sequence_length: Number of words in a sequence
    :param encoded_lyrics_matrix: A list where each cell represents a song. The cells contain a list of ints, where each cell
    corresponds to a word in the songs lyrics and the value is the index of the word in our word2vec vocabulary.
    :return: numpy arrays containing features of the melody for the training, validation and test sets.
    """
    file_type = pkl_file_path.split('.')[-1]
    # Save/load the file with the appropriate name according to the settings used:
    pkl_file_path = f'{pkl_file_path.rstrip("." + file_type)}_{str(feature_method)}_sl_{sequence_length}.{file_type}'
    if os.path.exists(pkl_file_path):  # If file exists, use it instead of building it again
        with open(pkl_file_path, 'rb') as f:
            melody_train, melody_val, melody_test = pickle.load(f)
        return melody_train, melody_val, melody_test

    if feature_method == 'naive':  # Use appropriate melody feature method
        melody_features = extract_melody_features_1(melodies_list, sequence_length, encoded_lyrics_matrix)
    else:
        melody_features = extract_melody_features_2(melodies_list, sequence_length, encoded_lyrics_matrix)

    melody_train = melody_features[:train_num]
    melody_test = melody_features[train_num:]
    melody_train, melody_val = create_validation_set(melody_train, val_size, seed)

    with open(pkl_file_path, 'wb') as f:
        pickle.dump([melody_train, melody_val, melody_test], f)
        print('Dumped midi files')

    return melody_train, melody_val, melody_test


print("Loaded Successfully")
