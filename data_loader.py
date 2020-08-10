"""
This file manages the loading of the data
"""
import csv
import os
import pickle
import string

import numpy as np
import pretty_midi


def get_midi_files(midi_pickle, midi_folder, artists, names):
    """
    This function loads the midi files
    :param midi_pickle: path for the pickle file
    :param midi_folder: path for the midi folder
    :param artists: list of artist
    :param names: list of song names
    :return: list of pretty midi objects
    """
    # If the pickle file is already exists, read that file
    pretty_midi_songs = _read_pickle_if_exists(pickle_path=midi_pickle)
    if pretty_midi_songs is None:  # If the pickle is exists, covert the list into variables
        pretty_midi_songs = []
        lower_upper_files = get_lower_upper_dict(midi_folder)
        if len(artists) != len(names):
            raise Exception('Artists and Names lengths are different.')
        for artist, song_name in zip(artists, names):
            if song_name[0] == " ":
                song_name = song_name[1:]
            song_file_name = f'{artist}_-_{song_name}.mid'.replace(" ", "_")
            if song_file_name not in lower_upper_files:
                print(f'Song {song_file_name} does not exist, even though'
                      f' the song is provided in the training or testing sets')
                continue
            original_file_name = lower_upper_files[song_file_name]
            midi_file_path = os.path.join(midi_folder, original_file_name)
            try:
                pretty_midi_format = pretty_midi.PrettyMIDI(midi_file_path)
                pretty_midi_songs.append(pretty_midi_format)
            except Exception:
                print(f'Exception raised from Mido using this file: {midi_file_path}')

        _save_pickle(pickle_path=midi_pickle, content=pretty_midi_songs)
    return pretty_midi_songs


def get_lower_upper_dict(midi_folder):
    """
    This function maps between lower case name to upper case name
    :param midi_folder: midi folder path
    :return: A dictionary between lower case name to upper case name
    """
    lower_upper_files = {}
    for file_name in os.listdir(midi_folder):
        if file_name.endswith(".mid"):
            lower_upper_files[file_name.lower()] = file_name
    return lower_upper_files


def get_input_sets(input_file, pickle_path, word2vec, midi_folder) -> (list, list, list):
    """
    This function loads the training and testing set that provided by the course staff.
    In addition some pre-processing methods are work here.
    :param input_file: training or testing set path
    :param pickle_path: training or testing pickle path
    :param word2vec: dictionary maps between a word and a vector
    :param midi_folder: the midi folder that we use to validate if song is exists
    :return: Nothing
    """
    # If the pickle file is already exists, read that file
    pickle_value = _read_pickle_if_exists(pickle_path=pickle_path)
    # We want only songs with midi file
    lower_upper_files = get_lower_upper_dict(midi_folder)
    if pickle_value is not None:  # If the pickle is exists, covert the list into variables
        artists, names, lyrics = pickle_value[0], pickle_value[1], pickle_value[2]
    else:  # The pickle file is exists.
        artists, names, lyrics = [], [], []
        with open(input_file, newline='') as f:
            lines = csv.reader(f, delimiter=',', quotechar='|')
            for row in lines:
                artist_name = row[0]
                song_name = row[1]
                if song_name[0] == " ":
                    song_name = song_name[1:]
                song_file_name = f'{artist_name}_-_{song_name}.mid'.replace(" ", "_")
                if song_file_name not in lower_upper_files:
                    print(f'Song {song_file_name} does not exist, even though'
                          f' the song is provided in the training or testing sets')
                    continue
                original_file_name = lower_upper_files[song_file_name]
                midi_file_path = os.path.join(midi_folder, original_file_name)
                try:
                    pretty_midi.PrettyMIDI(midi_file_path)
                except Exception:
                    print(f'Exception raised from Mido using this file: {midi_file_path}')
                    continue
                song_lyrics = row[2]
                song_lyrics = song_lyrics.replace('&', '')
                song_lyrics = song_lyrics.replace('  ', ' ')
                song_lyrics = song_lyrics.replace('\'', '')
                song_lyrics = song_lyrics.replace('--', ' ')

                tokens = song_lyrics.split()
                table = str.maketrans('', '', string.punctuation)  # remove punctuation from each token
                tokens = [w.translate(table) for w in tokens]
                tokens = [word for word in tokens if
                          word.isalpha()]  # remove remaining tokens that are not alphabetic
                tokens = [word.lower() for word in tokens if word.lower() in word2vec]  # make lower case
                song_lyrics = ' '.join(tokens)
                artists.append(artist_name)
                names.append(song_name)
                lyrics.append(song_lyrics)
        _save_pickle(pickle_path=pickle_path, content=[artists, names, lyrics])

    return {'artists': artists, 'names': names, 'lyrics': lyrics}


def get_word2vec(word2vec_path, pre_trained, vector_size, encoding='utf-8') -> dict:
    """
    This function returns a dictionary that maps between word and a vector
    :param word2vec_path: path for the pickle file
    :param pre_trained: path for the pre-trained embedding file
    :param vector_size: the vector size for each word
    :param encoding: the encoding the the pre_trained file
    :return: dictionary maps between a word and a vector
    """
    # If the pickle file is already exists, read that file
    word2vec = _read_pickle_if_exists(word2vec_path)
    if word2vec is None:  # The pickle file is not exists.
        with open(pre_trained, 'r', encoding=encoding) as f:  # Read a pre-trained word vectors.
            list_of_lines = list(f)
        word2vec = _iterate_over_glove_list(list_of_lines=list_of_lines, vector_size=vector_size)
        _save_pickle(pickle_path=word2vec_path, content=word2vec)  # Save pickle for the next running
    return word2vec


def _iterate_over_glove_list(list_of_lines, vector_size):
    """
    This function iterates over the glove list line by line and returns a word2vec dictionary
    :param list_of_lines: List of glove lines
    :param vector_size: the size of the embedding vector size
    :return: dictionary maps between a word and a vector
    """
    word2vec = {}
    punctuation = string.punctuation
    for line in list_of_lines:
        values = line.split(' ')
        word = values[0]
        if word in punctuation:
            continue
        vec = np.asarray(values[1:], "float32")
        if len(vec) != vector_size:
            raise Warning(f"Vector size is different than {vector_size}")
        else:
            word2vec[word] = vec
    return word2vec


def _save_pickle(pickle_path, content):
    """
    This function saves a value to pickle file
    :param pickle_path: path for the pickle file
    :param content: the value you want to save
    :return: Nothing
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(content, f)


def _read_pickle_if_exists(pickle_path):
    """
    This function reads a pickle file
    :param pickle_path:path for the pickle file
    :return: the saved value in the pickle file
    """
    pickle_file = None
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_file = pickle.load(f)
    return pickle_file


print('Loaded Successfully')
