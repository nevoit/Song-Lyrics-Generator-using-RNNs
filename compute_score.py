"""
This file computes the scores of the generated sentences
"""
import numpy as np
from numpy import dot
from numpy.linalg import norm
from textblob import TextBlob


def calculate_cosine_similarity_n_gram(all_generated_lyrics, all_original_lyrics, n, word2vec):
    """
    This function computes the similarity between 'n' words that are adjacent to each other.
    :param all_generated_lyrics: list of all generated lyrics
    :param all_original_lyrics: list of all original lyrics
    :param n: size of grams
    :param word2vec: a dictionary between word and index
    :return: mean similarity between the all_generated_lyrics and all_original_lyrics
    """
    cos_sim_list = []
    for song_original_lyrics, song_generated_lyrics in zip(all_original_lyrics, all_generated_lyrics):
        if len(song_original_lyrics) != len(song_generated_lyrics):
            raise Exception('The vectors are not equal')
        cos_sim_song_list = []
        for i in range(len(song_original_lyrics) - n + 1):
            starting_index = i
            ending_index = i + n
            n_gram_original = song_original_lyrics[starting_index:ending_index]
            n_gram_generated = song_generated_lyrics[starting_index:ending_index]
            original_vector = np.mean([word2vec[word] for word in n_gram_original], axis=0)
            generated_vector = np.mean([word2vec[word] for word in n_gram_generated], axis=0)
            cos_sim = dot(original_vector, generated_vector) / (norm(original_vector) * norm(generated_vector))
            cos_sim_song_list.append(cos_sim)
        cos_sim_song = np.mean(cos_sim_song_list)
        cos_sim_list.append(cos_sim_song)
    return np.mean(cos_sim_list)


def calculate_cosine_similarity(all_generated_lyrics, all_original_lyrics, word2vec):
    # The similarity between the generated lyrics and the original lyrics.
    cos_sim_list = []
    for song_original_lyrics, song_generated_lyrics in zip(all_original_lyrics, all_generated_lyrics):
        original_vector = np.mean([word2vec[word] for word in song_original_lyrics], axis=0)
        generated_vector = np.mean([word2vec[word] for word in song_generated_lyrics], axis=0)
        cos_sim = dot(original_vector, generated_vector) / (norm(original_vector) * norm(generated_vector))
        cos_sim_list.append(cos_sim)
    return np.mean(cos_sim_list)


def get_polarity_diff(all_generated_lyrics, all_original_lyrics):
    # The polarity score is a float within the range [-1.0, 1.0].
    pol_diff_list = []
    for song_original_lyrics, song_generated_lyrics in zip(all_original_lyrics, all_generated_lyrics):
        generated_lyrics = ' '.join(song_original_lyrics)
        generated_blob = TextBlob(generated_lyrics)
        original_lyrics = ' '.join(song_generated_lyrics)
        original_blob = TextBlob(original_lyrics)
        pol_diff = abs(generated_blob.sentiment.polarity - original_blob.sentiment.polarity)
        pol_diff_list.append(pol_diff)
    return np.mean(pol_diff_list)


def get_subjectivity_diff(all_generated_lyrics, all_original_lyrics):
    # The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    pol_diff_list = []
    for song_original_lyrics, song_generated_lyrics in zip(all_original_lyrics, all_generated_lyrics):
        generated_lyrics = ' '.join(song_original_lyrics)
        generated_blob = TextBlob(generated_lyrics)
        original_lyrics = ' '.join(song_generated_lyrics)
        original_blob = TextBlob(original_lyrics)
        pol_diff = abs(generated_blob.sentiment.subjectivity - original_blob.sentiment.subjectivity)
        pol_diff_list.append(pol_diff)
    return np.mean(pol_diff_list)


print("Loaded Successfully")
