# The purpose of the assignment
Enabling students to experiment with building a recurrent neural net and using it on a real-world dataset.  
In addition to practical knowledge in the “how to” of building the network,  
an additional goal is introducing the students to the challenge of integrating different sources of information into a single framework.

# Introduction
In class we covered the topic of automatic sentence completion/generation.  
In addition to the completion of “regular” sentences,  
this technique can also be applied to other domains such as lyrics and melodies generation (a nice example is BachBot https://bachbot.com/).  
In this task you will train a neural net to generate lyrics based on the provided melody.

During the training of the model you will have access both to the lyrics of a song and its melody.  
The melodies are stored in .mid (MIDI files) and contain various types of information – notes, the instruments used etc.  
You are encouraged to experiment with various methods to incorporate this information with the lyrics.  
During the test phase, you are required to automatically generate lyrics for a provided melody.

Please note that this assignment cannot be measured using objective (i.e., absolute) performance measures.  
Instead, we will be evaluating your approach to the solution, the implementation, and your analysis of your model’s performance.

# Instructions
1. Please download the following:
* A .zip file containing all the MIDI files of the participating songs
* the .csv file with all the lyrics of the of the participating songs (600 train and 5 test)
* [Pretty_Midi](https://nbviewer.jupyter.org/github/craffel/pretty-midi/blob/master/Tutorial.ipynb) , a python library for the analysis of MIDI files

2. Implement a recurrent neural net (LSTM or GRU) to carry out the task described in the introduction.
* During each step of the training phase, your architecture will receive as input one word of the lyrics. Words are to represented using the Word2Vec representation that can be found online (300 entries per term, as learned in class).
* The task of the network is to predict the next word of the song’s lyrics. Please see the figure 1 for an illustration. You may use any loss function
* In addition to this textual information, you need to include information extracted from the MIDI file. The method for implementing this requirement is entirely up to your consideration. Figure 1 shows one of the more simplistic options – inserting the entire melody representation at each step.
* Note that your mechanism for selecting the next word should not be deterministic (i.e., always select the word with the highest probability) but rather be sampling-based. The likelihood of a term to be selected by the sampling should be proportional to its probability.
* You may add whatever additions you want to the architecture (e.g., regularization, attention, teacher forcing)
* You may create a validation set. The manner of splitting (and all related decisions) are up to you.

3. The Pretty_Midi package offers multiple options for analyzing .mid files.
Figures 2-4 demonstrate the types of information that can be gathered.

4. You can add whatever other information you consider relevant to further improve the performance of your model.

5. You are to evaluate two approaches for integrating the melody information into your model. The two approaches don’t have to be completely different (one can build upon the other, for example), but please refrain from making only miniature changes.

6. Please include the following information in your report regarding the training phase:
*	The chosen architecture of your model
*	A clear description of your approach(s) for integrating the melody information together with the lyrics
*	TensorBoard graphs showing the training and validation loss of your model.

7. Please include the following information in your report regarding the test phase:
*	For each of the melodies in the test set, produce the outputs (lyrics) for each of the two architectural variants you developed. The input should be the melody and the initial word of the output lyrics. Include all generated lyrics in your submission.
*	For each melody, repeat the process described above three times, with different words (the same words should be used for all melodies).
*	Attempt to analyze the effect of the selection of the first word and/or melody on the generated lyrics.



