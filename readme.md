# The purpose of the assignment
Enabling students to experiment with building a recurrent neural net and using it on a real-world dataset.  
In addition to practical knowledge in the “how to” of building the network,  
an additional goal is introducing the students to the challenge of integrating different sources of information into a single framework.

## Introduction
In this assignment, we were tasked with creating a Recurrent Neural Network that can learn song lyrics and their melodies and then given a melody and a few words to start with, predict the rest of the song. This is essentially done by generating new words for the song and attempting to be as “close” as possible to the original lyrics. However, this is quite subjective leading the evaluation of generated words to use imaginative methods. For the training phase, however, we used Crossed Entropy loss.
The melody files and lyrics for each song were given to us and the train / test sets were predefined. 20% of the training data was used as a validation set in order to track our progress between training iterations.

We implemented this using an LSTM network. LSTMs have proven in the past to be successful in similar tasks because of their ability to remember previous data, which in our case is relevant because each lyric depends on the words (and melody) that preceded it.
The network receives as input a sequence of lyrics and predicts the next word to appear. The length of this sequence greatly affects the network’s predicting abilities since 5 words in a row work much better than just a single word. We tried using different values to see how this changes the accuracy of the model. During the training phase, sequences from the actual lyrics are fed into the network to train. After fitting the model, we can generate the lyrics for a whole song by beginning with an initial “seed” which is a sequence of words, predicting a word and then using it to advance the sequence like a moving window.

## Instructions
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

## Dataset Analysis:
- 600 song lyrics for the training
-  5 songs for the test set.
- Midi files for each song containing just the song's melody.
- Song lyrics features:
	- The length of a song is the number of words in the lyrics that are also present in the word2vec data.
		- For the training set:
		- Minimal song length: 3 words (Perhaps a hip hop song with lots of slang)
		- Maximal song length: 1338
		- Average song length: 257.37
	- For the test set:
		- Minimal song length: 94 words
		- Maximal song length: 389
		- Average song length: 231.6

**Input Files:**
A screenshot of the input folder

------ pic 1 -------

You need to put files in two folders: input_files and midi_files, the other folders are generated automatically.
Inside input_files put the glove 6B 300d file and the training and testing set:

------ pic 2 -------

An example of the glove file:

------ pic 3 -------

An example for lyrics_train_set.csv (columns: artist, song name and lyrics):

------ pic 4 -------

Inside the folder midi_files put the midi files:

------ pic 5 -------

## Code Design

Our code consists of three scripts:

1. Experiment.py - the script that runs the experiments to find the optimal parameters for our LSTM network.
2. Data_loader.py - Loads the midi files, the lyrics, fixes irregularities and cleans the song file names, loads the word embeddings file, saves and loads the various .pkl files.
3. Prepare_data.py - Performs various helper functions on the data such as splitting it properly, creating a validation set and creating the word embeddings matrix.
4. Compute_score.py - Because of the nature of this task, it is difficult to judge the successfulness of our model based on classic loss functions such as MSE. So this script contains several different methods to automatically score the output of our model, such as measuring the cosine similarity or the subjectivity of the lyrics. Explained more later.
5. Extract_melodies_features - Extracts the features we want from the midi files and splits them into train / test / validation. Explained more later.
6. Lstm_lyrics.py - The first LSTM model. This one only takes into account the lyrics of the song. This is used for comparison to see the improvement of using melodies.
7. Lstm_melodies_lyrics.py - The second LSTM model. This one incorporates the features of the midi files of each song. More on this later.

## Melody Feature Integration
We devised two different methods to extract features from the melodies. One of them a more naive technique, and the other a more sophisticated way that expands the first method.

**Method #1**: Each midi file contains a list of all instruments used in the file. For each instrument, an Instrument object contains a list of all time periods this instrument was used, the pitch used (the note)  and velocity (how strong the note was played) as you can see in figure 1.

------ pic 6 -------

Figure 1: The data available for each instrument of the midi file

The midi file contains the length of the melody, and we know the number of words in the lyrics, so we can easily approximate how many seconds lasts each word on average. Based on this, we assign each word a time span and can deduce what instruments were played during that word and how strong. If a word appears during times 15.2 - 15.8, we can search through the instrument objects for which ones appeared during that time frame. 

Using this data, we can compute how many instruments were used, their average pitch and average velocity per word. This provides the network some information about the nature of the song during this lyric, i.e. a low or high pitch and how high the velocity is.

In addition, we can easily use the function get_beats()  of pretty midi to find all the beat changes in the song and their times. We simply count the number of beat changes during the word’s time frame and thus add another feature for our network.

**Method #2**: With the first method we have the average pitch used for each word. Now, we want a more precise measurement of this. Each pretty midi object has a function getPianoRoll(fs) which returns a matrix that represents the notes used in the midi file on a near continuous time scale (See figure 1). Specifically, it returns an array of size 128\*S where the size of S equals the length of the song (i.e the time of the last note played) multiplied by how many times each second a sample is taken, denoted by the parameter fs. E.g, for fs=10 every 1/10ths of a second a sample will be made, meaning 10 samples per second so for a song of 120 seconds we will have 1200 samples. Thus getPianoRoll(fs=10) will return a matrix of size 128x1200. By this method, we can control the granularity of the data with ease.

------ pic 7 -------
Figure 2: Piano roll matrix. The value in each cell is the velocity summed across instruments.

The reason for the 128 is that musical pitch has a possible range of 0 to 127. So each column in this matrix represents the notes played during this sample (in our example, the notes played every 100 milliseconds).

After creating this matrix, we can calculate how many notes are played, on average, per word. For example, if there are 2000 columns and a song has 50 words, it means that each word in the lyrics can be connected to about 40 notes. This is not precise of course, but a useful approximation.

------ pic 8 -------

Figure 3: Notes played during a specific word in a song. Here each lyric received 40 notes representing it (columns 10-39 not shown). There are still 128 rows for each possible note.

We then iterate over every word in the song’s lyrics and find the notes that were played during that particular lyric. For example, in Figure 3 we can see that for a certain word, notes number 57, 64 and 69 were played. 

------ pic 9 -------
Figure 4: The sum of the notes played during a specific word.

Finally, for each lyric-specific matrix, we sum each row to easily see what notes were played and how much. In figure 4, we can see the result of summing the matrix presented in figure 3. This is fed together with the array of word embeddings of each word in the sequence, thus attaching melody features to word features.

## Architecture:
We used a fairly standard approach to a bidirectional LSTM network, with the addition of allowing it to receive as input both an embedding vector and the melody features. We also created an LSTM network that doesn’t receive melodies just to study the impact of melody on the results.

Number of layers: Both versions receive as input a sequence of lyrics. Then there is an embedding layer after the input that uses the word2vec dictionary to convert each word to the appropriate vector representing it. The difference between the networks is that the one using the melodies has a concatenating layer that appends the vectors of lyrics to the vector of melodies.

Additionally, we tried feeding the network various sequence lengths: 1, 5 and 10. We wanted to see how much the sequence length affects the results.

In addition to the piano roll matrix we keep the features extracted in method 1.

------ pic 10 -------


- Layers 3 & 4 are only for the model that uses the melody features.
- Since RNNs have to receive input of fixed length, we use masking to ensure that the input is the same size each time.
- We simply concatenate all of the features and feed it into the LSTM to utilize the melody features. However, the features entered vary greatly between our two approaches.
- We used a relatively high drop rate of 60% since we don’t want the network to converge too quickly and overfit on the training data. We tried lower values initially and found more success with 60%.
- The input of the final layer depends on the number of units in the Bidirectional LSTM.
- The final output is a probability for each word, and we sample one from there according to the distribution.

Tensorboard Graph:

------ pic 11 -------

**Stopping criteria:**
Here we also experimented with several parameters: We used the EarlyStopping function monitoring on the validation loss with a minimum delta of 0.1 (Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.) and patience of 0 (Number of epochs with no improvement after which training will be stopped). We experimented with several values and found the most success with these.

**Network Hyper-Parameters Tuning:**
NOTE: Here we explain the reasons behind the choices of the parameters. 
After implementing our RNN, we optimized the different parameters used. Some parameters, like the number of units in an LSTM, it is very hard to predict what will work best so this method is the best way to find good values to use.
Each combination takes a long time to train (5-15 minutes):

- Learning Rate: We tried different values, ranging from 0.1 to 0.00001. After running numerous experiments, we found 0.00001 to work the best.
- Epochs: We tried epochs of 5, 10 and 150. We found 10 to work the best.
- Batch size: We tried 32 and 2048. 32 worked better.
- Units in LSTM: 64 and 256
- We tried all of the possible combinations of the parameters detailed above which led to a huge number of experiments but led to us finding the optimal settings which were used in the section below.

## Results Evaluation
In this assignment, we were asked to generate lyrics for the 5 songs in the test set. One way to evaluate the results is simply to see how many cases did our model predict the word that was actually used in the song. However, this is not actually a good method to evaluate the model since if it generated a word that was incredibly similar to it simple accuracy wouldn’t detect that. Note that we let our model predict the exact same number of words as in the original song. We devised a few methods to judge our models lyrical capabilities:

1. **Cosine Similarity**: this is a general method to compare the similarity of two vectors. So if our model predicted “happy”, and the original lyrics had the word “smile”, we take the vector of each word from the embedding matrix and calculate the cosine similarity, 1 being the best and 0 the worst. There are a few variations for this however:
	2. Comparing each word predicted to the word in the song - the most straightforward method. If a song has 200 words we will perform 200 comparisons according to the index of each word.
	3. Creating n-grams of the lyrics, calculating the average of each n-gram and then comparing the n-grams according to their order. This method is a bit better in our opinion, since if the model predicted words (“A”, “B”) and they appeared as (“B”, “A”) in the song, an n-gram style similarity will determine that this was a good prediction while a unigram style won’t.  So we tried with 1, 2, 3 and 5-grams.
4. **Polarity**: Using the TextBlob package, we computed the polarity of the generated lyrics and the original ones. Polarity is a score ranging from -1 to 1, -1 representing a negative sentence and 1 representing a positive one. We checked if the lyrics carry the same feelings and themes more or less. We present in the results the absolute difference between them, meaning that a polarity difference of 0 means the lyrics have similar sentiments.
5. **Subjectivity**: Again drawing from TextBlob, subjectivity is a measure of how subjective a sentence is, 0 being very objective and 1 being very subjective. We calculate the absolute difference between the generated lyrics and the original lyrics.

Note: In the final section where we predict song lyrics, we tried with different seeds as requested. With a sequence length of S, we take the first S words (i.e, words #1, #2, ..#S) and predict the rest of the song. We then skip the first S words and take words S+1 until 2S. Then we skip the first 2S words and use words 2S+1 until 3S. 
Example with Sequence Length of 3:
Seed 1, seed 2 and seed 3 -

------ pic 12 -------

## Full Experimental Setup:
Validation Set: Empirically, we learned that using a validation set is better than not if there isn’t enough data. We used the fairly standard 80/20 ratio between training and validation which worked well.

- Batch sizes - 32
- Epochs - 5
- Learning rate: 0.01
- Min delta for improvement: 0.1
- 256 units in the LSTM layer

Additionally, we tried feeding the network various sequence lengths of 1 and 5 to study the effect on the quality of the results.

**Experimental Results:**
The best results are in bold -

------ pic 13 -------

**Analysis**: unlike our expectations, the model with simpler features worked better in almost all cases, perhaps due to Occam’s Razor. We theorize that the features about the instruments provided a good abstraction of the features of the entire piano roll. 
However, it is clear that adding some melody features to the model improved it on all parameters (except subjectivity). Additionally, having a sequence length of 5 has mixed results and doesn’t seem to have much of an impact on the evaluation methods we chose. We will look into this manually in the next section.

An interesting point is that for all cosine similarity evaluations, an increased n gave a higher similarity. We are not sure why this happens, but we think that with greater values of n the “average” word is more similar. We tested the cosine similarity where n={length of song}, and indeed the similarity was over 0.9. We then tested with a random choice of words and all of the words in a song (i.e., the average vector of the whole song), and the cosine similarity was a staggering 0.75. 

**Generated Lyrics:**
For Brevity’s sake we’ll only show both models with a sequence of 1 and the advanced model with a sequence of 5.

**Model with simple melody features - sequence length 1**
A screenshot from the TensorBoard framework: 
------ pic 14 -------

1. **Lyrics for the bangles - eternal flame**

	**Seed text: close **
	**close**  feelin dreams baby that cause like friends im have day cool let be their would your wit ignorance such forgiven oh may doll nothing down i now around suddenly ball have empty that beautiful how you lonely no goes gone you are called of for wanted me life of stress apart say i all way, required 55 words

	**Seed text: your** 
	**your**  gentle i were remember how swear she neither too girl out through with more love me me eyes said have i used heartache hmm anymore desire fighting she when stay be part lights spend by bite again say try ruining slide lover i eyes get always honey of maybe to it hope its white i, required 55 words

	**Seed text: eyes **
	**eyes**  and walk the night woah not live you his world more when just wakes you you fans me to it son sleeping you up i that it da we me let the i longing my do maybe warm fought a believe guys the hear blind dont your through this a down what tell gonna oh, required 55 words

2. **Lyrics for billy joel - honesty**
	**Seed text: if **
	**if**  do hell your as you hard so the be of mable we love do fat give about em with if show you me its of some can top tell if like over baby an the out that a right get as their leaves are oh come happy joy fight me thief give i goodbye sharing like hey all it last you open right i to tonight wake be shift i sister no i on got years wear to make show dont learn be you the live from outer jump drag the myself face shes raps, required 95 words

	**Seed text: you **
	**you**  sherry really but take my girl you and its kick knew so or the a tuya love no how love have of the me there the like its if i winter see reason baa i have would want im high him dancin ever but worked wanna the i mean the you when ill say get well leave up just actor that shit now do the chaka over dead got better to no my the imitating me and my can here and itself footsteps to like leave looked are phone for will will keep my mind, required 95 words

	**Seed text: search **
	**search**  it so class und any you and that friends cried day whoa fine the i three the in the you lovin its a and said hall way others let night hey beautiful dreams dishes save beer store evil back summer yeah forget when well both strong said you me way your the repeat jolly im what told the really to love huh the you baby go river get id and uranus what around with the down and you would always i heart dont with once go land mind come still so to them one else, required 95 words

3. Lyrics for cardigans - lovefool
	**Seed text: dear **
	**dear**  to pick tears slide low live such ill yourself me deep out crazy never kick i the belongs get others shelter before her it i wasnt survive ring off baby im to want life ho hanging if i each high you out mine you won rang woman i the do you we you certain guy the jesus my my much flame to you just you world pretty me to dont fault to ear know see love guide, required 77 words
	
	**Seed text: i **
	i  dumb look me kit i ive and clothes type meet all of didnt love baby the to you i the baby heart these and up look i out just family the what baby theyre all my love down sittin money be from something stars out no while now your got guide and time some was my you off would you is na man he and remember down hes best in hand be shotgun to leaves the that, required 77 words
	
	**Seed text: fear **
	**fear**  at ive no i your be friend kill thats you years im so right your hurts a if love ill night ever feel what his like ride behind love but man a going can good and gone do see if have name all turn the is start the about you down breaking you at the lady did hard call you the about threatening ass thing together in fall love i they its a up drop youre out, required 77 words

4. Lyrics for aqua - barbie girl

	**Seed text: hiya **
	**hiya**  put there to copa out kick when sad when it my cars girl the with in i me the some a around eyes stay cause be clock we never still cant missed anytime motion quiet ive go hot it on the a you had and sign live tennessee no fools got so i father hope for never for you the just it there me my believe other oh red your dont dream the drives the they chorus the happy crosses they to i i because if won this i want didnt ask the, required 93 words

	**Seed text: barbie **
	**barbie**  me go to country smiling all now from love she my is this world not that in to though i beat your be bad new hard cant pretty to wont to round do things without try it walking of ill things in man love a hands were for well you to no chuckie gonna i wish done arms tell lets it beat waiting found we good man write i nigga at do never you it ooh try are attention yeah oh hurt that too without roll yourself with the you feeling switch dont, required 93 words

	**Seed text: hi **
	**hi**  this feeling gotta that alone im do she sweet and you ever you the in had the the raise up skies it youre do me its inspired song with what that feel other mine time the easily what when you and three cause beat and its gets christmas your you sad a behind nothing a i number back or never and who your move beat you driving you i love and of do other like on go when oh yea heart plane after her that mine never soul like one you made you, required 93 words

5. Lyrics for blink 182 - all the small things 
	**Seed text: all **
	**all**  live ive fire love did my right so truck reading it life its sin heal well two home we confused mony its song you tried could disguise know find for amadeus where sailor you and the to wo insane yeah skin wind ride song me heart up bite a a new a i let money world didnt on, required 58 words
	
	**Seed text: the **
	**the**  love want risk whoa breakin take need cebu me amadeus control weve lose and try cryin away know hopes away what theres makes in you right drunk live always ever one bop your lovely on steal bet i say somebody say gonna sad stay frosty a grease scene his hangin your dry touch mind i you you your, required 58 words
	
	**Seed text: small **
	**small**  lost with sun find when casbah you time huh to please for you see make the life dont you me to she the waitin honey weed all fill fired wish on alone thats like im the to and yeah long sure the broadway the need somebody always achy dont well i as seen my that boy your that, required 58 words

**Model with advanced melody features - sequence length 5**
A screenshot from the TensorBoard framework: 
——— pic 15 ———-

1. **Lyrics for the bangles - eternal flame**

	**Seed text: close your eyes give me **
	**close your eyes give me** rest dreams want that ill what knows im yeah good trust gonna be those find your anyone temper boys dead oh forever somethin mine have i this would playin total yeah planetary that fast from you two like moon believe you will truth, required 43 words

	**Seed text: hand darling do you feel **
	**hand darling do you feel** of for isnt me more of rising midnight got i for cant whiskey i as rock need swear how watched mind thats got too when day love me me or every yeah i dreams forgotten cryin bag mony flesh how no own be, required 43 words
	
	**Seed text: My heart beating understand the** 
	**My heart beating understand the** air comin lying through began girl never another thorns bought slim i or go some coming of maybe to it fly so fuck i and thing the ive safe here nothing you long eyes day no but danced you you abandoned me to, required 43 words


2. **Lyrics for billy joel - honesty**
	**Seed text: If you search for tenderness **
	**If you search for tenderness**  my radio finger you now i that it ooh we me gonna the i gin my know maybe hair won a better bleed the live fantasy dont your too say a have baby think cant is do warm your tell you hard so the be of ow do love know believin really ever free when up said you me so of us can heartache more up what hot up before the got that a not are tell their di will oh down hands, required 83 words

	**Seed text:  it isnt hard to find  **
	** it isnt hard to find **   tired whats me pryor wanna i goodbye sueno what them all it last you since not i to shes joy be america i law like i be they lover catch and right said we stayin be you the nothing life fingers front anyone the myself without try dream cat da but he my thats you and so goin knew so ever the a waters love no from is was of the me right the what its up i wit have watching lied dont, required 83 words

	**Seed text: Can have the love need **
	**Can have the love need**  was over now im high story shoot give can wire wont the i mean the you no at got are only true want just handed that train youre know the yesterday hot women never alone to love my the believing it and my can need and clover tricks and if true needs will gun for let let much my then it dont note crowd wind you and that save nowhere good whoa alive the i breathe the it the you weve its a, required 83 words

3. Lyrics for cardigans - lovefool
	**Seed text: Dear i fear were facing **
	**Dear i fear were facing**  and every distant cause quite gonna ive them fast dreams wintry song dre my here world summer get music when only follow along every you me cause your the making dub im if such the da to love trying the you up one taking last hes and boss baby would when the see and you been look i from we with shit one taught then, required 65 words
	
	**Seed text: A problem you love me**
	**A problem you love me**  down wanna so to did man fool to wrote girls bought wake nothing boys at wait me though got crazy they goin the the drives go quite hiding keep again it i wasnt passed morning boy want oh to now night forgive sick up i fun has you got than you cares pushing wrong i the know you we you silk guy the rainy my, required 65 words
	
	**Seed text: No longer know and maybe **
	**No longer know and maybe**  my hes alien to you just you around needed me to dont trouper to bleed know have is dying seemed long me leapin i by a turned probably beautiful all of friend love want the to you i the want as chance and want little i got but plays the if up theyre for my is see fake loves be life hold money got like, required 65 words

4. Lyrics for aqua - barbie girl

	**Seed text: hiya barbie hi ken do **
	**hiya barbie hi ken do**  nobody youre of they dying and come little get my you face find you is rain look need and rock see them till in body be cute to jive the your feel by like i your love friend drag who you heard oh dont take your pieces a up love at ive give take baby really what ring else is but look a off with back and believe do have up yeah stop all gotta the im start the ever you, required 81 words

	**Seed text: a ride sure jump in **
	**a ride sure jump in **  fly for they for you the just my feel me my better other all making your dont dream the stone the come ba the hands covered come to i i kiss baby cares say i now friend ask the me one to throw asked for this could love heart my is say around he that in to knows i beat your love other another hard there red to turn to sometimes know hey stay am it street of at hey in, required 81 words

	**Seed text: you want to go for  **
	**you want to go for **  have killing you not the dawn than hard id you the ever unopened dimension over old in fall love i come its a want desire this got should feel to mess say goin when alive when it my bar thats the when in i me the us a would or remember ill love limit do time wanna there rhymes sounding rendezvous quiet by one stay it on the a you try and town nothing moonlit like stormy they dont i middle, required 81 words

5. Lyrics for blink 182 - all the small things 
	**Seed text: All the small things true **
	**All the small things true**   his love a such away its only you to no bread ill i sing deep sun think walk it beat fuck best do back look bone i lot feel know they you it gotta am will shaking was all running that around stay bitch wait with, required 46 words
	
	**Seed text: Care truth brings ill take **
	**Care truth brings ill take**   the you feeling push dont say goodbye an that hear im do how sweet and you his you the it new the the traveled want sayin it this do a its wannabe song with if that take other than come the horse if when you and, required 46 words
	
	**Seed text: One lift your ride best  **
	**One lift your ride best **  breathe make beat and its reason black of you fine a sleep mine a i scene too ever time and why your room beat you sings you i is and in do matter what on one no oh anybody from tu touch think that than they, required 46 words


## Analysis of how the Seed and Melody Effects the Generated Lyrics

We see that the lyrics are mostly unintelligible, and tend to have words that are very common in the data set (the word “love” appears over 40 times in the generated lyrics and it is indeed a common lyric in popular songs). It doesn’t appear that more advanced melody features improved the subjective quality of the lyrics produced, like how our quantitative methods deemed that it doesn’t improve much. We did notice however a peculiar feature, where once a word appeared for the first time, it tended to appear many times after (or similar variations of it, e.g. if “i” appeared then “i”, “me” or “my” tend to appear after it a lot”). This is to be expected from a model that maintains a cell state and predicts words based on their embedding.

Also it’s apparent that the seed chosen wildly changes the words produced. We think this is because the melody plays a much smaller part in predicting the lyrics compared to the seed, so even with the same melody the dominating factor in producing the lyrics is the seed - see our evaluation table above; the results are slightly better with the melody attached, but not by much, meaning that the first word assists the model much more compared to, say, a baseline of a random word each time. 

Personally, we don’t see much of an improvement in using 5 words as a seed versus just the first word. Occasionally it leads to better combinations but it’s a hit-or-miss usually. We think this is because of 2 main reasons:
Many songs contain slang that isn’t in the word embedding matrix so we cannot learn from them or predict them
Many song lyrics aren’t completely coherent and the words are fairly independent of each other (for a good example see the original lyrics of the last song in the test set, “All the Small Things” by Blink 182).
