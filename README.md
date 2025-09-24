# Goal
The goal of this project is to automate **part-of-speech tagging**, the process of identifying a word in a text as a noun, verb, adjective, adverb, etc., based on both its definition and its context.

Part-of-speech tagging is a fully supervised learning task because we have a corpus of words labeled with the correct part-of-speech tag.

* Input example: 'The big company rises'
* Output example: 'DT', 'ADJ', 'NN', 'VBZ'

# Dataset
The dataset used for training the model consists of 80 small sentence-tag pairs that imitate the WSJ corpus.

The dataset is listed in [data/data.py](https://github.com/andjadenic/Neural-Hidden-Markov-Model/blob/master/data/data.py) file.

## Data preprocessing
The project utilizes three vocabulary classes: `char_vocab`, `word_vocab`, and `tag_vocab`, defined in [data/preprocessing.py](https://github.com/andjadenic/Neural-Hidden-Markov-Model/blob/master/data/preprocessing.py), which are used to preprocess raw sentences and tags.

# Word and tag embeddings
Both words and tags are mapped into D-dimensional space. Those embeddings are later used to form model parameters.

* Tag embedding is a simple neural network made out of a single fully connected linear layer, followed by a ReLU activation function, defined in [models/tag_embedding.py](https://github.com/andjadenic/Neural-Hidden-Markov-Model/blob/master/models/word_embedding.py). 
* Word embedding vectors derived from a Convolutional Neural Network (CNN), using D convolutional kernels and max pool, defined in [models/word_embedding.py](https://github.com/andjadenic/Neural-Hidden-Markov-Model/blob/master/models/word_embedding.py). This allows the model to automatically learn lexical representations based on prefix, suffix, and stem information about a word. 


# Model architecture
[Hidden Markov Model](https://web.stanford.edu/~jurafsky/slp3/A.pdf) is assumed to be the underlying model, commonly used for the POS tagging task.

In the POS tagging task, hidden states are tags, and observed states are words.
Model parameters, transition and emission probability matrices A and B, defined in [models/nhmm](https://github.com/andjadenic/Neural-Hidden-Markov-Model/blob/master/models/nhmm.py) are learned during training and used for inference.

## Transition architecture


## Emission architecture
The emission probability of producing the word `v` given the tag`k` is computed as a normalized scalar product of the embedded word `v` and the embedded tag `k`:

$$P{word_v | tag_k} = \dfrac{e^{\text{embedded}(word_v) \cdot \text{embedded}(tag_k)}}{\sum_{j=1}^k e^{\text{embedded}(word_j) \cdot \text{embedded}(tag_k)}}$$.



# Training

# Inference

# Utils

