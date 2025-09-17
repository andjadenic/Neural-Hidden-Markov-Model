from data import *
import torch

class Tag_vocabulary:
    def __init__(self, annotations):
        self.tags = sorted(set(tag for sublist in annotations for tag in sublist))

        self.tag2id = {tag: id for id, tag in enumerate(self.tags, start=1)}
        self.id2tag = {id: tag for id, tag in enumerate(self.tags, start=1)}

        self.tag2id['PAD'] = 0
        self.id2tag[0] = 'PAD'
        self.tags.append('PAD')

        self.L = max([len(s) for s in annotations])

    def __len__(self):
        return len(self.tags)


class Char_vocabulary:
    def __init__(self, sentences):
        tokenized_sentences = [sentence.split() for sentence in sentences]  # token is one word
        chars = set([char for tok_sentence in tokenized_sentences for token in tok_sentence for char in token])
        char2id = {char: id for id, char in enumerate(chars, start=1)}
        char2id['PAD'] = 0
        id2char = {id: char for id, char in enumerate(chars, start=1)}
        id2char[0] = 'PAD'

        self.char2id = char2id
        self.id2char = id2char

        self.L_sentence = max([len(sentence) for sentence in tokenized_sentences])
        self.L_token = max([len(token) for sentence in tokenized_sentences for token in sentence])


    def __len__(self):
        return len(self.char2id)


class Word_vocabulary:
    def __init__(self, sentences):
        tokenized_sentences = [sentence.split() for sentence in sentences]  # token is one word
        self.words = sorted(set([word for sentence in tokenized_sentences for word in sentence]))

        self.word2id = {word:id for id, word in enumerate(self.words, start=1)}
        self.word2id['PAD'] = 0
        self.id2word = {id:word for id, word in enumerate(self.words, start=1)}
        self.id2word[0] = 'PAD'

        self.words.append('PAD')

        self.L_sentence = max([len(sentence) for sentence in tokenized_sentences])


    def __len__(self):
        return len(self.word2id)


def char_preprocess_sentences(sentences, vocab):
    '''
    :param sentences: batch of sentences given as list of list of strings

    :return: numerized and padded sentences char by char
    '''
    tokenized_sentences = [sentence.split() for sentence in sentences]

    Nb = len(sentences)
    L_sentence = max([len(sentence) for sentence in tokenized_sentences])
    L_token = max([len(token) for tokenized_sentence in tokenized_sentences for token in tokenized_sentence])

    preprocessed_sentences = torch.zeros((Nb, L_sentence, L_token), dtype=torch.int)
    for b, sentence in enumerate(tokenized_sentences):
        for i_s, token in enumerate(sentence):
            for i_c, char in enumerate(token):
                preprocessed_sentences[b, i_s, i_c] = vocab.char2id[char]
    return preprocessed_sentences


def preprocess_tags(raw_tags, tags_vocab):
    Nb = len(raw_tags)

    preprocessed_annotations = torch.zeros((Nb, tags_vocab.L), dtype=int)
    for b, tags in enumerate(raw_tags):
        for i, tag in enumerate(tags):
            preprocessed_annotations[b, i] = tags_vocab.tag2id[tag]
    return preprocessed_annotations


if __name__ == "__main__":
    # Build token (word) vocabulary
    word_vocab = Word_vocabulary(training_sentences)
    V = len(word_vocab)

    # Build char vocabulary
    char_vocab = Char_vocabulary(training_sentences)
    ch_V = len(char_vocab)

    # Preprocess sentences
    char_preprocessed_sentences = char_preprocess_sentences(training_sentences, char_vocab)  # (Nb, L_sentence, L_token)

    # Build tag vocabulary
    tags_vocab = Tag_vocabulary(training_tags)
    K = len(tags_vocab)

    # Preprocess tags
    preprocessed_annotations = preprocess_tags(training_tags, tags_vocab)
