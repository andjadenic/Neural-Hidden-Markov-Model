from data import *
import numpy as np

class Tags:
    def __init__(self, annotations):
        self.tags = sorted(set(tag for sublist in annotations for tag in sublist))

        self.tag2id = {tag: id for id, tag in enumerate(self.tags, start=1)}
        self.id2tag = {id: tag for id, tag in enumerate(self.tags, start=1)}

        self.tag2id['PAD'] = 0
        self.id2tag[0] = 'PAD'

    def __len__(self):
        return len(self.tags)


class Vocabulary:
    def __init__(self, sentences):
        tokenized_sentence = [sentence.split() for sentence in sentences]
        words = sorted(set(word for sublist in tokenized_sentence for word in sublist))

        self.word2id = {word: id for id, word in enumerate(words, start=1)}
        self.id2word = {id: word for id, word in enumerate(words, start=1)}

        self.word2id['PAD'] = 0
        self.id2word[0] = 'PAD'
    def __len__(self):
        return len(self.word2id)

def preprocess_data(sentences, raw_tags, vocab, tag_vocab):
    out_list_sentences = []
    out_list_tags = []
    max_L = 0

    # tokenize and numericize sentences
    for sentence in sentences:
        tokens = sentence.split()
        if len(tokens) > max_L:
            max_L = len(tokens)
        numericized_sentence = []
        for token in tokens:
            numericized_sentence.append(vocab.word2id[token])
        out_list_sentences.append(numericized_sentence)

    # pad sentences
    for numericized_sentence in out_list_sentences:
        numericized_sentence += (max_L - len(numericized_sentence)) * [vocab.word2id['PAD']]

    # numericize and pad tags
    for single_raw_ann in raw_tags:
        out_list_tag = []
        for raw_tag in single_raw_ann:
            out_list_tag.append(tag_vocab.tag2id[raw_tag])
        out_list_tag += (max_L - len(single_raw_ann)) * [tag_vocab.tag2id['PAD']]
        out_list_tags.append(out_list_tag)
    return np.array(out_list_sentences), np.array(out_list_tags)



if __name__ == "__main__":
    tags_vocab = Tags(training_tags)
    vocab = Vocabulary(training_sentences)

    training_x, training_z = preprocess_data(training_sentences, training_tags, vocab, tags_vocab)
