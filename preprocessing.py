from data import *

class Tags:
    def __init__(self, annotations):
        self.tags = sorted(set(tag for sublist in annotations for tag in sublist))
        self.tag2id = {tag: id for id, tag in enumerate(self.tags)}
        self.id2tag = {id: tag for id, tag in enumerate(self.tags)}
    def __len__(self):
        return len(self.tags)


class Vocabulary:
    def __init__(self, sentences):
        words = sorted(set(word for sublist in sentences for word in sublist))
        self.word2id = {word: id for id, word in enumerate(words)}
        self.id2word = {id: word for id, word in enumerate(words)}
    def __len__(self):
        return len(self.word2id)

def numericize_words(q):
    word2id = {}
    id2word = {}
    for id, word in enumerate(q):
        word2id[word] = id
        id2word[id] = word
    return word2id,id2word

if __name__ == "__main__":
    tags = Tags(annotations)
    print(len(tags))
    vocab = Vocabulary(sentences)
    print(vocab.word2id)
    print(vocab.id2word)
    print(len(vocab))