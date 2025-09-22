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


def char_preprocess_sentences(xs):
    '''
    xs: list of preprocesed sentences on char level.
        each sentence is (L_sentence, L_token) tensor

    return: padded tensor of xs
    '''
    Nb = len(xs)
    L_sentence = max([x.shape[0] for x in xs])
    L_token = max([x.shape[1] for x in xs])

    preprocessed_sentences = torch.zeros((Nb, L_sentence, L_token), dtype=torch.int)
    for b, tensor_sentence in enumerate(xs):
        l1, l2 = tensor_sentence.shape
        preprocessed_sentences[b, :l1, :l2] = tensor_sentence
    return preprocessed_sentences


def char_matrix(word_vocab, char_vocab):
    word_V = len(word_vocab)
    out = torch.zeros((word_V, char_vocab.L_token),
                      dtype=torch.int)
    for word_id in range(1, word_V):
        word = word_vocab.id2word[word_id]
        for i, char in enumerate(word):
            out[word_id, i] = char_vocab.char2id[char]
    return out


def preprocess_target(raw_tags, tag_vocab):
    '''
    Preprocess
    raw_tags: list of strings
    '''
    output = []
    for tag in raw_tags:
        output.append(tag_vocab.tag2id[tag])
    return torch.tensor(output, dtype=torch.long)


def preprocess_sentence(sentence, word_vocab):
    '''
    :param sentence: (str) a single sentence

    :return: (L, ) (torch.tensor) numericize sentence word by word
    '''
    tokenize_sentence = sentence.split()
    preprocessed_sentence = []
    for token in tokenize_sentence:
        preprocessed_sentence.append(word_vocab.word2id[token])
    return torch.tensor(preprocessed_sentence, dtype=torch.long)


