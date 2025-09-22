from pythonProject1.models.word_embedding import *
from pythonProject1.models.tag_embedding import *
from pythonProject1.models.nhmm import *
from pythonProject1.data.data import *
from pythonProject1.data.collate_fn import *
from torch.utils.data import DataLoader
from pythonProject1.training import Small_WSJ_Dataset


if "__name__" == "__main__":
    # Make testing Dataset and DataLoader

    char_vocab = Char_vocabulary(small_training_sentences)
    char_V = len(char_vocab)
    tag_vocab = Tag_vocabulary(small_training_tags)
    K = len(tag_vocab)
    word_vocab = Word_vocabulary(small_training_sentences)
    word_V = len(word_vocab)

    testing_dataset = Small_WSJ_Dataset(raw_x=small_training_sentences,
                                        raw_y=small_training_tags,
                                        word_vocab=word_vocab,
                                        tag_vocab=tag_vocab)

    testing_dataloader = DataLoader(testing_dataset,
                                    batch_size=Nb,
                                    shuffle=False,
                                    collate_fn=collate_fn)

    # Load the trained models and set them for evaluation
    word_embedding_model = CNN_word_embedding(char_vocab, d, D, width)
    tag_embedding_model = Basic_tag_embedding(K, D)
    nhmm_model = NHMM(d, width, D, word_V, char_V, K, num_layers=1, dropout=0)

    word_embedding_model.eval()
    tag_embedding_model.eval()
    nhmm_model.eval()

    checkpoint = torch.load("models.pth")

    word_embedding_model.load_state_dict(checkpoint['word_embedding_model'])
    tag_embedding_model.load_state_dict(checkpoint['tag_embedding_model'])
    nhmm_model.load_state_dict(checkpoint['nhmm_model'])

    # Test predictions with loaded model
    sentence, targets = testing_dataset[0]
    print(sentence)
    print(targets)