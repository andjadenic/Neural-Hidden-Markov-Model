training_sentences = ['The big company rises',
                      'The company fell',
                      'A company rises']
training_tags = [['DT', 'ADJ', 'NN', 'VBZ'],
                        ['DT', 'NN', 'VBD'],
                        ['DT', 'NN', 'VBZ']]
q = ['The company fell']

if __name__ == "__main__":
    tokens = training_sentences[0].split()
    print(tokens)