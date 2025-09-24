small_training_sentences = ['The big company rises',
                            'The market is strong',
                            'The company fell',
                            'The company rises']

small_training_tags = [['DT', 'ADJ', 'NN', 'VBZ'],
                       ['DT', 'NN', 'VBZ', 'ADJ'],
                       ['DT', 'NN', 'VBD'],
                       ['DT', 'NN', 'VBZ']]


raw_sentences = [
    'The big company rises', 'The market is strong', 'The company fell', 'The company rises', 'A new trend emerges',
    'Global markets are volatile', 'The firm expanded rapidly', 'Investors bought shares today', 'The index dropped significantly', 'Our revenue increased',
    'The small business grows', 'Stock prices fluctuated wildly', 'The economy appears stable', 'The fund\'s value declined', 'A risky investment failed',
    'The corporation announced layoffs', 'Their profit margin improved', 'A sudden crisis struck', 'The bank\'s rates are low', 'The CEO resigned',
    'The stock market rebounded', 'A financial report was positive', 'The government intervened', 'The sector shows growth', 'A large competitor appeared',
    'The economy is growing slowly', 'Company earnings were high', 'A new product launched', 'Their business thrives', 'The debt level decreased',
    'This market is unpredictable', 'The corporation acquired another company', 'Their strategy paid off', 'A major deal was signed', 'The consumer confidence rose',
    'The market capitalization soared', 'A potential bubble formed', 'The interest rates changed', 'A positive outlook is shared', 'The company is stable',
    'The stock exchange opened', 'New regulations were enacted', 'The industry is competitive', 'A global downturn began', 'The company\'s stock rose',
    'The business is expanding', 'The firm\'s assets increased', 'A sudden sell-off occurred', 'Their earnings report was weak', 'The company\'s future is bright',
    'The economy recovered quickly', 'A new competitor entered', 'The market is now calm', 'Their stock value dropped', 'The company announced a merger',
    'The stock market closed', 'The economic climate is good', 'A bullish trend started', 'The profit margins improved', 'The company\'s shares are up',
    'The market is bearish', 'The firm\'s outlook is grim', 'A recession is coming', 'The government\'s policy changed', 'The business is struggling',
    'The company\'s revenue declined', 'A financial crisis emerged', 'The stock price fell again', 'The market is stagnant', 'The company is failing',
    'The market is volatile again', 'A new CEO was appointed', 'The company\'s debt increased', 'The profit is falling', 'The economy is weak',
    'The business is booming', 'A strategic partnership was formed', 'The company\'s value is high', 'Their profit is rising', 'The economy is strong',
    'The market is recovering fast', 'The firm is doing well', 'A new report was released', 'The company\'s performance is good', 'The market is stable now',
    'The stock market is up', 'A financial scandal was exposed', 'The company\'s stock is low', 'Their earnings are down', 'The economy is in trouble',
    'The business is suffering', 'A new investment was made', 'The company\'s debt is high', 'The profit is down', 'The economy is contracting',
    'The market is turbulent', 'A new business model emerged', 'The company\'s valuation soared', 'Their profit is excellent', 'The economy is healthy',
    'The business is expanding globally', 'A new acquisition was completed', 'The company\'s sales are up', 'Their profit is growing', 'The economy is thriving'
]

raw_tags = [
    ['DT', 'ADJ', 'NN', 'VBZ'], ['DT', 'NN', 'VBZ', 'ADJ'], ['DT', 'NN', 'VBD'], ['DT', 'NN', 'VBZ'], ['DT', 'ADJ', 'NN', 'VBZ'],
    ['ADJ', 'NNS', 'VBP', 'ADJ'], ['DT', 'NN', 'VBD', 'RB'], ['NNS', 'VBD', 'NNS', 'RB'], ['DT', 'NN', 'VBD', 'RB'], ['PRP$', 'NN', 'VBD'],
    ['DT', 'ADJ', 'NN', 'VBZ'], ['NN', 'NNS', 'VBD', 'RB'], ['DT', 'NN', 'VBZ', 'ADJ'], ['DT', 'NN', 'POS', 'NN', 'VBD'], ['DT', 'ADJ', 'NN', 'VBD'],
    ['DT', 'NN', 'VBD', 'NNS'], ['PRP$', 'NN', 'NN', 'VBD'], ['DT', 'ADJ', 'NN', 'VBD'], ['DT', 'NN', 'POS', 'NNS', 'VBP', 'ADJ'], ['DT', 'NN', 'VBD'],
    ['DT', 'NN', 'VBD'], ['DT', 'ADJ', 'NN', 'VBD', 'ADJ'], ['DT', 'NN', 'VBD'], ['DT', 'NN', 'VBZ', 'NN'], ['DT', 'ADJ', 'NN', 'VBD'],
    ['DT', 'NN', 'VBZ', 'VBG', 'RB'], ['NN', 'NNS', 'VBD', 'ADJ'], ['DT', 'ADJ', 'NN', 'VBD'], ['PRP$', 'NN', 'VBZ'], ['DT', 'NN', 'NN', 'VBD'],
    ['DT', 'NN', 'VBZ', 'ADJ'], ['DT', 'NN', 'VBD', 'DT', 'NN'], ['PRP$', 'NN', 'VBD', 'RP'], ['DT', 'ADJ', 'NN', 'VBD', 'VBN'], ['DT', 'NN', 'NN', 'VBD'],
    ['DT', 'NN', 'NN', 'VBD'], ['DT', 'ADJ', 'NN', 'VBD'], ['DT', 'NN', 'NNS', 'VBD'], ['DT', 'ADJ', 'NN', 'VBZ', 'VBN'], ['DT', 'NN', 'VBZ', 'ADJ'],
    ['DT', 'NN', 'VBD'], ['ADJ', 'NNS', 'VBD', 'VBN'], ['DT', 'NN', 'VBZ', 'ADJ'], ['DT', 'ADJ', 'NN', 'VBD'], ['DT', 'NN', 'POS', 'NN', 'VBD'],
    ['DT', 'NN', 'VBZ', 'VBG'], ['DT', 'NN', 'POS', 'NNS', 'VBD'], ['DT', 'ADJ', 'NN', 'VBD'], ['PRP$', 'NNS', 'NN', 'VBD', 'ADJ'], ['DT', 'NN', 'POS', 'NN', 'VBZ', 'ADJ'],
    ['DT', 'NN', 'VBD', 'RB'], ['DT', 'ADJ', 'NN', 'VBD'], ['DT', 'NN', 'VBZ', 'RB', 'ADJ'], ['PRP$', 'NN', 'NN', 'VBD'], ['DT', 'NN', 'VBD', 'DT', 'NN'],
    ['DT', 'NN', 'VBD'], ['DT', 'ADJ', 'NN', 'VBZ', 'ADJ'], ['DT', 'ADJ', 'NN', 'VBD'], ['DT', 'NN', 'NNS', 'VBD'], ['DT', 'NN', 'POS', 'NNS', 'VBP', 'RP'],
    ['DT', 'NN', 'VBZ', 'ADJ'], ['DT', 'NN', 'POS', 'NN', 'VBZ', 'ADJ'], ['DT', 'NN', 'VBZ', 'VBG'], ['DT', 'NN', 'POS', 'NN', 'VBD'], ['DT', 'NN', 'VBZ', 'VBG'],
    ['DT', 'NN', 'POS', 'NN', 'VBD'], ['DT', 'ADJ', 'NN', 'VBD'], ['DT', 'NN', 'NN', 'VBD'], ['DT', 'NN', 'VBZ', 'ADJ'], ['DT', 'NN', 'VBZ', 'VBG'],
    ['DT', 'NN', 'VBZ', 'ADJ', 'RB'], ['DT', 'ADJ', 'NN', 'VBD', 'VBN'], ['DT', 'NN', 'POS', 'NN', 'VBD'], ['DT', 'NN', 'VBZ', 'VBG'], ['DT', 'NN', 'VBZ', 'ADJ'],
    ['DT', 'NN', 'VBZ', 'VBG'], ['DT', 'ADJ', 'NN', 'VBD', 'VBN'], ['DT', 'NN', 'POS', 'NN', 'VBZ', 'ADJ'], ['PRP$', 'NN', 'VBZ', 'VBG'], ['DT', 'NN', 'VBZ', 'ADJ'],
    ['DT', 'NN', 'VBZ', 'VBG', 'RB'], ['DT', 'NN', 'VBZ', 'VBG'], ['DT', 'ADJ', 'NN', 'VBD', 'VBN'], ['DT', 'NN', 'POS', 'NN', 'VBZ', 'ADJ'], ['DT', 'NN', 'VBZ', 'ADJ', 'RB'],
    ['DT', 'NN', 'VBZ', 'RP'], ['DT', 'ADJ', 'NN', 'VBD', 'VBN'], ['DT', 'NN', 'POS', 'NN', 'VBZ', 'ADJ'], ['PRP$', 'NNS', 'VBP', 'RP'], ['DT', 'NN', 'VBZ', 'IN', 'NN'],
    ['DT', 'NN', 'VBZ', 'VBG'], ['DT', 'ADJ', 'NN', 'VBD', 'VBN'], ['DT', 'NN', 'POS', 'NN', 'VBZ', 'ADJ'], ['DT', 'NN', 'VBZ', 'VBG'], ['DT', 'NN', 'VBZ', 'VBG'],
    ['DT', 'NN', 'VBZ', 'ADJ'], ['DT', 'ADJ', 'NN', 'VBD'], ['DT', 'NN', 'POS', 'NN', 'VBD'], ['PRP$', 'NN', 'VBZ', 'ADJ'], ['DT', 'NN', 'VBZ', 'ADJ'],
    ['DT', 'NN', 'VBG', 'RB'], ['DT', 'ADJ', 'NN', 'VBD', 'VBN'], ['DT', 'NN', 'POS', 'NNS', 'VBP', 'RP'], ['PRP$', 'NN', 'VBZ', 'VBG'], ['DT', 'NN', 'VBZ', 'VBG']
]


training_sentences = []
training_tags = []

for sentence, annotation in zip(raw_sentences, raw_tags):
    tokenized_sentence = sentence.split()
    if len(tokenized_sentence) == len(annotation):
        training_sentences.append(sentence)
        training_tags.append(annotation)