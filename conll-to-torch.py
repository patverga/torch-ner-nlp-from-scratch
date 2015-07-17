__author__ = 'pv'

window = 2
vocab = {}
labels = {}
w2v_file = 'polyglot-en.w2v'
in_dir = '/home/pv/data/conll-2003/'
out_dir = './'

# create token -> index map
with open(w2v_file) as f:
    next(f)
    for line in f:
        token = line.split(' ')[0]
        vocab[token] = str(len(vocab) + 1)
print ('loaded ' + str(len(vocab)) + ' tokens to vocab')

# iterate over train, dev, test files for conll
for f in ['eng.testa', 'eng.testb', 'eng.train']:
    print 'Processing ' + f
    in_file = in_dir + f
    out_file = out_dir + f + '.torch'

    tokens = []
    chunks = []
    ner = []
    labeled_windows = []

    for line in open(in_file, 'r'):
        line = line.strip()
        if not line.startswith('-DOCSTART-'):
            if line != '':
                parts = line.split(' ')
                tokens.append(parts[0])
                chunks.append(parts[2])
                if parts[3] not in labels:
                    labels[parts[3]] = str(len(labels) + 1)
                ner.append(labels[parts[3]])
            # new sentence
            else:
                # process the last sentence into labeled windows
                for i in range(0, len(tokens)):
                    # each line starts with label \t
                    current_window = [ner[i] + '\t']
                    for j in range(i - window, i + window + 1):
                        if j < -1 or j > len(tokens):
                            token = '<PAD>'
                        elif j == -1:
                            token = '<S>'
                        elif j == len(tokens):
                            token = '<\S>'
                        else:
                            token = tokens[j]
                        token_idx = vocab[token] if token in vocab else vocab['<UNK>']
                        current_window.append(token_idx)
                    labeled_windows.append(' '.join(current_window))

                tokens = []
                chunks = []
                ner = []

    # write the windows to file
    out = open(out_file, 'w')
    for w in labeled_windows:
        out.write(w + '\n')
    out.close()