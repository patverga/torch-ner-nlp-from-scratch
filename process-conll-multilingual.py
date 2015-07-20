__author__ = 'pv'

window = 2
vocabs = [{},{}]
labels = {}
out_dir = 'data/multilingual/'
vocab_map_file = out_dir + 'vocab-map.index'
label_map_file = out_dir + 'label-map.index'
vector_out_file = 'data/embeddings/multilingual.index'
w2v_files = ['data/embeddings/en_input_embeddings.txt', 'data/embeddings/es_input_embeddings.txt']
in_dirs = ['data/conll2003/', 'data/conll2002/']
en_files = ['eng.testa', 'eng.testb', 'eng.train']
es_files = ['esp.testa', 'esp.testb', 'esp.train']
in_files = [en_files, es_files]


# create token -> index map
offset = 0
out = open(vector_out_file, 'w')
for i in range(0,2):
    vocab = vocabs[i]
    for line in open(w2v_files[i], 'r'):
        parts = line.split('\t')
        token = parts[0]
        vocab[token] = str(len(vocab) + 1 + offset)
        vector = parts[1]
        out.write(vocab[token] + '\t' + vector)
    offset += len(vocab)
out.close()

for vocab in vocabs:
    vocab['<UNK>'] = offset+1
    vocab['<PAD>'] = offset+2
    vocab['<S>'] = offset+3
    vocab['<\S>'] = offset+4
print ('loaded ' + str(offset) + ' tokens to vocab')


offset = 0
for i in range(0, 2):
    vocab = vocabs[i]
    # iterate over train, dev, test files for conll
    for in_file in in_files[i]:
        print 'Processing ' + in_file
        out_file = out_dir + in_file + '.index'

        tokens = []
        chunks = []
        ner = []
        labeled_windows = []

        for line in open(in_dirs[i] + in_file, 'r'):
            line = line.strip()
            if not line.startswith('-DOCSTART-'):
                if line != '':
                    parts = line.split(' ')
                    tokens.append(parts[0])
                    chunks.append(parts[2])
                    if parts[-1] not in labels:
                        labels[parts[-1]] = str(len(labels) + 1)
                    ner.append(labels[parts[-1]])
                # new sentence
                else:
                    # process the last sentence into labeled windows
                    for j in range(0, len(tokens)):
                        # each line starts with label \t
                        current_window = [ner[j] + '\t']
                        for k in range(j - window, j + window + 1):
                            if k < -1 or k > len(tokens):
                                token = '<PAD>'
                            elif k == -1:
                                token = '<S>'
                            elif k == len(tokens):
                                token = '<\S>'
                            else:
                                token = tokens[k]
                            token_idx = vocab[token] if token in vocab else vocab['<UNK>']
                            current_window.append(str(token_idx))
                        labeled_windows.append(' '.join(current_window))

                    tokens = []
                    chunks = []
                    ner = []

        # write the windows to file
        out = open(out_file, 'w')
        for w in labeled_windows:
            out.write(w + '\n')
        out.close()

        offset += len(vocabs[i])

# export maps
out = open(vocab_map_file, 'w')
for i in range(0,2):
    for t, k in vocabs[i].iteritems():
        out.write(t + '\t' + str(k) +'\n')
out.close()

out = open(label_map_file, 'w')
for l, i in labels.iteritems():
    out.write(l + '\t' + str(i) +'\n')
out.close()

