import pickle

UNK = "@@UNKNOWN@@"
END = "@end@"
START = "@start@"

def convert_glove():
    path_to_pickle_glove = "/home/junkyul/conda/neural-el_resources/glove.pkl"

    custom_vocab = "sentences_custom.txt"
    custom_glove = "glove_custom.txt"

    vocab_file = open(custom_vocab, 'w')
    glove_file = open(custom_glove, 'w')

    vocab_file.write(UNK+'\n')        # match first 3 lines to sentences.txt
    vocab_file.write(END+'\n')
    vocab_file.write(START+'\n')
    glove_vectors = pickle.load(open(path_to_pickle_glove, 'rb'))

    glove_file.write(UNK + ' ' + " ".join( [str(el) for el in glove_vectors['unk']]) + "\n")
    glove_file.write(END + ' ' + " ".join([str(el) for el in glove_vectors['eos']]) + "\n")
    glove_file.write(START + ' ' + " ".join([str(el) for el in glove_vectors['<s>']]) + "\n")

    for k in glove_vectors:
        if k in ['unk', '<s>', 'eos']:
            continue
        vocab_file.write(k + '\n')
        glove_file.write(k + ' ' + " ".join( [str(el) for el in glove_vectors[k]]) + "\n")


    vocab_file.close()
    glove_file.close()

def convert_coherence():
    path_to_pickle_coherences = "/home/junkyul/conda/neural-el_resources/vocab/cohstringG9_vocab.pkl"
    coherences = pickle.load(open(path_to_pickle_coherences, "rb"))

    custom_coherence = "coherences_custom.txt"
    coh_file = open(custom_coherence, 'w')
    coh_file.write(UNK + '\n')

    for el in coherences[1][1:]:        # list of coherence tokens
        coh_file.write(el + '\n')
    coh_file.close()

if __name__ == "__main__":
    convert_coherence()
