import numpy as np

class VocabularyHandler():
    #handle the vocabulary, load a pretrained word embedding file

    def __init__(self, path_to_word_embedding_file, add_empty_and_unkown_tokens = True):
        self.add_empty_and_unkown_tokens = add_empty_and_unkown_tokens
        self.vocabulary, self.embedding_matrix, self.embedding_dimension = self.load_embedding_file(path_to_word_embedding_file=path_to_word_embedding_file)
        self.word_to_id_dictionnary = self.hashtable_from_vocab(self.vocabulary)
        print ('Length of vocabulary, including <UNK>, <EOF> and <EMPTY> token: ' + str(len(self.vocabulary)))
        print ('Embedding dimension: '+str(self.embedding_dimension))

    def load_embedding_file(self, path_to_word_embedding_file):
        #return an array representing the vocabulary in the word_embedding_file_path, a matrix of floats corresponding to the pretrained embedding of the vocabulary and the embedding dimension
        vocabulary = []
        embedding_matrix = []

        print ('loading ' + path_to_word_embedding_file + '...')

        with open(path_to_word_embedding_file) as f:

            #catch the embedding dim
            word_embedding_dimension = len(f.readline().strip().split(' ')) - 1
            f.seek(0) #rewind the file pointer to the beginning

            if self.add_empty_and_unkown_tokens:
                # add the unknown token with values set to zeros
                vocabulary.append('<UNK>')
                embedding_matrix.append(np.asarray(np.zeros(word_embedding_dimension, dtype=float)))

                # add the enf of file token <EOF>
                vocabulary.append('<EOF>')
                embedding_matrix.append(np.asarray(np.zeros(word_embedding_dimension, dtype=float)))

                # add the empty token <EMPTY> after the EOF token
                vocabulary.append('<EMPTY>')
                embedding_matrix.append(np.asarray(np.zeros(word_embedding_dimension, dtype=float)))

            #read the vocabulary and the associated embeddings
            for line in f.readlines():
                row = line.strip().split(' ')
                vocabulary.append(row[0])
                word_embedding = []
                for i in range(len(row) - 1):
                    word_embedding.append(float(row[i+1]))
                embedding_matrix.append(word_embedding)


        print (path_to_word_embedding_file + ' loaded.')

        return vocabulary, np.asarray(embedding_matrix), word_embedding_dimension

    def hashtable_from_vocab(self, vocabulary_array):
        #return a dictionnary matching a 'word' to a token id
        word_to_id_dictionnary = dict()
        for index, word in enumerate(vocabulary_array):
            word_to_id_dictionnary[word] = index
        return word_to_id_dictionnary