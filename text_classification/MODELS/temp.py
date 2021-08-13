from tqdm import tqdm
import numpy as np

def load_glove():
    embedding_index = {}
    with open('./glove.6B.100d.txt', encoding = 'utf8') as f:

        for i, line in tqdm(enumerate(f)):
            values = line.split()
            word = ''.join(values[0])
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    return embedding_index

if __name__=='__main__':
    
    embedding_index = {}
    with open('./glove.6B.100d.txt', encoding = 'utf8') as f:

        for i, line in tqdm(enumerate(f)):
            values = line.split()
            word = ''.join(values[0])
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            print(word, coefs.shape)
    #np.save('glove_embeddings.npy', embedding_index)
    
    #a= np.load('glove_embeddings.npy', allow_pickle=True)
    #print(a)
