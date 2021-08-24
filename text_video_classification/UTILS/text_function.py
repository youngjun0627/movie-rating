import csv
from .preprocessing_text import Tokenizer
import numpy as np
import os
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def get_tokenizer2(plot):
    ps = PorterStemmer()
    plot = re.sub('[^a-zA-Z]', ' ', plot)
    plot = plot.lower()
    plot = plot.split()
    plot = [ps.stem(word) for word in plot if not word in stopwords.words('english')]
    return plot

def process_glove_line(line, dim):
    word = None
    embedding = None

    try:
        splitLine = line.split()
        word = " ".join(splitLine[:len(splitLine)-dim])
        embedding = np.array([float(val) for val in splitLine[-dim:]])
    except:
        print(line)

    return word, embedding

def load_glove_model(glove_filepath, dim):
    with open(glove_filepath, encoding="utf8" ) as f:
        content = f.readlines()
        model = {}
        for line in content:
            word, embedding = process_glove_line(line, dim)
            if embedding is not None:
                model[word] = embedding
        return model


def text_func():
    plots = []
    DATASET_PATH = '/home/uchanlee/uchanlee/uchan_dataset/movie_id_plots_synopsis.csv'
    for line in csv.reader(open(DATASET_PATH, 'r', encoding='utf-8-sig')):
        moviename, id, plot, synopsis = line
        if synopsis=='\n///\n///':
            continue
        plots.append(synopsis.split('///')[0])
   
    #tokenizer = get_tokenizer('basic_english')
    #counter = Counter()
    ps = PorterStemmer()
    corpus = []
    for plot in plots:
        plot = re.sub('[^a-zA-Z]', ' ', plot)
        plot = plot.lower()
        plot = plot.split()
        plot = [ps.stem(word) for word in plot if not word in stopwords.words('english')]
        plot = ' '.join(plot)
        corpus.append(plot)
    '''
    # when 6B.100d
    EMBEDDING_FILE = '/home/uchanlee/uchanlee/uchan/text_classification/UTILS/glove.6B.100d.txt'
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
    #print(embeddings_index.values())
    '''
    
    EMBEDDING_FILE = '/home/uchanlee/uchanlee/uchan/text_classification/UTILS/glove.pickle'
    if os.path.join(EMBEDDING_FILE):
        embeddings_index = pickle.load(open(EMBEDDING_FILE, 'rb'))
    else:
        EMBEDDING_FILE = '/home/uchanlee/uchanlee/uchan/text_classification/UTILS/glove.840B.300d.txt'
        embeddings_index = load_glove_model(EMBEDDING_FILE, 300)
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    voc_size=10000 # Vocabulary size
    embed_size=300 #word vector size

    tokenizer = Tokenizer(num_words =10000)
    tokenizer.fit_on_texts(list(corpus))
    word_index = tokenizer.word_index #Total words in the corpus
    nb_words = min(voc_size, len(word_index))
    #Initialize weight matrix for embedding layer
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) 
    embedding_matrix = np.pad(embedding_matrix, ((1,0),(0,0)), 'constant', constant_values = 0)
    for word, i in word_index.items():
        if i >= voc_size: continue #Skip the words if vocab size is reached
        embedding_vector = embeddings_index.get(word) #Extract the pretrained values from GloVe
        if embedding_vector is not None: embedding_matrix[i+1] = embedding_vector

    #Finding max words
    l = 0
    for x in corpus:
            l = max(l,len(x.split(' ')))

    #Padding the sequences for input
    sent_length= l

    return word_index, embedding_matrix, nb_words+1 # vocab, embedding_weight

if __name__=='__main__':
    EMBEDDING_FILE = '/home/uchanlee/uchanlee/uchan/text_classification/UTILS/glove.840B.300d.txt'
    a = load_glove_model(EMBEDDING_FILE, 300)
    
    with open('glove.pickle', 'wb') as fw:
        pickle.dump(a,fw)
