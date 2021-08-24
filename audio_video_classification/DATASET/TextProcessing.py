import numpy as np
import re
import string
from nltk.corpus import stopwords
from collections import Counter
import os
import csv
from embed_model import Embed
import torch

class MakeString:
    def process(self, text):
        return str(text)

class ReplaceBy:
    def __init__(self, replace_by):
        self.replace_by = replace_by

    def process(self, text):
        for replace, by in self.replace_by:
            text = text.replace(replace, by)
        return text

class LowerText:
    def process(self, text):
        return text.lower()

class ReduceTextLength:
    def __init__(self, limited_text_length):
        self.limited_text_length = limited_text_length
    def process(self, text):
        return text[:self.limited_text_length]

class VectorizeText:
    def __init__(self):
        pass
    def process(self, text):
        return text.split()

class FilterPunctuation:
    def __init__(self):
        pass
    def process(self, words_vector):
        reg_exp_filter_rule = re.compile("[%s]"%re.escape(string.punctuation))
        words_vector = [reg_exp_filter_rule.sub("",word) for word in words_vector]
        return words_vector

class FilterNonalpha:
    def __init__(self):
        pass
    def process(self, words_vector):
        words_vector = [word for word in words_vector if word.isalpha()]
        return words_vector

class FilterStopWord:
    def __init__(self, language):
        self.language = language
    def process(self, words_vector):
        stop_words = set(stopwords.words(self.language))
        words_vector = [word for word in words_vector if not word in stop_words]
        return words_vector

class FilterShortWord:
    def __init__(self, min_length):
        self.min_length = min_length
    def process(self, words_vector):
        words_vector = [word for word in words_vector if len(word) > self.min_length]
        return words_vector

class TextProcessor:
    def __init__(self, processor_list):
        self.processor_list = processor_list
    def process(self, text):
        for processor in self.processor_list:
            text = processor.process(text)
        return text

class VocabularyHelper:
    def __init__(self, textProcessor):
        self.textProcessor = textProcessor
        self.vocabulary = Counter()
    def update(self, text):
        words_vector = self.textProcessor.process(text=text)
        self.vocabulary.update(words_vector)
    def get_vocabulary(self):
        return self.vocabulary

def text_preprocessing(text):


        #for text in textlist:
        #with open('text.txt','r') as f:
        text_len = np.vectorize(len)
        text_length = text_len(text)

        makeString = MakeString()
        replace_by = [("."," "), ("?"," "), (","," "), ("!"," "),(":"," "),(";"," ")]
        replaceBy =ReplaceBy(replace_by=replace_by)

        lowerText = LowerText()

        FACTOR=8
        reduceTextLength = ReduceTextLength(limited_text_length=500)

        vectorizeText = VectorizeText()
        filterPunctuation = FilterPunctuation()
        filterNonalpha = FilterNonalpha()
        filterStopWord = FilterStopWord(language = "english")

        min_length = 2
        filterShortWord = FilterShortWord(min_length=min_length)
        processor_list_1 = [makeString,
                replaceBy,
                lowerText,
                reduceTextLength,
                vectorizeText,
                filterPunctuation,
                filterNonalpha,
                filterStopWord,
                filterShortWord]

        textprocessor1 = TextProcessor(processor_list=processor_list_1)
        voca = textprocessor1.process(text = text)
        vocabularyHelper = VocabularyHelper(textProcessor = textprocessor1)
        
        vocabularyHelper.update(text)

        vocabularies = [key for key,_ in vocabularyHelper.get_vocabulary().items()]
        #vocabularies = ' '.join([key for key,_ in vocabularyHelper.get_vocabulary().items()])
        #print(vocabularies)
        return vocabularies


if __name__=='__main__':
    
    path = '/mnt/data/guest0/uchan/DATA2'
    textlist = []
    for moviename in os.listdir(path):
        csv_path = os.path.join(path, moviename, 'data.csv')
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            rdr = csv.reader(f)
            for idx, line in enumerate(rdr):
                if idx==0:
                    continue
                textlist.append(text_preprocessing(line[5]))
    #print(textlist)
    total_vocab = []
    for text in textlist:
        #print(len(text))
        for t in text:
            total_vocab.append(t)
    word_to_idx = {word : i for i, word in enumerate(total_vocab)}
    model = Embed(len(total_vocab))
    for text in textlist:
        inputs = [word_to_idx[word] for word in text]
        print(inputs)
        inputs = torch.tensor(inputs, dtype = torch.long)
        output = model(inputs)
        print(output)
