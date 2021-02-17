from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from nltk import FreqDist
from flask import current_app
import pickle, re
import collections
import string

class SpamClassifier:
    word_features = None
    corpus = None

    def load_model(self, model_name):
        model_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'], model_name+'.pk')
        model_word_features_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'],model_name +'_word_features.pk')
        with open(model_file, 'rb') as mfp:
            self.classifier = pickle.load(mfp)
        with open(model_word_features_file, 'rb') as mwfp:
            self.word_features = pickle.load(mwfp)


    def extract_tokens(self, text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text, label)
         parameters:
                text: array of texts
                target: array of target labels

        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        tokens = []
        for i,j in zip(text,target):
                nopunc = "".join([char for char in i if char not in string.punctuation])
                clean_words = [word for word in nopunc.split() if word.isalpha() and len(word)>3]
                tup = (clean_words,j)
                tokens.append(tup)
        return tokens
            

    def get_features(self, corpus):
        """
        returns a Set of unique words in complete corpus.
        parameters:- corpus: tokenized corpus along with target labels

        Return Type is a set

        https://gist.github.com/baojie/6015546

        all_words = []
        for doc, label in corpus:
            all_words.extend(doc)
        word_distribution = FreqDist(all_words)
        return word_distribution.keys()
        """
        feature = []
        for i in corpus:
            txt = [feature.append(x.lower()) for x in i[0]]

        return set(feature)
        
       
        

    def extract_features(self, document):
        """
        maps each input text into feature vector
        parameters:- document: string

        Return type : A dictionary with keys being the train data set word features.
                      The values correspond to True or False
        """
        features={}
        doc_words = set(document)
        #features = {}
        #for word in self.word_features:
            #features['contains(%s)' %word] = (word in doc_words)
        features = {word:(word in doc_words) for word in self.word_features}
        return features

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        #assert corpus

        self.corpus = self.extract_tokens(text,labels)
        
        self.word_features = self.get_features(self.corpus)

        train_set = apply_features(self.extract_features, self.corpus)

        self.classifier = NaiveBayesClassifier.train(train_set)

        return self.classifier, self.word_features

    def predict(self, text):
        """
        Returns prediction labels of given input text.
        """
        if isinstance(text, (list)):
            pred = []
            for sentence in list(text):
                pred.append(self.classifier.classify(self.extract_features(sentence.split())))
            return pred
        if isinstance(text, (collections.OrderedDict)):
            pred = collections.OrderedDict()
            for label, sentence in text.items():
                pred[label] = self.classifier.classify(self.extract_features(sentence.split()))
            return pred
        return self.classifier.classify(self.extract_features(text.split()))


if __name__ == '__main__':

    print('Done')
