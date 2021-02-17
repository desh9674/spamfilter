from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
import pandas as pd
import pickle
import collections
from sklearn.model_selection import train_test_split
import string

#from sklearn.feature_extraction.text import CounterVectorizer
##########  THis is just the SAMPLE MODEL CREATION PROGRAM ########
# lot of work with flask is needed along with this to complete the capstone project  

#word_features =None


class SpamClassifier:

    word_features = None
    classifier = None

    def extract_tokens(self, text, target):   
        """
        returns array of tuples where each tuple is defined by (tokenized_text, label)
        parameters:
                text: array of texts
                target: array of target labels
                
        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        corpus=[]
        for i,j in zip(text,target):
            nopunc = "".join([char for char in i if char not in string.punctuation])
            clean_words = [word for word in nopunc.split() if word.isalpha() and len(word)>3]
            tup = (clean_words,j)
            corpus.append(tup)
        
        
        return corpus
        
    
    def get_features(self, corpus):
        """ 
        returns a Set of unique words in complete corpus. 
        parameters:- corpus: tokenized corpus along with target labels (i.e)the ouput of extract_tokens function.
        
        Return Type is a set
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
        #word_features = CounterVectorizer().fit_transform(document)
        #iterate through the word_features to find if the doc_words contains it or not
        t = {word:(word in doc_words) for word in self.word_features}
                
        return t#features
        

    def train(self, text, labels):
        
        """
        Returns trained model and set of unique words in training data
        also set trained model to 'self.classifier' variable and set of 
        unique words to 'self.word_features' variable.
        """

        #call extract_tokens
        self.corpus= self.extract_tokens(text, labels)
                        
        #call get_features
        self.word_features= self.get_features(self.corpus)
                        
        #Extracting training set
        train_set = apply_features(self.extract_features, self.corpus)
                        
        #Now train the NaiveBayesClassifier with train_set
        self.classifier = NaiveBayesClassifier.train(train_set)
                
        return self.classifier, self.word_features



    def predict(self, text):

        """
        Returns prediction labels of given input text. 
        Allowed Text can be a simple string i.e one input email, a list of emails, or a dictionary of emails identified by their labels.
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
    
    data = pd.read_csv('emails.csv')
    train_X, test_X, train_Y, test_Y = train_test_split(data["text"].values,
                                                            data["spam"].values,
                                                            test_size = 0.25,
                                                            random_state = 50,
                                                            shuffle = True,
                                                            stratify=data["spam"].values)
    classifier = SpamClassifier()
    classifier_model, model_word_features = classifier.train(train_X, train_Y)
    model_name = 'spam_classifier_model.pk'
    model_word_features_name = 'spam_classifier_model_word_features.pk'
    with open(model_name, 'wb') as model_fp:
        pickle.dump(classifier_model, model_fp)
    with open(model_word_features_name, 'wb') as model_fp:
        pickle.dump(model_word_features, model_fp)
    print('DONE')
    
