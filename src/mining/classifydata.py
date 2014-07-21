'''
Created on Jul 20, 2014

@author: The Queen
'''
# import datapreparation
from nltk import NaiveBayesClassifier,classify
from src.preprocessing import datapreparation

# print '----------------------------naive bayes------------------------------'
classifier = []

def train(clean_documents):
    global classifier
    
    naive_train_data = datapreparation.create_naive_train_data(clean_documents)
    classifier = NaiveBayesClassifier.train(naive_train_data)
    
def test():
    naive_test_data = datapreparation.create_naive_test_data()
# print naive_data[-1]
# d1 = doc_cls[1:100]
#gc.collect()
# naive_data2 = naive_train_data
# random.shuffle(naive_data2)
# edge = (len(naive_data)/3) * 2

# print '##############################################################'
    print 'accuracy: ', classify.accuracy(classifier,naive_test_data)
    print classifier.most_informative_features()
    print classifier.show_most_informative_features()