'''
Created on Jul 17, 2014

@author: The Queen
'''
from src.preprocessing import datapreparation
from nltk import cluster
from nltk.cluster import euclidean_distance,cosine_distance
# import nltk
import numpy
# from sklearn.neighbors import KNeighborsClassifier
from numpy import array
import random
import winsound
import gc

numpy.set_printoptions(threshold='nan')

# train_data_file = open('r8-train-all-terms.txt','r')
# train_data = train_data_file.read()
# train_data_file.close()
# doc_cls = datapreparation.extract_document_classes(train_data)
# 
# docs = datapreparation.extract_documents(doc_cls)
# clean_documents = datapreparation.clean_analyse_documents(docs)
# 
# # vectors in python do not take strings as indices. so we have to define which document is doc no. 1 and so on
# # and which word is word no. 1 and so on
# datapreparation.create_doc_word_index(clean_documents)

def clusterdata(clean_documents):
    vector_space = datapreparation.create_vector_space(clean_documents)
    vector_space2 = datapreparation.normalize_vector_space(vector_space)
    
    vec2 = datapreparation.create_test_data() 
    
    l = len(datapreparation.documents)
    # 8 is the number of classes
    clusterer = cluster.kmeans.KMeansClusterer(8, cosine_distance, repeats=1, avoid_empty_clusters=True)
    # clusterer = cluster.kmeans.KMeansClusterer(8, euclidean_distance, repeats=10, avoid_empty_clusters=True)  
    #   
    clusters = clusterer.cluster(array(vector_space2), True)
    for i in range(0,10):
        print datapreparation.documents[i]
        cls = clusterer.classify(vec2[i][:])
        print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
        
    for i in range(100,140):
        print datapreparation.documents[i]
        cls = clusterer.classify(vec2[i][:])
        print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
    
    for i in range(l/15,l/14):
        print datapreparation.documents[i]
        cls = clusterer.classify(vec2[i][:])
        print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
# print '-------------------------- test 1 --------------------------------'
# # print clusterer.classify(vec2[0])
# print datapreparation.documents[0]
# cls = clusterer.classify(vec2[0][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
# # cls2 = clusterer.classify(vec2[:][0])
# # print 'estimated class is: ' + str(cls2) + ' which is: ' + datapreparation.classes[cls2]
# print '-----------------------------------------------------------------'
# 
# print '-------------------------- test 2 --------------------------------'
# # print clusterer.classify(vec2[0])
# print datapreparation.documents[45]
# cls = clusterer.classify(vec2[45][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
# # cls2 = clusterer.classify(vec2[:][45])
# # print 'estimated class is: ' + str(cls2) + ' which is: ' + datapreparation.classes[cls2]
# print '-----------------------------------------------------------------'
# 
# print '-------------------------- test 3 --------------------------------'
# # print clusterer.classify(vec2[0])
# print datapreparation.documents[-20]
# cls = clusterer.classify(vec2[-20][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
# # cls2 = clusterer.classify(vec2[:][-20])
# # print 'estimated class is: ' + str(cls2) + ' which is: ' + datapreparation.classes[cls2]
# # print 'expected: ' + 
# print '-----------------------------------------------------------------'
# 
# print '-------------------------- test 4 --------------------------------'
# # print clusterer.classify(vec2[0])
# print datapreparation.documents[-6]
# cls = clusterer.classify(vec2[-6][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
# print '-----------------------------------------------------------------'
# 
# print '-------------------------- test 5 --------------------------------'
# # print clusterer.classify(vec2[0])
# print datapreparation.documents[1]
# cls = clusterer.classify(vec2[1][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
# print '-----------------------------------------------------------------'
# 
# print '-------------------------- test 6 --------------------------------'
# # print clusterer.classify(vec2[0])
# print datapreparation.documents[62]
# cls = clusterer.classify(vec2[62][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
# print '-----------------------------------------------------------------'
# 
# print '-------------------------- test 7 --------------------------------'
# # print clusterer.classify(vec2[0])
# print datapreparation.documents[63]
# cls = clusterer.classify(vec2[63][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
# print '-----------------------------------------------------------------'

#print clusterer.classify_vectorspace(array([45]))
#print 'cluster names are: ' , clusterer.cluster_names()
# print 'cluster means are: ', clusterer.means()
# print 'cluster_vectorspace() result: ', clusterer.cluster_vectorspace(vector_space, True)
# print 'cluster means are: ', clusterer.means()

# print '----------------------------naive bayes------------------------------'
# naive_train_data = datapreparation.create_naive_train_data(clean_documents)
# naive_test_data = datapreparation.create_naive_test_data()
# print naive_data[-1]
# d1 = doc_cls[1:100]
#gc.collect()
# naive_data2 = naive_train_data
# random.shuffle(naive_data2)
# edge = (len(naive_data)/3) * 2
# classifier = NaiveBayesClassifier.train(naive_train_data)
# print '##############################################################'
# print 'accuracy: ', classify.accuracy(classifier,naive_test_data)
# print classifier.most_informative_features()
# print 'classify() result: ', clusterer.classify(vector_space)
#print 'cluster means are: ', clusterer.means()
#print 'accuracy: ', classify.accuracy(classifier,vector_space)


    print 'e'
    
    # call me when you are done!!!!
    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)