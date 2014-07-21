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
# import random
# import gc

numpy.set_printoptions(threshold='nan')

def clusterdata(clean_documents):
    
    print 'creating vector space'
    vector_space = datapreparation.create_vector_space(clean_documents)
    
    print 'normalizing vectorspace'
    vector_space2 = datapreparation.normalize_vector_space(vector_space)
    
#     l = len(datapreparation.documents)

    # 8 is the number of classes
    print 'start the clusterer'
    clusterer = cluster.kmeans.KMeansClusterer(8, cosine_distance, repeats=10, avoid_empty_clusters=True)
    # clusterer = cluster.kmeans.KMeansClusterer(8, euclidean_distance, repeats=10, avoid_empty_clusters=True)  
    
    print 'now cluster vector space'
    clusters = clusterer.cluster(array(vector_space2), True)
    
    print '\n ', numpy.count_nonzero(clusters)
    
    print 'create test data'
    vec2 = datapreparation.create_test_data()
    
    print 'normalize test data'
    vec2_normalized = datapreparation.normalize_vector_space(vec2)

    print 'now test the clusterer'
    for i in range(0,len(vec2)):
#         cls = clusterer.classify(vec2[i][:])
        cls = clusterer.classify(vec2_normalized[i][:])
#         print vec2[i][:]
        print 'estimated class is: ', cls, '\n'
#     for i in range(0,10):
#         print datapreparation.documents[i]
#         cls = clusterer.classify(vec2[i][:])
#         print '\nestimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
#         
#     for i in range(100,140):
#         print datapreparation.documents[i]
#         cls = clusterer.classify(vec2[i][:])
#         print '\nestimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
#     
#     for i in range(l/15,l/12):
#         print datapreparation.documents[i]
#         cls = clusterer.classify(vec2[i][:])
#         print '\nestimated class is: ' + str(cls) + ' which is: ' + datapreparation.classes[cls]
        
        
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

    print 'e'
    
    