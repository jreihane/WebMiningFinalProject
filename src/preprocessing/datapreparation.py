'''
Created on Jul 5, 2014

@author: The Queen
'''
from nltk import word_tokenize,WordNetLemmatizer,NaiveBayesClassifier,cluster,classify
from nltk.cluster import euclidean_distance,cosine_distance
from nltk.corpus import stopwords
from numpy import array
import numpy
import winsound
#import nltk
# from random import randrange

numpy.set_printoptions(threshold='nan')
# f = open('vec.txt', 'w')

train_data_file = open('r8-train-all-terms.txt','r')
train_data = train_data_file.read()
train_data_file.close()
#test_data = open('..\\data\\r8-test-all-terms.txt')
stop_words = stopwords.words('english')

cls_documents_dic = {}
# cls_documents_tuple = []
words_array = []
classes = []
documents = []
clean_documents = []

def create_test_data():
    global clean_documents,words_array
    
    test_data = numpy.zeros((len(clean_documents),len(words_array)),int)
    
    for d_index,doc in enumerate(clean_documents):
        words = word_tokenize(doc)
        
        for w_index,word in enumerate(words_array):
            if word in words:
                test_data[d_index][w_index] = test_data[d_index][w_index] + 1
            
#         i = i + 1
    
    return test_data

    
def extract_document_classes(raw_data):
    ## every line is a document
    document_class_vector = raw_data.split('\n')
    
    return document_class_vector;


def extract_documents(document_class_vector):
    global cls_documents_dic,classes,documents#,cls_documents_tuple
    
    for doc_cls in document_class_vector:
        
        ## there may be empty lines
        if(len(doc_cls) > 0):
            cls_text = doc_cls.split('\t')
#             classes.append(cls_text[0])
            documents.append(cls_text[1])
            if(cls_documents_dic.has_key(cls_text[0])):
                cls_documents_dic[cls_text[0]].append(cls_text[1])
            else:
                cls_documents_dic[cls_text[0]] = [cls_text[1]]
            
#             cls_documents_tuple.append((cls_text[1],cls_text[0]))
    
    classes = cls_documents_dic.keys()
    
    return documents;

# ---------------------------------- clean_analyse_documents --------------------------------------------
def clean_analyse_documents(documents):
    #for idx, val in enumerate(documents):
    word_counts = []
    doc_no_stop_word = []
    global stop_words,cls_documents_dic,clean_documents
    wordlemmatizer = WordNetLemmatizer()
    
#     print 'classes are: ', cls_documents_dic.keys()
    
    no_stop_word_docs_cls = {}
    for cls_doc in cls_documents_dic:
        no_stop_word_docs = []
        docs = cls_documents_dic[cls_doc]
        for doc in docs:
            # tokenizing and stop word removal
            tokens = word_tokenize(doc)
            no_stop_word_array = []
            no_stop_word = ''
            word_counts.append(len(tokens))
            for token in tokens:
                if token not in stop_words:
                    no_stop_word_array.append(token)
                    no_stop_word = no_stop_word + ' ' + token
            
            doc_no_stop_word.append(len(no_stop_word_array))
            no_stop_word_docs.append(no_stop_word)
        
        if no_stop_word_docs_cls.has_key(cls_doc):
            no_stop_word_docs_cls.append(no_stop_word_docs)
        else:
            no_stop_word_docs_cls[cls_doc] = no_stop_word_docs
    
    
    lemmatized_docs_cls = {}
    for cls_doc in no_stop_word_docs_cls:
        no_stop_word_docs = no_stop_word_docs_cls[cls_doc]
#         print no_stop_word_docs[0]
        # lemmatization
        lemmatized_docs = []
        lemmatized_docs_array = []
        for doc in no_stop_word_docs:
            words = word_tokenize(doc)
            lem_done_words = []
            doc_lemmatized = ''
            lemmatized_doc_array = []
            for word in words:
                if(word not in lem_done_words):
                    w = word.lower()
                    s = wordlemmatizer.lemmatize(w)
                    lem_done_words.append(s)
                    doc_lemmatized = doc_lemmatized + ' ' + s
                    
                    lemmatized_doc_array.append(doc_lemmatized)
                    
            lemmatized_docs.append(doc_lemmatized)
            lemmatized_docs_array.append(len(lemmatized_doc_array))
            
            clean_documents.append(doc_lemmatized)
        
        if lemmatized_docs_cls.has_key(cls_doc):
            lemmatized_docs_cls[cls_doc].append(lemmatized_docs)
        else:
            lemmatized_docs_cls[cls_doc] = lemmatized_docs
    
            
#     c_d_w_tuples = []
#     for cls_doc in lemmatized_docs_cls:
#         docs = lemmatized_docs_cls[cls_doc]
#         
#         w_dic = {}
#         w_v_array = []
#         for doc in docs:
#             words = word_tokenize(doc)
#             
#             for word in words:
#                 w_dic[word] = True
#             
#             w_v_array.append(w_dic)
#         
#         c_d_w_tuples.append((w_v_array,cls_doc))
#         #cls_documents_tuple
#         
#     print c_d_w_tuples[0]
#     print 'The result of analyzing documents is as follows: '
#     print 'the average number of tokens in documents is: ' + str((sum(word_counts)/len(documents)))
#     print 'maximum number of tokens in documents is: ' + str(max(word_counts))
#     print 'minimum number of tokens in documents is: ' + str(min(word_counts))
# 
#     print 'the average number of tokens in documents after removing stop words is:' + str((sum(doc_no_stop_word)/len(documents)))
#     print 'maximum number of tokens in documents after removing stop words is: ' + str(max(doc_no_stop_word))
#     print 'minimum number of tokens in documents after removing stop words is: ' + str(min(doc_no_stop_word))
#     
#     print 'the average of number of tokens in lemmatized documents is: ' + str((sum(lemmatized_docs_array)/len(documents)))
#     print 'maximum number of tokens after lemmatization is:' + str(max(lemmatized_docs_array))
#     print 'minimum number of tokens after lemmatization is:' + str(min(lemmatized_docs_array))
    
    return lemmatized_docs_cls
# ---------------------------------- end of clean_analyse_documents --------------------------------------------

    ## TODO: SYNONYM EXPANSION
    # ---------- !!!!!!!! ??????????????? !!!!!!!! --------------- 
    
def create_doc_word_index(documents_classes):
    global documents_array
    global words_array
    used_words = []
    for cls_doc in documents_classes:
        documents = documents_classes[cls_doc]
        for doc in documents:
            words = word_tokenize(doc)
            for word in words:
                if(word not in used_words):
                    words_array.append(word)
                    used_words.append(word)
    
## create vector space model based on clean_words
def create_vector_space(documents):
    global words_array,cls_documents_dic
    
    vector_space = numpy.zeros((len(cls_documents_dic.keys()),len(words_array)),int)
    
    i = 0
    for cls in cls_documents_dic.keys():
        docs = documents[cls]
        
        for doc in docs:
            doc_words = word_tokenize(doc)
            for w_index,word in enumerate(words_array):
                if word in doc_words:
                    vector_space[i][w_index] = vector_space[i][w_index] + 1
            
        i = i + 1
            
    print '--------------------------------------------------------------------------'
#     print vector_space
#     print '--------------------------------------------------------------------------'
    
    print numpy.count_nonzero(vector_space)
    print len(vector_space)
    print len(vector_space[0])
    
    return vector_space

# # call me when you start running the code!
# winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

# doc_cls = extract_document_classes(train_data)
# 
# docs = extract_documents(doc_cls)
# clean_documents = clean_analyse_documents(docs)
# 
# # vectors in python do not take strings as indices. so we have to define which document is doc no. 1 and so on
# # and which word is word no. 1 and so on
# create_doc_word_index(clean_documents)
# vector_space = create_vector_space(clean_documents)
# 
# vec2 = create_test_data() #array(vector_space[:][1])

# print 'classes are: ', classes
# 
# # 8 is the number of classes
# clusterer = cluster.kmeans.KMeansClusterer(8, cosine_distance, repeats=15, avoid_empty_clusters=True) 
#   
# clusters = clusterer.cluster(array(vector_space), True)
# 
# print '-------------------------- test 1 --------------------------------'
# # print clusterer.classify(vec2[0])
# cls = clusterer.classify(vec2[0][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + classes[cls]
# print '-----------------------------------------------------------------'
# 
# print '-------------------------- test 2 --------------------------------'
# # print clusterer.classify(vec2[0])
# cls = clusterer.classify(vec2[45][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + classes[cls]
# print '-----------------------------------------------------------------'
# 
# print '-------------------------- test 3 --------------------------------'
# # print clusterer.classify(vec2[0])
# cls = clusterer.classify(vec2[-7][:])
# print 'estimated class is: ' + str(cls) + ' which is: ' + classes[cls]
# print '-----------------------------------------------------------------'


#print clusterer.classify_vectorspace(array([45]))
#print 'cluster names are: ' , clusterer.cluster_names()
# print 'cluster means are: ', clusterer.means()
# print 'cluster_vectorspace() result: ', clusterer.cluster_vectorspace(vector_space, True)
# print 'cluster means are: ', clusterer.means()


# d1 = doc_cls[1:100]
# classifier = NaiveBayesClassifier.train(cls_documents_tuple[0:2])
# print '##############################################################'
# print 'accuracy: ', classify.accuracy(classifier,cls_documents_tuple[2:])

#print 'classify() result: ', clusterer.classify(vector_space)
#print 'cluster means are: ', clusterer.means()
#print 'accuracy: ', classify.accuracy(classifier,vector_space)


# print 'e'

# call me when you are done!!!!
# winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)