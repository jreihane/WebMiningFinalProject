'''
Created on Jul 5, 2014

@author: The Queen
'''
from nltk import word_tokenize,WordNetLemmatizer,NaiveBayesClassifier,cluster,classify
from nltk.cluster import euclidean_distance
from nltk.corpus import stopwords
import numpy
import nltk


train_data_file = open('r8-train-all-terms.txt','r')
train_data = train_data_file.read()
train_data_file.close()
#test_data = open('..\\data\\r8-test-all-terms.txt')
stop_words = stopwords.words('english')

cls_documents_dic = {}
words_array = []

    
def extract_document_classes(raw_data):
    ## every line is a document
    document_class_vector = raw_data.split('\n')
    
    return document_class_vector;


def extract_documents(document_class_vector):
    global cls_documents_dic
    documents = []
    
    for doc_cls in document_class_vector:
        
        ## there may be empty lines
        if(len(doc_cls) > 0):
            cls_text = doc_cls.split('\t')
            documents.append(cls_text[1])
            if(cls_documents_dic.has_key(cls_text[0])):
                cls_documents_dic[cls_text[0]].append(cls_text[1])
            else:
                cls_documents_dic[cls_text[0]] = [cls_text[1]]
            
    return documents;

# ---------------------------------- clean_analyse_documents --------------------------------------------
def clean_analyse_documents(documents):
    #for idx, val in enumerate(documents):
    word_counts = []
    doc_no_stop_word = []
    global stop_words
    wordlemmatizer = WordNetLemmatizer()
    
    #print len(documents)
    no_stop_word_docs = []
    for doc in documents:
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
        
    
#     for cls in cls_documents_dic:
#         docs = cls_documents_dic[cls]
#         i = 0
#         num_word = 0
#         for doc in docs:
#             doc_words = word_tokenize(doc)
#             for w_index, word in enumerate(doc_words):
#                 num_word = num_word + doc_words.count(word)
# #                 print num_word
#                 vector_space[i][w_index] = num_word
#         i = i + 1
        
    print 'The result of analyzing documents is as follows: '
    print 'the average number of tokens in documents is: ' + str((sum(word_counts)/len(documents)))
    print 'maximum number of tokens in documents is: ' + str(max(word_counts))
    print 'minimum number of tokens in documents is: ' + str(min(word_counts))

    print 'the average number of tokens in documents after removing stop words is:' + str((sum(doc_no_stop_word)/len(documents)))
    print 'maximum number of tokens in documents after removing stop words is: ' + str(max(doc_no_stop_word))
    print 'minimum number of tokens in documents after removing stop words is: ' + str(min(doc_no_stop_word))
    
    print 'the average of number of tokens in lemmatized documents is: ' + str((sum(lemmatized_docs_array)/len(documents)))
    print 'maximum number of tokens after lemmatization is:' + str(max(lemmatized_docs_array))
    print 'minimum number of tokens after lemmatization is:' + str(min(lemmatized_docs_array))
    
    return lemmatized_docs
# ---------------------------------- end of clean_analyse_documents --------------------------------------------

    ## TODO: SYNONYM EXPANSION
    # ---------- !!!!!!!! ??????????????? !!!!!!!! --------------- 
    
def create_doc_word_index(documents):
    global documents_array
    global words_array
    used_words = []
    for doc in documents:
        words = word_tokenize(doc)
        for word in words:
            if(word not in used_words):
                words_array.append(word)
                used_words.append(word)
        
## create vector space model based on clean_words
def create_vector_space(documents):
    global words_array
    
    vector_space = numpy.zeros((len(words_array) + 1,len(documents) + 1),int)
#     for w_index, word in enumerate(words_array):
#         for d_index, doc in enumerate(documents):
#             vector_space[w_index][d_index] = doc.count(word)
    
    for cls in cls_documents_dic:
        docs = cls_documents_dic[cls]
        i = 0
        num_word = 0
        for doc in docs:
            doc_words = word_tokenize(doc)
            for w_index, word in enumerate(doc_words):
                num_word = num_word + doc_words.count(word)
#                 print num_word
                vector_space[i][w_index] = num_word
        i = i + 1
            
        
    print '--------------------------------------------------------------------------'
    print vector_space
    print '--------------------------------------------------------------------------'
    
    print numpy.count_nonzero(vector_space)
    print len(vector_space)
    print len(vector_space[0])
    
    return vector_space


doc_cls = extract_document_classes(train_data)
docs = extract_documents(doc_cls)
clean_documents = clean_analyse_documents(docs)
# vectors in python do not take strings as indices. so we have to define which document is doc no. 1 and so on
# and which word is word no. 1 and so on
create_doc_word_index(clean_documents)
vector_space = create_vector_space(clean_documents)

# classifier = NaiveBayesClassifier.train(train_set)
# clusterer = cluster.kmeans.KMeansClusterer(5, euclidean_distance, repeats=10, avoid_empty_clusters=True) 
#  
# clusters = clusterer.cluster(vector_space, True)#.accuracy(classifier,vector_space)
# print 'clusters are: ', clusters
# print 'cluster names are: ' , clusterer.cluster_names()
# print 'cluster means are: ', clusterer.means()
# print 'cluster_vectorspace() result: ', clusterer.cluster_vectorspace(vector_space, True)
# print 'cluster means are: ', clusterer.means()


d1 = doc_cls[1:100]
classifier = NaiveBayesClassifier.train(vector_space)
print 'accuracy: ', classify.accuracy(classifier,doc_cls[100:])

#print 'classify() result: ', clusterer.classify(vector_space)
#print 'cluster means are: ', clusterer.means()
#print 'accuracy: ', classify.accuracy(classifier,vector_space)


# a = len(words_array) + 100
# b = len(docs) + 1
# print a , b
# vector_space = numpy.zeros((a,b),int)
# print vector_space[len(words_array)][len(docs)]

# test()
# test2()
print 'e'





