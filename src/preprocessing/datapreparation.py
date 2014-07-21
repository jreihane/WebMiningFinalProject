'''
Created on Jul 5, 2014

@author: The Queen
'''
from nltk import word_tokenize,WordNetLemmatizer
from nltk.corpus import stopwords
import numpy
#import nltk
# from random import randrange

numpy.set_printoptions(threshold='nan')
# f = open('vec.txt', 'w')

# train_data_file = open('..\\data\\r8-train-all-terms.txt','r')
# train_data = train_data_file.read()
# train_data_file.close()

stop_words = stopwords.words('english')

cls_documents_dic = {}
cls_documents_dic_test_data = {}
# cls_documents_tuple = []
words_array = []
classes = []
documents = []
clean_documents = []

def create_test_data():
    global clean_documents,words_array,documents
    
    test_data = numpy.zeros((len(clean_documents),len(words_array)),int)
#     test_data = numpy.zeros((len(documents),len(words_array)),int)
#     test_data = numpy.zeros((len(words_array),len(documents)),int)
    
    for w_index,word in enumerate(words_array):
#         word_count = 0
        for d_index,doc in enumerate(clean_documents):
#             for d_index,doc in enumerate(documents):
#             word_count = word_count + doc.count(word)
            test_data[d_index][w_index] = doc.count(word)
        
#         test_data[w_index][d_index] = word_count/len(words_array)
#         test_data[d_index][w_index] = word_count/len(words_array)
        
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
    global documents_array, words_array
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
def create_vector_space(clean_documents_cls):
    global words_array,cls_documents_dic
    
    vector_space = numpy.zeros((len(clean_documents_cls.keys()),len(words_array)),int)
#     vector_space = numpy.zeros((len(words_array),len(clean_documents_cls.keys())),int)
    
    for w_index,word in enumerate(words_array):
        for c_index,cls in enumerate(cls_documents_dic.keys()):
            word_count = 0
            docs = cls_documents_dic[cls]
            for doc in docs:
                word_count = word_count + doc.count(word)
                vector_space[c_index][w_index] = vector_space[c_index][w_index] + doc.count(word)
#                 vector_space[w_index][c_index] = vector_space[w_index][c_index] + doc.count(word)
            
#             vector_space[w_index][c_index] = word_count
#             vector_space[c_index][w_index] = word_count

#     for i in range(0,15):
#         print '--------------------------------------------------------------------------'
# #         print 'word[i]: ' + words_array[i],'\n ', vector_space[i][:]
#         print vector_space[:][i]
#     print '--------------------------------------------------------------------------'
    
    print numpy.count_nonzero(vector_space)
    print len(vector_space)
    print len(vector_space[0])
    
    return vector_space

#---------------------------------------------------------------------------
def normalize_vector_space(vector_space):
    global words_array
    
    lenth = len(words_array)
    for (x,y), value in numpy.ndenumerate(vector_space):
        vector_space[x][y] = value/lenth
    
    return vector_space
#---------------------------------------------------------------------------

def create_naive_train_data(documents_classes):
    
    c_d_w_tuples = []
    for cls_doc in documents_classes:
        docs = documents_classes[cls_doc]
         
        w_dic = {}
        for doc in docs:
            words = word_tokenize(doc)
             
            for word in words:
                w_dic[word] = True
             
#             w_v_array.append(w_dic)
         
        c_d_w_tuples.append((w_dic,cls_doc))
        #cls_documents_tuple
        
    return c_d_w_tuples

def create_naive_test_data():
    
    test_data_file = open('..\\data\\r8-test-all-terms.txt')
    test_data = test_data_file.read()
    test_data_file.close()
    
    document_class_vector = test_data.split('\n')
    test_documents = []
    cls_documents_dic_test_data = {}
    
    for doc_cls in document_class_vector:
        if(len(doc_cls) > 0):
            cls_text = doc_cls.split('\t')
#             classes.append(cls_text[0])
            test_documents.append(cls_text[1])
            if(cls_documents_dic_test_data.has_key(cls_text[0])):
                cls_documents_dic_test_data[cls_text[0]].append(cls_text[1])
            else:
                cls_documents_dic_test_data[cls_text[0]] = [cls_text[1]]
    
    c_d_w_tuples = []
    for cls_doc in cls_documents_dic_test_data:
        docs = cls_documents_dic_test_data[cls_doc]
         
        w_dic = {}
        for doc in docs:
            words = word_tokenize(doc)
             
            for word in words:
                w_dic[word] = True
             
#             w_v_array.append(w_dic)
         
        c_d_w_tuples.append((w_dic,cls_doc))
        #cls_documents_tuple
        
    return c_d_w_tuples

