'''
Created on Jul 5, 2014

@author: The Queen
'''
from nltk import word_tokenize,WordNetLemmatizer,NaiveBayesClassifier,classify
from nltk.corpus import stopwords

train_data_file = open('r8-train-all-terms.txt','r')
train_data = train_data_file.read()
#test_data = open('..\\data\\r8-test-all-terms.txt')
stop_words = stopwords.words('english')
clean_documents = []

def extract_document_classes(raw_data):
    ## every line is a document
    document_class_vector = raw_data.split('\n')
    
    return document_class_vector;


def extract_documents(document_class_vector):
    documents = []
    for doc_cls in document_class_vector:
        
        ## there may be empty lines
        if(len(doc_cls) > 0):
            cls_text = doc_cls.split('\t')
            documents.append(cls_text[1])
            
    return documents;

def analyse_documents(documents):
    #for idx, val in enumerate(documents):
    word_counts = []
    doc_no_stop_word = []
    global stop_words,clean_documents
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
        
        #no_stop_word_tokens = word_tokenize(no_stop_word)
        doc_no_stop_word.append(len(no_stop_word_array))
        no_stop_word_docs.append(no_stop_word)
    
    #print 'fffff ' + str(len(no_stop_word_docs))
    
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
    
    clean_documents = lemmatized_docs

def create_class_dic(document_class_vector):
    document_class_dic = {}
    
    ## every line has a class + \t + text 
    for doc_cls in document_class_vector:
        
        ## there may be empty lines
        if(len(doc_cls) > 0):
            cls_text = doc_cls.split('\t')
            
            if(document_class_dic.has_key(cls_text[0]) == False):
                document_class_dic.update({cls_text[0]: [cls_text[1]]})
            else:
                document_class_dic[cls_text[0]].append(cls_text[1])
    
    return document_class_dic;


def extract_raw_information(data):
    ## tokenize text
    words_in_text = word_tokenize(data)
    
    ## lemmatize words to prevent any double workings
    wordlemmatizer = WordNetLemmatizer()
    wordtokens = [wordlemmatizer.lemmatize(word.lower()) for word in words_in_text]
    
    # remove stop words
    
    clean_words = {}
    
    for word in wordtokens:
        if word not in stop_words:
            clean_words[word] =  True
        
    ## TODO: SYNONYM EXPANSION
    
    
    ## create vector space model based on clean_words
    # for each word in clean_words find the counts
    

#extract_information(train_data)
#docs = extract_document_classes(train_data)
#r = prepare_document_classes(docs)

doc_cls = extract_document_classes(train_data)
docs = extract_documents(doc_cls)
#print len(docs)
analyse_documents(docs)
print 'e'





