'''
Created on Jul 20, 2014

@author: The Queen
'''
# import preprocessing.datapreparation
from src.preprocessing import datapreparation
from src.mining import clusterdata,classifydata

train_data_file = open('r8-train-all-terms.txt','r')
train_data = train_data_file.read()
train_data_file.close()
doc_cls = datapreparation.extract_document_classes(train_data)

print 'classes are: ', datapreparation.classes

docs = datapreparation.extract_documents(doc_cls)
clean_documents = datapreparation.clean_analyse_documents(docs)

# vectors in python do not take strings as indices. so we have to define which document is doc no. 1 and so on
# and which word is word no. 1 and so on
datapreparation.create_doc_word_index(clean_documents)

print '---------------------cluster------------------------'
clusterdata.cluster(clean_documents)
# print '----------------------------------------------------'

# print '---------------------classify------------------------'
# classifydata.train(clean_documents)
# classifydata.test()