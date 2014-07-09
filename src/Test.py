'''
Created on Jul 3, 2014

@author: The Queen
'''
import nltk

x = "DFGDFg drget wec asea"
x_t = nltk.word_tokenize(x)
print len(x_t)

nltk.help.upenn_tagset(tagpattern=None)