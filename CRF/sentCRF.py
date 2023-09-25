# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:32:48 2023

@author: Shailesh
"""

import pycrfsuite

from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from sklearn.metrics import classification_report

def readCorpus(fpath):
    sents = []
    with open(fpath) as fd:
        sent = []
        for l in fd:
            #lt = l.strip().decode("utf8")
            lt = l.strip()
            if not lt:
                sents.append(sent)
                sent = []
            else:
                w_t = lt.split('\t')
                sent.append([w_t[0], w_t[1]])
    return sents

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        #'bias',
        'word.lower=' + word.lower(),
        #'word[-3:]=' + word[-3:],
        #'word[-2:]=' + word[-2:],
        #'word.isupper=%s' % word.isupper(),
        #'word.istitle=%s' % word.istitle(),
        #'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        #'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        pass
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            #'-1:word.istitle=%s' % word1.istitle(),
            #'-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            #'-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        pass
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            #'+1:word.istitle=%s' % word1.istitle(),
            #'+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            #'+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, label in sent]
def sent2tokens(sent):
    return [token for token, label in sent]


train_sents = readCorpus("test.txt")
x_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]


print("start append train set.")
trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(x_train, y_train):
    trainer.append(xseq, yseq)
print("append train set done.")
trainer.set_params({
    'c1': 1.0,
    'c2': 1e-3,
    'max_iterations': 500,
    'feature.possible_transitions': True
    })

trainer.train("trained_model")
test_object = pycrfsuite.Tagger()
test_object.open("trained_model")


#Test on the first sentence of the training data
#First load the trained model
tagger = pycrfsuite.Tagger()
tagger.open('trained_model')

#Read the first sentence
example_sent = train_sents[0]
print(' '.join(sent2tokens(example_sent)), end='\n\n')

#Predict and test
print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))

#This will give you the precision, recall, F-score
def sentiment_classification_report(y_true, y_pred):
 
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

##Evaluation of the labeller
y_pred = [tagger.tag(xseq) for xseq in x_train]
print(sentiment_classification_report(y_train, y_pred))