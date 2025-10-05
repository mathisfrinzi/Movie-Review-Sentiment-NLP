import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('movie_reviews')

name = 'en_core_web_sm'

#import spacy
#try:
 #   nlp_en = spacy.load(name, disable=['ner', 'parser'])
#except:
    #import subprocess
    #import sys
    #!python -m spacy download "en_core_web_sm"
    #subprocess.check_call([sys.executable, '-m', 'spacy', 'download', name])
    #nlp_en = spacy.load(name, disable=['ner', 'parser'])

documents = [(movie_reviews.raw(fileid), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

corpus_raw = [ x[0] for x in documents ]
y_corpus = [ x[1] for x in documents ]

#Vectorial representation of the data
vectorizer = CountVectorizer(stop_words='english', max_features=15000, binary=False)
X_corpus = vectorizer.fit_transform(corpus_raw)

X_corpus = X_corpus.toarray()

#Partition of the data
PERCENT_TEST = 0.3
PERCENT_TRAIN_VAL = 1-PERCENT_TEST
PERCENT_VAL = 0.25
PERCENT_TRAIN = 1-PERCENT_VAL

N = X_corpus.shape[0]
Y_corpus = np.array([int(i=='pos') for i in y_corpus])

indices = np.arange(N)
random.shuffle(indices)
X_corpus = X_corpus[indices]
Y_corpus = Y_corpus[indices]

X_train_val = X_corpus[:int(N*PERCENT_TRAIN_VAL),:]
y_train_val = Y_corpus[:int(N*PERCENT_TRAIN_VAL),]

X_test = X_corpus[int(N*PERCENT_TRAIN_VAL):,:]
y_test = Y_corpus[int(N*PERCENT_TRAIN_VAL):,]

Nprime = X_train_val.shape[0]

X_train = X_train_val[:int(Nprime*PERCENT_TRAIN),:]
y_train = y_train_val[:int(Nprime*PERCENT_TRAIN),]

X_val = X_train_val[int(Nprime*PERCENT_TRAIN):,:]
y_val = y_train_val[int(Nprime*PERCENT_TRAIN):,]

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
C = [10**i for i in range(-10,20,1)]
score = []
for c in C:
  print('Entrainement du modèle pour une complexité : ',c)
  clf_lr = LogisticRegression(C=c)
  try:
    clf_lr.fit(X_train, y_train)
    score.append(accuracy_score(y_val, clf_lr.predict(X_val)))
  except:
    pass

print('The different scores are : ',score)
max_score = max(score)
bestC = C[score.index(max_score)]
print('The best complexity is : C =',bestC)
print('The best score is :',max_score*100,'%')

text_init = np.array(corpus_raw.copy())[indices][0]

## Interface

from tkinter import *
class Interface(Tk):
    def __init__(self, text_init = ''):
        global max_score, bestC, X_train, y_train
        Tk.__init__(self)
        self.title('Accuracy : {0}%. Write in english your comment : '.format(max_score*100))
        Label(self, text = "Write your comment about a film you appreciate")
        self.modele = LogisticRegression(C=bestC)
        self.modele.fit(X_train,y_train)
        text = Text(self,
            height=10,
            width=50,
            bg='lightgrey',
            fg='black',
            font=('Arial', 12),
            wrap='word',
        )
        text.insert('1.0', text_init)
        text.pack(padx=10, pady=10)
        self.lab = Label(self, text = " T")
        self.lab.pack()
        self.text_ = text
        self._after()
        self.mainloop()
    def _after(self):
        texte = self.text_.get("1.0",END)
        try:
            X_new = vectorizer.transform(np.array([texte]))
            if texte != '':
                value = (self.modele.predict(X_new))
                if value[0] == 0:
                    self.lab['bg'] = 'red'
                    comment = 'negative'
                else:
                    self.lab['bg'] = 'green'
                    comment = 'positive'
                self.lab['text'] = 'The tone of this review is predicted as {0}'.format(comment)
        except Exception as e:
            self.lab['text'] = 'ERROR : {0}'.format(e)
        self.after(10,self._after)


Interface(text_init)
