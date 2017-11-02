from __future__ import print_function
import os, sys, re
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras
import keras.utils
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Embedding
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
batch_size = 32

## reading the train and test files
texts_title = []
texts_text = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
TEXT_DATA_DIR = "/users/prateek.singhi/train_test_data"
currcnt = 0
cntline = 1
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    fpath = os.path.join(TEXT_DATA_DIR, name)
    if(".tsv" in fpath):
#        print fpath
        if sys.version_info < (3,):
            f = open(fpath)
        else:
            f = open(fpath, encoding='latin-1')
        t = f.read()
        lines = t.split('\n')
        lines = lines[1:]
        for line in lines:
#            print cntline
            cntline = cntline + 1
            linebreakup = line.split('\t')
            if len(linebreakup) > 0:
                if linebreakup[0] not in labels_index:
                    labels_index[linebreakup[0]] = currcnt + 1
                    currcnt = currcnt + 1
                if len(linebreakup) == 3:
                    texts_text.append(re.sub('    ',' ',linebreakup[2]))
                    texts_title.append(re.sub('    ',' ',linebreakup[1]))
                    labels.append(labels_index[linebreakup[0]])
        f.close()
#----------------------------------------------

# Training steps

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_text)
sequences = tokenizer.texts_to_sequences(texts_text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(len(labels_index), activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
t1 = time.time()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=12,
          validation_data=[x_val, y_val])
t2 = time.time()
print(t2-t1)
