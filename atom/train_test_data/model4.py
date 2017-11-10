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

MAX_SEQUENCE_LENGTH = 512
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 128
VALIDATION_SPLIT = 0.2

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
print(sys.argv)
model_name = sys.argv[1]
b_size = int(sys.argv[2])
num_epochs = int(sys.argv[3])

print("model_name: ", model_name)
print("b_size: ", b_size)
print("num_epochs: ", num_epochs)
# Training steps

x_val_list = np.load('val_text.npy')
val_len = len(x_val_list)
print("val samples", val_len)

texts_text = np.array(texts_text)
x_val_list = np.array(x_val_list)
texts_text = np.vstack([texts_text, x_val_list])

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_text)
sequences = tokenizer.texts_to_sequences(texts_text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
ssval = data[-val_len:]
data = data[:len(data)-val_len]

labels = to_categorical(np.asarray(labels))  
print('Shape of data tensor:', data.shape)
print('Shape of val tensor:', ssval.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]
x_val = np.array(ssval)


#x_val = np.load('ssx_val.npy')

kernel_size = 5
filters = 64
pool_size = 4
# LSTM
lstm_output_size = 64
'''
model = Sequential()
# embedding layer can be made trainable. Look into it
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
#model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(len(labels_index), activation='softmax'))

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Bidirectional(LSTM(lstm_output_size)))
model.add(Dropout(0.5))
model.add(Dense(len(labels_index), activation='softmax'))
'''
embedding_layer = Embedding(MAX_NB_WORDS + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)(embedded_sequences)
x = MaxPooling1D(pool_size=pool_size)(x)
x = Bidirectional(LSTM(lstm_output_size))(x)
x = Dropout(0.5)(x)
preds = Dense(len(labels_index), activation='softmax')(x)

# try using different optimizers and different optimizer configs
# model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

learnrate = 0.1
decay = 1e-3
for j in xrange(6):
    model = Model(sequence_input, preds)
    learnrate = learnrate*0.5
    if j==0:
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    else:
        model.load_weights('models/'+model_name+'_'+str(b_size)+'_'+str(num_epochs)+'_'+'iter_'+str(j-1)+'_weights.h5')
        sgd = SGD(lr=learnrate, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=num_epochs, batch_size=b_size)
    
    score = model.evaluate(x_test, y_test, batch_size=b_size)
    print("score: ", score)
    # semisupervised
    print('Predicting')
    predicted = model.predict(x_val, batch_size=b_size)
    res=np.argmax(predicted, axis=1) 
    ll = []

    for i in xrange(len(res)):
        if predicted[i,res[i]]>0.9:
            ll.append(i)
    if len(ll)>0:
        b = np.zeros((len(ll),8))
        b[np.arange(len(ll)), res[ll]]=1
        x_train = np.vstack([x_train, x_val[ll]])
        y_train = np.vstack([y_train, b])    
        x_val = np.delete(x_val, (ll), axis=0)
        y_val = np.delete(y_val, (ll), axis=0)
    else:
        print("iter: ",j)
        print("no predictions with confidence more than thresold")

    model.save_weights('models/'+model_name+'_'+str(b_size)+'_'+str(num_epochs)+'_'+'iter_'+str(j)+'_weights.h5')
    del model
