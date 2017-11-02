from __future__ import print_function
import os, sys, re
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

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

## reading the train and test files
texts_title = []
texts_text = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
#TEXT_DATA_DIR = "/users/prateek.singhi/train_test_data"
TEXT_DATA_DIR = "./"
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
#------------------------------------------------------------

print(sys.argv)
model_name = "1"#sys.argv[1]
b_size = 128#int(sys.argv[2])
num_epochs = 1#int(sys.argv[3])

print("model_name: ", model_name)
print("b_size: ", b_size)
print("num_epochs: ", num_epochs)

# Training steps----------------------------------------------

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
x_test = np.array(x_val)
y_test = np.array(y_val)
## Uncomment to use pre defined word vectors
#embeddings_index = {}
#f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
#for line in f:
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#f.close()
#
#print('Found %s word vectors.' % len(embeddings_index))
#embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#for word, i in word_index.items():
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        embedding_matrix[i] = embedding_vector

#embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))/np.sqrt(len(word_index) + 1)
'''
sz = len(word_index)
embedding_matrix = np.random.randn(sz + 1, EMBEDDING_DIM)/np.sqrt(sz + 1)


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
'''
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

learnrate = 0.1
decay = 1e-6
for j in xrange(3):
    model = Model(sequence_input, preds)
    learnrate = learnrate*0.1
    if j==0:
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    else:
        model.load_weights('models/'+model_name+'_'+str(b_size)+'_'+str(num_epochs)+'_'+'iter_'+str(j-1)+'_weights.h5')
        sgd = SGD(lr=learnrate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=num_epochs, batch_size=b_size)
    
    score = model.evaluate(x_test, y_test, batch_size=b_size)
    print("score: ", score)
    print('Predicting')
    predicted = model.predict(x_val, batch_size=b_size)
    res=np.argmax(predicted, axis=1) 
    ll = []

    for i in xrange(len(res)):
        if predicted[i,res[i]]>0.8:
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


#------------------------------train phase 2-------------------------------

#things to try 


#score = model.evaluate(x_test, y_test, batch_size=16)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)


''' # saving and loading whole model - weights structure train params
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
'''


''' # saving and loading only the weights
model.save_weights('my_model_weights.h5')

model.load_weights('my_model_weights.h5')
If you need to load weights into a different architecture (with some layers in common), for instance for fine-tuning or transfer-learning, you can load weights by layer name:

model.load_weights('my_model_weights.h5', by_name=True)

for eg:
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # will be loaded
model.add(Dense(10, name='new_dense'))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
'''
