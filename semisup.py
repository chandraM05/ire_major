import os, sys, re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 128
VALIDATION_SPLIT = 0

val_text = []    
val_labels = []  

TEXT_DATA_DIR = "./../contains_all"

for name in sorted(os.listdir(TEXT_DATA_DIR)):
    fpath = os.path.join(TEXT_DATA_DIR, name)
    if(".txt" in fpath):
        print(fpath)
        
        if sys.version_info < (3,):
            f = open(fpath)
        else:
            f = open(fpath, encoding='latin-1')
        t = f.read()
        lines = t.split('\n')
        lines = lines[1:]
        #print(len(lines))
        for line in lines:
#            print cntline
            val_text.append(re.sub('    ',' ',line))
            val_labels.append(0)
        f.close()

val_text = np.array(val_text)
val_labels = np.array(val_labels)
np.save('val_text.npy', val_text)
np.save('val_labels.npy', val_labels)
print("saved")

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(val_text)
sequences = tokenizer.texts_to_sequences(val_text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(val_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

ssx_val = data[:]
ssy_val = labels[:]
np.save('ssx_val.npy', ssx_val)
np.save('ssy_val.npy', ssy_val)
