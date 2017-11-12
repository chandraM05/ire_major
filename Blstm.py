from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

# chandra
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import csv
import operator
from keras.utils.np_utils import to_categorical


max_features = 200000
maxlen = 100
batch_size = 32

train_category=[];train=[];test_category=[];test=[]


print('Loading data...')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

dict_class={}
dict_class['DEMO']=1
dict_class['DISE']=2
dict_class['FAML']=3
dict_class['GOAL']=4
dict_class['PREG']=5
dict_class['SOCL']=6
dict_class['TRMT']=7
# dict = {'DEMO': 1, 'DISE': 2, 'TRMT': 3,'GOAL': 4, 'PREG': 5, 'FAML': 6,'SOCL': 7}

dictionary={}
word_count=0

with open('/home/chandra/3rdsem/ire/train_test_data/ICHI2016-TrainData.tsv','rb') as tsvin, open('word2vec.txt', 'wb') as csvout:
    tsvin = csv.reader(tsvin, delimiter='\t')
    csvout = csv.writer(csvout)

    for row in tsvin: #title,category etc
        break
    
    for row in tsvin:
        train_category.append(dict_class[row[0]])
        example_sent=row[1].decode('utf8')+" "+row[2].decode('utf8')
        # print (example_sent)
        word_tokens = word_tokenize(example_sent)

        temp=[]
        for w in word_tokens:
		    if w not in stop_words:
		    	word=ps.stem(w)
		    	if word in dictionary :
		    		index=dictionary[word]
		    	else :
		    		dictionary[word]=word_count
		    		index=word_count
		    		word_count+=1
		        temp.append(index)
        train.append(np.array(temp))

with open('/home/chandra/3rdsem/ire/train_test_data/new_ICHI2016-TestData_label.tsv','rb') as tsvin, open('word2vec.txt', 'wb') as csvout:
    tsvin = csv.reader(tsvin, delimiter='\t')
    csvout = csv.writer(csvout)
    
    for row in tsvin:
        test_category.append(dict_class[row[0]])
        example_sent=row[1].decode('utf8')+" "+row[2].decode('utf8')
        word_tokens = word_tokenize(example_sent)

        temp=[]
        for w in word_tokens:
		    if w not in stop_words:
		    	word=ps.stem(w)
		    	if word in dictionary :
		    		index=dictionary[word]
		    	else :
		    		dictionary[word]=word_count
		    		index=word_count
		    		word_count+=1
		        temp.append(index)
        test.append(np.array(temp))


x_train=np.array(train)
x_test=np.array(test)
y_train=to_categorical(np.array(train_category))
y_test=to_categorical(np.array(test_category))


print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print (type(x_test[0]),type(x_train[0]),type(y_train[0]),type(y_test[0]))

# for i in x_test :
# 	print (i,type(i))
# 	break
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

# try using different optimizers and different optimizer configs
# model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,batch_size=40,epochs=10)

scores = model.evaluate(x_test,y_test , verbose=0)
print (scores)

pred=model.predict(x_test, batch_size=1)
for i in pred :
    print (i)
# print (pred,len(pred),len(pred[0]))
# , verbose=0, steps=None
