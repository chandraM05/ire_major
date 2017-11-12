
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA

from nltk.stem import PorterStemmer
import csv
import operator


train_category=[]
train =[]
# unlabelled =[]
test_category=[]
pred_category=[]
test=[]

# Preprocessing the data
with open('/home/chandra/3rdsem/ire/train_test_data/ICHI2016-TrainData.tsv','rb') as tsvin, open('word2vec.txt', 'wb') as csvout:
    tsvin = csv.reader(tsvin, delimiter='\t')
    csvout = csv.writer(csvout)

    for row in tsvin: #title,category etc
        break
    
    for row in tsvin:
        train_category.append(row[0])
        train.append(row[1]+" "+row[2])

with open('/home/chandra/3rdsem/ire/train_test_data/new_ICHI2016-TestData_label.tsv','rb') as tsvin, open('word2vec.txt', 'wb') as csvout:
    tsvin = csv.reader(tsvin, delimiter='\t')
    csvout = csv.writer(csvout)
    
    for row in tsvin:
        test_category.append(row[0])
        test.append(row[1]+" "+row[2])


# f = open('/home/chandra/3rdsem/ire/proj/bigfile.txt','r')
# unlabelled = f.readlines()
# f.close()
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# np.append(arr, [[7, 8, 9]], axis=0)
# print arr,type(arr),type([[7, 8, 9]])


#  cosine similarity

stopWords = stopwords.words('english')

vectorizer = CountVectorizer(stop_words = stopWords)
transformer = TfidfTransformer()

# unlabelledVectorizerArray = vectorizer.fit_transform(unlabelled).toarray()
trainVectorizerArray = vectorizer.fit_transform(train).toarray()
testVectorizerArray = vectorizer.transform(test).toarray()

print type(trainVectorizerArray)

cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)


# labellinng the unlabelled messages 
f = open('/home/chandra/3rdsem/ire/proj/files.txt','r') 
files = f.readlines()
f.close()

for file in files :
    unlabelled=[]
    f = open('/home/chandra/3rdsem/ire/contains_all/'+file.strip(),'r')
    unlabelled = f.readlines()
    f.close()

    unlabelledVectorizerArray = vectorizer.transform(unlabelled).toarray()
    for i in unlabelledVectorizerArray :
        print len(i),len(trainVectorizerArray[0])
        break

    for i in range(0,len(unlabelledVectorizerArray)):
        simiraities=[]
        for vector in trainVectorizerArray:
            cosine = cx(vector, unlabelledVectorizerArray[i])
            simiraities.append(cosine)
        
        top100cs= sorted(range(len(simiraities)), key=lambda i: simiraities[i])[-100:]
        freq={}
        maxfreq=0
        for j in top100cs :
            if train_category[j] not in freq :
                freq[train_category[j]]=1
            else :
                freq[train_category[j]]=freq[train_category[j]]+1
                if maxfreq < freq[train_category[j]] :
                    maxfreq = freq[train_category[j]]
                    
        if maxfreq >= 50 :
            pred_class=""
            for key in freq :
                if freq[key] == maxfreq :
                    pred_class = key
            add=[]
            add.append(unlabelledVectorizerArray[i])
            np.append(trainVectorizerArray,add, axis=0)
            train_category.append(pred_class)
            print pred_class,maxfreq
    print len(trainVectorizerArray)


# test data processing
count=0
for i in range(0,len(testVectorizerArray)):
    simiraities=[]
    for vector in trainVectorizerArray:
        cosine = cx(vector, testVectorizerArray[i])
        simiraities.append(cosine)
    
    top100cs= sorted(range(len(simiraities)), key=lambda i: simiraities[i])[-100:]
    # print top100cs
    freq={}
    maxfreq=0
    for j in top100cs :
        if train_category[j] not in freq :
            freq[train_category[j]]=1
        else :
            freq[train_category[j]]=freq[train_category[j]]+1
            if maxfreq < freq[train_category[j]] :
                maxfreq = freq[train_category[j]]
    pred_class=""
    for key in freq :
        if freq[key] == maxfreq :
            pred_class = key

    if pred_class == test_category[i] :
        count+=1
        print maxfreq, pred_class , test_category[i],count,i
print "accuracy :"+float(count)/len(test_category)
