
# coding: utf-8

# In[25]:

import pandas as pd
from sklearn.neural_network import MLPClassifier
#train  = open("ICHI2016-TrainData.tsv","r")
#print(train.readline())
#print(train.readline())


# In[26]:

DataFram=pd.DataFrame.from_csv('ICHI2016-TrainData.tsv', sep='\t')


# In[ ]:

for ind,rows in DataFram.iterrows():
    print(ind)
    f1.write(str(ind)+"\n")


# In[45]:

'''f=open("train","w")
counti=0
f1=open("trainingClasses","w")
for ind,rows in DataFram.iterrows():
    #if counti<2:
    #    print(rows[0]+" "+rows[1])
    #counti+=1
    f.write(rows[0]+" "+rows[1]+"\n")
    f1.write(str(ind)+"\n")
f.close()
f1.close()'''
d={}
d['SOCL']=1
d['TRMT']=2
d['DEMO']=3
d['FAML']=4
d['PREG']=5
d['GOAL']=6
d['DISE']=7


# In[62]:

f=open("MetaMapClasses","r")
f1=open("Classes","w")
for r in f:
    f1.write(r.split("|")[0]+"\n")
f1.close()


# In[68]:

f=open("Classes","r")
li=[]
for r in f:
    li.append(r.strip("\n"))


# In[69]:

f=open("train1","r")
c=[]
for r in f:
    if(len(r)==0):
        continue
    c.append(r)
print(len(c))


# In[71]:

f=open("train1","r")
f2=open("trainingClasses","r")
f1=open("Classes.txt","w")
counti=0
for r in c:
    r=r.strip("\n").strip(" ")
    #if len(r)==0:
    #    continue
    vect=[]
    for i in range(0,133):
        vect.append(0)
    i=0
    for ct in li:
        #print(c)
        vect[i]=r.count(ct)
        i+=1
    s=f2.readline()
    s=str(d[s.strip("\n")])+","
    for i in vect:
        s=s+str(i)+","
    f1.write(s[:-1]+"\n")
    counti+=1
print(counti)
f1.close()


# In[72]:

f=open("Classes.txt","r")
x=[]
y=[]
for r in f:
    s=r.split(",")
    l=[]
    y.append(int(s[0]))
    for k in range(1,len(s)):
        l.append(int(s[k]))
    x.append(l)
print(len(x))


# In[73]:

clf=MLPClassifier(activation='relu', alpha=1e-05,
        epsilon=1e-08, hidden_layer_sizes=(25,26),
       learning_rate_init=0.0001,solver='lbfgs', tol=0.00001)
clf.fit(x, y)


# In[74]:

f=clf.predict_proba([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0
]])
print(f)


# In[75]:

queries=[]
counti=0;
f=open("queries","r")
for r in f:
    r=r.split(",")
    l=[]
    for k in r:
        l.append(int(k))
    queries.append(l)


# In[33]:




# In[87]:

pre=0
while len(queries)!=pre:
    pre=len(queries)
    cla=[0,0,0,0,0,0,0,0]
    clases=[]
    for i in range(0,7):
        l=[]
        clases.append(l)
    for i in range(0,len(queries)):
        f=clf.predict_proba([queries[i]])
        for ki in f:
            for k in range(0,len(ki)):
                if ki[k]>=.5:
                    clases[k].append(i)
                    cla[k]+=1
                    break
    mini=min(cla)
    maxi=max(cla)
    print(maxi)
    for i in range(0,mini):
        for ys in range(0,len(clases[i])):
            x.append(queries[clases[i][ys]])
            y.append(ys)
            del queries[clases[i][ys]]
    clf.fit(x, y)


# In[81]:

print(len(x))


# In[ ]:
