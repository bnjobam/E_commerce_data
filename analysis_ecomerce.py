import scipy
import xgboost
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
import pickle
#https://www.kaggle.com/retailrocket/ecommerce-dataset/download
events=pd.read_csv('../ecommerce-dataset/events.csv')
item_prop1=pd.read_csv('../ecommerce-dataset/item_properties_part1.csv')
item_prop2=pd.read_csv('../ecommerce-dataset/item_properties_part2.csv')

item_prop=pd.concat([item_prop1,item_prop2], axis=0)

def get_data(table):
    views=table.loc[table.event!='transaction']
    views=views.sort_values(by=['visitorid','itemid','timestamp'])
#     addtocart=table.loc[table.event=='addtocart']
    visitors=list(set(views.visitorid))

    items=list(set(views.itemid))
    viewtable=[]
    addtocart=[]

    for i in tqdm(range(len(visitors))):

#         rowv[0]=visitors[i]
#         print(visitors[i])
        temp=views.loc[views.visitorid==visitors[i]]
        temp.index=list(range(temp.shape[0]))
        v_items=list(set(temp.itemid))
        for j in range(len(v_items)):
            temp1=temp.loc[temp.itemid==v_items[j]]
            temp1.index=list(range(temp1.shape[0]))
            for r in range(temp1.shape[0]):
                rowv=np.zeros((6))
                rowv[0]=v_items[j]
                rowv[1]=temp1.timestamp[r]
                rowv[2]=temp1.timestamp[r]-temp1.timestamp[0]
                rowv[3]=r+1
                wd,we=0,0
                for k in range(r+1):
                    if datetime.utcfromtimestamp(temp1.timestamp[k]/1000).strftime('%a') in ['Mon','Tue','Wed', 'Thu']:
                        wd+=1
                    else:
                        we+=1
                rowv[4]=wd
                rowv[5]=we

                if temp1.event[r]=='addtocart':
                    addtocart.append(1)

                else:
                    addtocart.append(0)
                viewtable.append(rowv)
    return viewtable,addtocart

def one_hot_encode(y):
    N = len(y)
    K = len(set(y))

    Y = np.zeros((N,K))

    for i in range(N):
        Y[i,y[i]] = 1

    return Y

data,target=get_data(events)


#generating more variables
avail_itemprop=item_prop.loc[item_prop.property=='available']
cavail=[]
for r in tqdm(range(len(data.itemid)), desc='Appending...'):
    tavail=avail_itemprop.loc[avail_itemprop.itemid==data.iloc[r,1]].sort_values(by=['timestamp'])
    if tavail.shape[0]==0:
        cavail.append(2)#0 for not available and 1 otherwise
    else:
        for i in range(tavail.shape[0]):
            if data.iloc[r,2]<tavail.iloc[i,0]:
                cavail.append(tavail.iloc[i-1,3])
                break
            elif i==tavail.shape[0]-1:
                cavail.append(tavail.iloc[-1,3])



data=pd.DataFrame(data, columns=['itemid','timestamp', 'timestamp-first_visit_timestamp', 'number of views so far','weekdaycount', 'weekend count'])
data['view/addtocart(1)']=target
data['availability']=cavail
data=data.drop('Unnamed: 0', axis=1)
data.to_csv('transformed_events.csv')


n=6  #the approximate number if view items to include

target=np.array(target)
ll=[i for i in range(len(target)) if target[i]==1]
l2=[i for i in range(len(target)) if train.availability[i]!=2]
tr=list(set([i for i in range(int(n*len(ll)))]).intersection(set(l2)))+list(set(ll).intersection(set(l2)))*n
trdata=data.iloc[tr,[2,3,4,5,7]]
trtarget=target[tr]
trainX,testX,trainY,testY=train_test_split(trdata,trtarget,test_size=0.15, random_state=1)

model=xgboost.XGBClassifier(n_jobs=-1,learning_rate=0.2, verbosity=1)
xgmodel=model.fit(trainX,trainY)

Y=xgmodel.predict(testX)
xgmodel.score(testX,testY)
#0.79
cm=one_hot_encode(testY).T.dot(one_hot_encode(Y))
cm=pd.DataFrame(cm, columns=['pred_view','pred_addtocart'], index=['true_view','true_addtocart'])
cm.loc['class_accuracy',:]=[cm.iloc[i,i]/cm.iloc[:,i].sum() for i in range(cm.shape[1])]
cm.to_csv('confusion_matrix.csv')


classifier=open('xgboost_model_veiw_addtocart_classifier.py', 'wb')
pickle.dump(xgmodel,classifier)
classifier.close()
