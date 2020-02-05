import xgboost
import pandas as pd
import numpy as np
from tqdm import tqdm
from datatime import datatime
from sklearn.model_selection import train_test_split
import pickle


data=pd.read_csv('transformed_events.csv')
events=pd.read_csv('../ecommerce-dataset/events.csv')

def modify_data(table,items_per_day):
    views=table.sort_values(by=['visitorid','itemid','timestamp'])
    visitors=list(set(views.visitorid))

    items=sorted(list(set(views.itemid)))
    viewtable=[]

    for i in tqdm(range(len(visitors))):

        temp=views.loc[views.visitorid==visitors[i]]
        v_items=len((temp.itemid))
        time=max(max(temp.timestamp)-min(temp.timestamp),1) #because we will be d1viding
        add=temp.loc[temp.event=='addtocart'].shape[0]
        trans=temp.loc[temp.event=='transaction'].shape[0]
        value=v_items*100000*24/time
        cvalue=(value<=items_per_day)*0.001+add*0.75+trans
        cvalue=(cvalue>=0.001)*1
        viewtable.append([visitors[i],v_items,add,trans,time,cvalue])
    return viewtable

#items_per_day can be determined as a business role, here I chose 20. A good
#choice should be the average number of views per day or a value related.
#400 seem ideal: a serious shopper will likely look at an item for about 2-3 minutes
#but I was a bit stricter
mydata=modify_data(events,20)
mydata.to_csv('custumer_value_indicator.csv')
