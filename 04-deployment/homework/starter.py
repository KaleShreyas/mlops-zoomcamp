#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('pip show scikit-learn')


# In[ ]:


import pickle
import sys
import pandas as pd


# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[ ]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[ ]:


year = int(sys.argv[1])
month = int(sys.argv[2])
print(year)
print(month)

# In[6]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}-0{}.parquet'.format(year, month))


# In[7]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[9]:


import statistics

# Standard deviation of preds
res = statistics.pstdev(y_pred)
 
# Printing result
print("Mean of the predicted duration for this dataset : " + str(statistics.mean(y_pred)))
print("Standard deviation of the predicted duration for this dataset : " + str(res))


# In[ ]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

