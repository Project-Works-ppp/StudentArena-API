import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
df = pd.read_csv("dice_com-job_us_sample.csv")


def identify_tokens(row):
    jobdescription = row['jobdescription']
    tokens = nltk.word_tokenize(jobdescription)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df['description'] = df.apply(identify_tokens, axis=1)

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))                  

def remove_stops(row):
    my_list = row['description']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

df['meaningful'] = df.apply(remove_stops, axis=1)

import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

df['meaningfulstring']=df['meaningful'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))

df['meaningfulstring']=df['meaningfulstring'].str.lower()
df['skills']=df['skills'].str.lower()
df['jobtitle']=df['jobtitle'].str.lower()
df['skills']=df['skills'].fillna(' ')

# see below and see job description strings are removed from skills
for i in df.index:
  if(df['skills'].iloc[i]=="see below" or df['skills'].iloc[i]=="(see job description)" ):
    df['skills'].iloc[i]=" "

def parameter(row):    
    return row['skills']+" "+row['meaningfulstring']

df['combined']=df.apply(parameter,axis=1)    

df.insert(0,'Job id',range(1,1+len(df)))


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined'])
cosine_sim = cosine_similarity(count_matrix)

def get_index_from_job_type(title):
    return df[df['jobtitle']==title]['Job id'].values[0]

def recommend(job):
  jobs_user_likes =job
  Job_id = get_index_from_job_type(jobs_user_likes)
  similar_jobs = list(enumerate(cosine_sim[Job_id]))
  sorted_similar_jobs = sorted(similar_jobs,key=lambda x:x[1],reverse=True)[1:]
  print("Top 10 similar jobs to "+job+" are:\n")
  return sorted_similar_jobs
k=recommend("java architect - denver, co - fulltime")
i=0

y_pred=[]
for element in k:
    if i==0:
      y_pred=df.iloc[(element[0])]
    #print(df.iloc[(element[0])])
    i=i+1
    if i>0:
      y_pred=y_pred.append(df.iloc[(element[0])])
    if i>10:
        break
for i in y_pred:
  print(i)