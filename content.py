import pandas as pd
import numpy as np
import math

Ratings=pd.read_csv('ratings.csv', encoding='ISO-8859-1')
Movies=pd.read_csv('movies.csv', encoding='ISO-8859-1')
Tags=pd.read_csv('tags.csv', encoding='ISO-8859-1')


TF= Tags.groupby(['movieId','tag'], as_index = False, sort = False).count().rename(columns = {'userId': 'tag_count_TF'})[['movieId','tag','tag_count_TF']]
Tag_distinct = Tags[['tag','movieId']].drop_duplicates()

DF =Tag_distinct.groupby(['tag'], as_index = False, sort = False).count().rename(columns = {'movieId': 'tag_count_DF'})[['tag','tag_count_DF']]

a=math.log10(len(np.unique(Tags['movieId'])))

DF['IDF']=a-np.log10(DF['tag_count_DF'])
TF = pd.merge(TF,DF,on = 'tag', how = 'left', sort = False)
TF['TF-IDF']=TF['tag_count_TF']*TF['IDF']

print(TF)
