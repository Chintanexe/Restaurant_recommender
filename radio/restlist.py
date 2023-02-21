import pandas as pd
import streamlit as st
import numpy as np
import os
from bokeh.models.widgets import Div
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re 

def show():
   
   zomato_real = pd.read_csv('data/zomato.csv')
   zomato = zomato_real.drop(['phone','book_table','online_order'],axis=1)
   
   zomato.duplicated().sum()
   zomato.drop_duplicates(inplace=True)
   zomato.isnull().sum()
   zomato.dropna(how='any',inplace=True)
   
   zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})
   
   zomato['cost'] = zomato['cost'].astype(str)  
   zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',',''))
   zomato['cost'] = zomato['cost'].astype(float)
   
   zomato = zomato.loc[zomato.rate!='NEW'] 
   zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
   remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
   zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype(float)
   zomato.name = zomato.name.apply(lambda x:x.title())
   
   #zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
   #zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)
   
   restaurants = zomato['name'].unique() 
   zomato['mean_rating'] = 0
   for i in range(len(restaurants)):
      zomato['mean_rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()
   ## sante spa
   zomato['name'].replace({'Santã\x83Â\x83Ã\x82Â\x83Ã\x83Â\x82Ã\x82Â\x83Ã\x83Â\x83Ã\x82Â\x82Ã\x83Â\x82Ã\x82Â\x83Ã\x83Â\x83Ã\x82Â\x83Ã\x83Â\x82Ã\x82Â\x82Ã\x83Â\x83Ã\x82Â\x82Ã\x83Â\x82Ã\x82Â© Spa Cuisine':'Sante spa cuisine'},inplace=True)

   zomato["reviews_list"] = zomato["reviews_list"].str.lower()
   zomato[['reviews_list', 'cuisines']].sample(5)

   import string
   PUNCT_TO_REMOVE = string.punctuation
   def remove_punctuation(text):
      return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
   zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))

   import nltk
   nltk.download('stopwords')
   from nltk.corpus import stopwords
   STOPWORDS = set(stopwords.words('english')) 

   def remove_stopwords(text):
      return " ".join([word for word in str(text).split() if word not in STOPWORDS])
   zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))

   def remove_urls(text):
      url_pattern = re.compile(r'https?://\S+|www\.\S+')
      return url_pattern.sub(r'', text)
   zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))
   
   zomrest = zomato.set_index('name')
   zomrest = zomrest.sample(frac= 0.6)
   indices = pd.Series(zomrest.index)
   indices[indices == 'San Churro Cafe'].index[0]

   tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
   tfidf_matrix = tfidf.fit_transform(zomrest['reviews_list'])

   cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
   def recommend(name, cosine_similarities = cosine_similarities):
   
      recommend_restaurant = []
    
    # Find the index of the hotel entered
      idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
      score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
      top30_indexes = list(score_series.iloc[0:31].index)
    
    # Names of the top 30 restaurants
      for each in top30_indexes:
          recommend_restaurant.append(list(zomrest.index)[each])
    
    # Creating the new data set to show similar restaurants
      df_new = pd.DataFrame(columns=['cuisines', 'mean_rating', 'cost'])
    
    # Create the top 30 similar restaurants with some of their columns
      for each in recommend_restaurant:
         df_new = df_new.append(pd.DataFrame(zomrest[['cuisines','mean_rating', 'cost']][zomrest.index == each].sample()))
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
      df_new = df_new.drop_duplicates(subset=['cuisines','mean_rating', 'cost'], keep=False)
      df_new = df_new.sort_values(by='mean_rating', ascending=False).head(10)
    
      print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
      return df_new

   st.title('Restaurant Recommender')

   option = st.selectbox('Select Your Area', ('Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
       'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
       'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
       'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
       'Koramangala 4th Block', 'Koramangala 5th Block',
       'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
       'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
       'Old Airport Road', 'Rajajinagar', 'Residency Road',
       'Sarjapur Road', 'Whitefield'))

   title = st.selectbox('Select Your Restaurant', (tuple(zomato[zomato.city ==option]['name'].unique())))

      #def details(pd):
   recdf = recommend(title)

   st.write(recdf)




    #TYPE OF RESTURANT


    #LOCATION


    #URL