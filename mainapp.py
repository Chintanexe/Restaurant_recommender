import streamlit as st
from radio import home,zomato,restlist

st. sidebar.image('data/icon3.png',use_column_width='auto')
st.sidebar.title("All about the App...")

option = st.sidebar.radio(
    'Navigate through various features of the app!',
    ('Home.','Restuarants Recommender.','Uber and Zomato.')
)
if option == 'Home.':
    home.home()
if option == 'Uber and Zomato.':
    zomato.funct()
if option == 'Restuarants Recommender.':
    restlist.show()
st. sidebar.image('data/unnamed.png',use_column_width='auto')
