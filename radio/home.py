import streamlit as st
from PIL import Image

def home():
    image = Image.open('data/Toit.jpg')
    st.image(image, use_column_width=True)
    st.warning("The app to get you to the closest and most highly rated places to eat!")
    st.markdown("Based on data from Zomato, the app covers restaurants from various regions across Bangalore to recommend the 10 most similar restaurants to the restaurant of your liking.")
    st.success("Because hunger is also an emergency!! :ambulance:" ":100:")
    