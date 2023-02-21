import streamlit as st
from bokeh.models.widgets import Div

def funct():
    st.title('Book an Uber or Order from Zomato')
    st.image('data/Uber.png',use_column_width='auto')
    if st.button('Book an Uber'):
        js = "window.open('https://m.uber.com')"  # New tab or window
        js = "window.location.href = 'https://m.uber.com'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

    st.image('data/zomorder.jpg', use_column_width='auto')
    if st.button('Order from Zomato'):
        js = "window.open('https://www.zomato.com')"  # New tab or window
        js = "window.location.href = 'https://www.zomato.com/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)