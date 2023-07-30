import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


movies_data = pd.read_csv("https://raw.githubusercontent.com/danielgrijalva/movie-stats/7c6a562377ab5c91bb80c405be50a0494ae8e582/movies.csv")

movies_data.info()

movies_data.duplicated()

movies_data.count()

movies_data.dropna()


st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

st.write("""Average Movie Budget, Grouped by Genre""")
avg_budget = movies_data.groupby('genre')['budget'].mean().round()
avg_budget = avg_budget.reset_index()
genre = avg_budget['genre']
avg_bud = avg_budget['budget']

fig = plt.figure(figsize = (19, 10))

plt.bar(genre, avg_bud, color = 'maroon')
plt.xlabel('genre')
plt.ylabel('budget')
plt.title('Matplotlib Bar Chart Showing the Average \
Budget of Movies in Each Genre')

st.pyplot(fig)

col1, col2 = st.columns(2)
col1.write('# This is Column 1')
col2.write('# This is Column 2')

st.write('### Columns of different sizes')
col1, col2, col3, col4 = st.columns([1,3,1,2])

col1.write('# This is Column 1')
col2.write('# This is Column 2')
col3.write('# This is Column 3')
col4.write('# This is Column 4')

