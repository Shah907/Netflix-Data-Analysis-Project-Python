# Netflix-Data-Analysis-Project-Python
File https://www.kaggle.com/datasets/shivamb/netflix-shows

**Import Libaries and data**


```python
import os
import numpy as np # linear algebra operations
import pandas as pd # used for data preparation
import plotly.express as px #used for data visualization
from textblob import TextBlob #used for sentiment analysis

df = pd.read_csv('netflix.csv')

```

## Checking number of rows and columns in data


```python
df.shape
```




    (8807, 12)



# Checking content available in Dataset


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>Kirsten Johnson</td>
      <td>NaN</td>
      <td>United States</td>
      <td>September 25, 2021</td>
      <td>2020</td>
      <td>PG-13</td>
      <td>90 min</td>
      <td>Documentaries</td>
      <td>As her father nears the end of his life, filmm...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>TV Show</td>
      <td>Blood &amp; Water</td>
      <td>NaN</td>
      <td>Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban...</td>
      <td>South Africa</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
      <td>International TV Shows, TV Dramas, TV Mysteries</td>
      <td>After crossing paths at a party, a Cape Town t...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s3</td>
      <td>TV Show</td>
      <td>Ganglands</td>
      <td>Julien Leclercq</td>
      <td>Sami Bouajila, Tracy Gotoas, Samuel Jouy, Nabi...</td>
      <td>NaN</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>1 Season</td>
      <td>Crime TV Shows, International TV Shows, TV Act...</td>
      <td>To protect his family from a powerful drug lor...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s4</td>
      <td>TV Show</td>
      <td>Jailbirds New Orleans</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>1 Season</td>
      <td>Docuseries, Reality TV</td>
      <td>Feuds, flirtations and toilet talk go down amo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s5</td>
      <td>TV Show</td>
      <td>Kota Factory</td>
      <td>NaN</td>
      <td>Mayur More, Jitendra Kumar, Ranjan Raj, Alam K...</td>
      <td>India</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
      <td>International TV Shows, Romantic TV Shows, TV ...</td>
      <td>In a city of coaching centers known to train I...</td>
    </tr>
  </tbody>
</table>
</div>



# How to check columns name of dataset


```python
df.columns
```




    Index(['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added',
           'release_year', 'rating', 'duration', 'listed_in', 'description'],
          dtype='object')



# Taking the count of ratings available


```python
x = df.groupby(['rating']).size().reset_index(name='counts')
print(x)
```

          rating  counts
    0     66 min       1
    1     74 min       1
    2     84 min       1
    3          G      41
    4      NC-17       3
    5         NR      80
    6         PG     287
    7      PG-13     490
    8          R     799
    9      TV-14    2160
    10      TV-G     220
    11     TV-MA    3207
    12     TV-PG     863
    13      TV-Y     307
    14     TV-Y7     334
    15  TV-Y7-FV       6
    16        UR       3
    

# Creating the Piechart based on Content rating


```python
pieChart = px.pie(x, values='counts', names='rating', title='Distribution of content ratings on Netflix')
pieChart.show()
```



# Analyzing the top 5 Directors on Netflix


```python
df['director']=df['director'].fillna('Director not specified')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>Kirsten Johnson</td>
      <td>NaN</td>
      <td>United States</td>
      <td>September 25, 2021</td>
      <td>2020</td>
      <td>PG-13</td>
      <td>90 min</td>
      <td>Documentaries</td>
      <td>As her father nears the end of his life, filmm...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>TV Show</td>
      <td>Blood &amp; Water</td>
      <td>Director not specified</td>
      <td>Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban...</td>
      <td>South Africa</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
      <td>International TV Shows, TV Dramas, TV Mysteries</td>
      <td>After crossing paths at a party, a Cape Town t...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s3</td>
      <td>TV Show</td>
      <td>Ganglands</td>
      <td>Julien Leclercq</td>
      <td>Sami Bouajila, Tracy Gotoas, Samuel Jouy, Nabi...</td>
      <td>NaN</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>1 Season</td>
      <td>Crime TV Shows, International TV Shows, TV Act...</td>
      <td>To protect his family from a powerful drug lor...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s4</td>
      <td>TV Show</td>
      <td>Jailbirds New Orleans</td>
      <td>Director not specified</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>1 Season</td>
      <td>Docuseries, Reality TV</td>
      <td>Feuds, flirtations and toilet talk go down amo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s5</td>
      <td>TV Show</td>
      <td>Kota Factory</td>
      <td>Director not specified</td>
      <td>Mayur More, Jitendra Kumar, Ranjan Raj, Alam K...</td>
      <td>India</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
      <td>International TV Shows, Romantic TV Shows, TV ...</td>
      <td>In a city of coaching centers known to train I...</td>
    </tr>
  </tbody>
</table>
</div>




```python
directors_list = pd.DataFrame()
print(directors_list)
```

    Empty DataFrame
    Columns: []
    Index: []
    


```python
directors_list = df['director'].str.split(',', expand=True).stack()
print(directors_list)
```

    0     0           Kirsten Johnson
    1     0    Director not specified
    2     0           Julien Leclercq
    3     0    Director not specified
    4     0    Director not specified
                        ...          
    8802  0             David Fincher
    8803  0    Director not specified
    8804  0           Ruben Fleischer
    8805  0              Peter Hewitt
    8806  0               Mozez Singh
    Length: 9612, dtype: object
    


```python
directors_list = directors_list.to_frame()
print(directors_list)
```

                                 0
    0    0         Kirsten Johnson
    1    0  Director not specified
    2    0         Julien Leclercq
    3    0  Director not specified
    4    0  Director not specified
    ...                        ...
    8802 0           David Fincher
    8803 0  Director not specified
    8804 0         Ruben Fleischer
    8805 0            Peter Hewitt
    8806 0             Mozez Singh
    
    [9612 rows x 1 columns]
    


```python
directors_list.columns = ['Director']
print(directors_list)
```

                          Director
    0    0         Kirsten Johnson
    1    0  Director not specified
    2    0         Julien Leclercq
    3    0  Director not specified
    4    0  Director not specified
    ...                        ...
    8802 0           David Fincher
    8803 0  Director not specified
    8804 0         Ruben Fleischer
    8805 0            Peter Hewitt
    8806 0             Mozez Singh
    
    [9612 rows x 1 columns]
    


```python
directors = directors_list.groupby(['Director']).size().reset_index(name='Total Count')
print(directors)
```

                           Director  Total Count
    0                Aaron Moorhead            2
    1                   Aaron Woolf            1
    2      Abbas Alibhai Burmawalla            1
    3              Abdullah Al Noor            1
    4           Abhinav Shiv Tiwari            1
    ...                         ...          ...
    5116                Çagan Irmak            1
    5117           Ísold Uggadóttir            1
    5118        Óskar Thór Axelsson            1
    5119           Ömer Faruk Sorak            2
    5120               Şenol Sönmez            2
    
    [5121 rows x 2 columns]
    


```python
directors = directors[directors.Director != 'Director not specified']
```


```python
print(directors)
```

                           Director  Total Count
    0                Aaron Moorhead            2
    1                   Aaron Woolf            1
    2      Abbas Alibhai Burmawalla            1
    3              Abdullah Al Noor            1
    4           Abhinav Shiv Tiwari            1
    ...                         ...          ...
    5116                Çagan Irmak            1
    5117           Ísold Uggadóttir            1
    5118        Óskar Thór Axelsson            1
    5119           Ömer Faruk Sorak            2
    5120               Şenol Sönmez            2
    
    [5120 rows x 2 columns]
    


```python
directors = directors.sort_values(by=['Total Count'], ascending = False)
print(directors)
```

                 Director  Total Count
    4021    Rajiv Chilaka           22
    4068      Raúl Campos           18
    261         Jan Suter           18
    4652      Suhas Kadav           16
    3236     Marcus Raboy           16
    ...               ...          ...
    2341         J. Davis            1
    2342  J. Lee Thompson            1
    2343  J. Michael Long            1
    609    Smriti Keshari            1
    2561    Joaquín Mazón            1
    
    [5120 rows x 2 columns]
    


```python
top5Directors = directors.head()
print(top5Directors)
```

               Director  Total Count
    4021  Rajiv Chilaka           22
    4068    Raúl Campos           18
    261       Jan Suter           18
    4652    Suhas Kadav           16
    3236   Marcus Raboy           16
    


```python
top5Directors = top5Directors.sort_values(by=['Total Count'])
barChart = px.bar(top5Directors, x='Total Count', y = 'Director', title = 'Top 5 Directors on Netflix')
barChart.show()
```



# Analyzing the top 5 Actors on Netflix


```python
df['cast']=df['cast'].fillna('No cast specified')
cast_df = pd.DataFrame()
cast_df = df['cast'].str.split(',',expand=True).stack()
cast_df = cast_df.to_frame()
cast_df.columns = ['Actor']
actors = cast_df.groupby(['Actor']).size().reset_index(name = 'Total Count')
actors = actors[actors.Actor != 'No cast specified']
actors = actors.sort_values(by=['Total Count'], ascending=False)
top5Actors = actors.head()
top5Actors = top5Actors.sort_values(by=['Total Count'])
barChart2 = px.bar(top5Actors, x='Total Count', y='Actor', title='Top 5 Actors on Netflix')
barChart2.show()
```



# Analyzing the content produced on netflix based on years


```python
df1 = df[['type', 'release_year']]
df1 = df1.rename(columns = {"release_year":"Release Year", "type": "Type"})
df2 = df1.groupby(['Release Year', 'Type']).size().reset_index(name='Total Count')
```


```python
print(df2)
```

         Release Year     Type  Total Count
    0            1925  TV Show            1
    1            1942    Movie            2
    2            1943    Movie            3
    3            1944    Movie            3
    4            1945    Movie            3
    ..            ...      ...          ...
    114          2019  TV Show          397
    115          2020    Movie          517
    116          2020  TV Show          436
    117          2021    Movie          277
    118          2021  TV Show          315
    
    [119 rows x 3 columns]
    


```python
df2 = df2[df2['Release Year']>=2000]
graph = px.line(df2, x = "Release Year", y="Total Count", color = "Type", title = "Trend of Content Produced on Netfilx Every Year")
graph.show()
```



# Sentiment Analysis of Netflix Content


```python
df3 = df[['release_year', 'description']]
df3 = df3.rename(columns = {'release_year':'Release Year', 'description':'Description'})
for index, row in df3.iterrows():
  d=row['Description']
  testimonial = TextBlob(d)
  p = testimonial.sentiment.polarity
  if p==0:
    sent = 'Neutral'
  elif p>0:
    sent = 'Positive'
  else:
    sent = 'Negative'
  df3.loc[[index, 2], 'Sentiment']=sent

df3 = df3.groupby(['Release Year', 'Sentiment']).size().reset_index(name = 'Total Count')

df3 = df3[df3['Release Year']>2005]
barGraph = px.bar(df3, x="Release Year", y="Total Count", color = "Sentiment", title = "Sentiment Analysis of Content on Netflix")
barGraph.show()
```

# Genre Distribution over the Years (Line Chart)


```python
genre_year = df.groupby(['release_year', 'listed_in']).size().reset_index(name='count')
fig = px.line(genre_year, x='release_year', y='count', color='listed_in', title='Genre Distribution Over the Years')
fig.show()

```



# Top 5 Countries Contributing Content (Bar Chart):


```python
top_countries = df['country'].value_counts().head(5)
fig = px.bar(top_countries, x=top_countries.index, y=top_countries.values, title='Top 5 Countries Contributing Content')
fig.show()

```



# Content Types Ratio (Pie Chart):


```python
content_type_ratio = df['type'].value_counts()
fig = px.pie(names=content_type_ratio.index, values=content_type_ratio.values, title='Content Types Ratio')
fig.show()

```



# Seasons vs. Episodes (Scatter Plot):


```python
tv_shows = df[df['type'] == 'TV Show']
fig = px.scatter(tv_shows, x='duration', y='duration', title='Seasons vs. Episodes')
fig.show()

```



# Release Year Distribution: 
    Display a histogram showing the distribution of content releases over the years.


```python
fig = px.histogram(df, x='release_year', title='Release Year Distribution')
fig.show()

```



# Country-wise Content Production: 

    Display a choropleth map showing the distribution of content production across different countries.
        python


```python
country_counts = df['country'].value_counts().reset_index()
country_counts.columns = ['country', 'count']
fig = px.choropleth(country_counts, locations='country', locationmode='country names', color='count',
                    title='Country-wise Content Production')
fig.show()

```



# Word Cloud of Titles: 
    
    Create a word cloud visualization of the titles of the content available on Netflix.


```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(background_color='white').generate(' '.join(df['title']))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Titles')
plt.axis('off')
plt.show()

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[29], line 1
    ----> 1 from wordcloud import WordCloud
          2 import matplotlib.pyplot as plt
          4 wordcloud = WordCloud(background_color='white').generate(' '.join(df['title']))
    

    ModuleNotFoundError: No module named 'wordcloud'

