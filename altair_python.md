# Altair visualization snippets

```python
import altair as alt
import pandas as pd

```
Structure:  
1)alt = Call Altair  
2).Chart(data) = Name the data  
3).mark_X() = Plot using mark of type X  
4).encode(...) = Encoding X and Y axis and other visual encoding options  
5).properties(...) = Add other options (titles, filters...)  
6).interactive() = Make interactive  

## Histogram

```python
#Get data
brain = pd.read_csv("https://raw.githubusercontent.com/rezpe/datos_viz/master/brain.csv")

alt.Chart(brain).mark_bar().encode(
    x=alt.X('Body Weight', bin=alt.Bin(maxbins=30)),
    y='count()'
).interactive()
```

## Bar Chart

```python
#Get data
trends = pd.read_csv('https://raw.githubusercontent.com/rezpe/datos_viz/master/google_trends.csv')

alt.Chart(trends).mark_bar().encode(
    x='search_term',
    y='mean(value)',
    color='search_term'
).interactive()
```

## Line graph

```python
#Get data
trends = pd.read_csv('https://raw.githubusercontent.com/rezpe/datos_viz/master/google_trends.csv')

alt.Chart(trends).mark_line().encode(
    x='yearmonth(date):T',
    y='mean(value)',
    color='search_term'
).interactive()

```
## Map

```python
#Get data
weather_df = pd.read_csv("https://github.com/rezpe/datos_viz/blob/master/worldTemperatures.csv?raw=true", header=None)
weather_df.columns = ['0', '1', '2', 'date', '4', 'temp', '6', '7', '8', '9', '10', 'lat', 'lon','country', 'city']

#Background countries map
countries = alt.topo_feature("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json",
                             feature="countries")
background = alt.Chart(countries).mark_geoshape(
    fill="lightgrey",
    stroke="white")

#Point plot
temp = weather_df.groupby("city").mean().reset_index() #group data by city

cities = alt.Chart(temp).mark_point().encode(
    latitude="lat",
    longitude="lon",
    size=alt.value(1),
    color=alt.Color("temp",scale=alt.Scale(range=["lightblue","orange","red"]))
)
##Combine plots
background+cities

```
## Combine plots

### Horizontally
```python
#Get data
trends = pd.read_csv('https://raw.githubusercontent.com/rezpe/datos_viz/master/google_trends.csv')

#Plot 1
comp_trends = alt.Chart(trends).mark_bar().encode(
    x='search_term',
    y='mean(value)',
    color='search_term'
).properties(
    width=100
)

#Plot 2
trends_line = alt.Chart(trends).mark_line().encode(
    x="yearmonth(date):T",
    y='mean(value)',
    color="search_term"
).properties(
    width=600
)

#Combine plots 1 and 2
(comp_trends|trends_line) 
```
### Multiple plot

```python
#Get data
brain = pd.read_csv("https://raw.githubusercontent.com/rezpe/datos_viz/master/brain.csv")

#Plot 1
hist_brain = alt.Chart(brain).mark_bar().encode(
    x=alt.X('Brain Weight',bin=alt.Bin(maxbins=100)),
    y="count()"
).properties(
    title="Distribución del peso de los cerebros",
    width=200
)

#Plot 2
hist_body = alt.Chart(brain).mark_bar().encode(
    x=alt.X('Body Weight',bin=alt.Bin(maxbins=100)),
    y="count()"
).properties(
    title="Distribución del peso de los cuerpos",
    width=200
)

#Plot 3
scatter_brain_body = alt.Chart(brain).mark_point().encode(
    x='Brain Weight',
    y='Body Weight'
).properties(
    title="Distribución del peso de los cerebros",
    width=400
)

#Combine plots (plot 3 on top, plot 1 and 2 below)
scatter_brain_body&(hist_body|hist_brain)
```

## Filter by selection
1) define filter (type and what to select) = alt.selection(type= , encodings=[x])  
2) In the plot to select:  .properties(`selection=your_selection`)
3) In the plot to show selection:  .transform_filter(`your_selection`)

```python
#Get data
weather_df = pd.read_csv("https://github.com/rezpe/datos_viz/blob/master/worldTemperatures.csv?raw=true", header=None)
weather_df.columns = ['0', '1', '2', 'date', '4', 'temp', '6', '7', '8', '9', '10', 'lat', 'lon','country', 'city']

temp = weather_df.groupby(["city","country"]).mean().reset_index() #Group by city and country

#Create selection (single selection)
select_ctry = alt.selection(type="single",encodings=["x"])

#Plot 1 (the one to show the selection)
ranking_city = alt.Chart(temp).mark_bar().encode(
    x=alt.X("city",
            sort=alt.Sort(field="temp",
                          order="descending"),
            
            ),
    y="mean(temp)",
    tooltip="city"
).properties(
    width=800,
    title="Ranking de paises por temperatura media"
).transform_filter( #<----- THE SELECTION FILTER
    select_ctry
)

#Plot 2 (the one to make selection)
ranking_country = alt.Chart(temp).mark_bar().encode(
    x=alt.X("country",
            sort=alt.Sort(field="temp",
                          op="mean",
                          order="descending"),
            axis=None
            ),
    y="mean(temp)",
    tooltip="country"
).properties(
    width=800,
    title="Ranking de ciudades por temperatura media",
    selection=select_ctry #<---- THE SELECTION
)

#Combine plots
ranking_country&ranking_city

```

## Range selection

```python
```python
#Get data
weather_df = pd.read_csv("https://github.com/rezpe/datos_viz/blob/master/worldTemperatures.csv?raw=true", header=None)
weather_df.columns = ['0', '1', '2', 'date', '4', 'temp', '6', '7', '8', '9', '10', 'lat', 'lon','country', 'city']

temp = weather_df.groupby("city").mean().reset_index()

#Background countries map
countries = alt.topo_feature("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json",
                             feature="countries")

background = alt.Chart(countries).mark_geoshape(
    fill="lightgrey",
    stroke="white"
)

#Define selection (interval)
select_city = alt.selection(type="interval",encodings=["x"])

#Plot 1 (the one to show selection)
cities = alt.Chart(temp).mark_point().encode(
    latitude="lat",
    longitude="lon",
    size=alt.value(1),
    color=alt.Color("temp",scale=alt.Scale(range=["lightblue","orange","red"]))
).transform_filter(
    select_city
).properties(
    height=800,
    width=800
)

#Plot 2 (the one to make selection)
ranking_city = alt.Chart(temp).mark_bar().encode(
    x=alt.X("city",
            sort=alt.Sort(field="temp",
                          op="mean",
                          order="descending"),
            
            axis=None),
    y="mean(temp)",
    tooltip="city"
).properties(
    width=800,
    height=100,
    title="Ranking de ciudades por temperatura media",
    selection=select_city
)

#Combine plots / maps
ranking_city&(background+cities)

```
