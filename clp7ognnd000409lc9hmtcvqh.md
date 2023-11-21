---
title: "Analyzing UFO Sightings Worldwide | Episode 2"
seoTitle: "UFO Discoveries: Analyzing Global Sightings"
seoDescription: "Dive into UFO Insights: Explore global sightings, uncover patterns, and join the data-driven quest to unravel the mysteries of unidentified flying objects."
datePublished: Tue Nov 21 2023 01:49:15 GMT+0000 (Coordinated Universal Time)
cuid: clp7ognnd000409lc9hmtcvqh
slug: analyzing-ufo-sightings-worldwide-episode-2
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1700182577692/0cefb30f-2907-40cb-87ad-838d4d9875af.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1700182565892/8fb40721-30d7-4e25-956e-89dae1c9db8e.png
tags: programming-blogs, data-science, machine-learning, data-analysis, kaggle

---

Our skies are full of mysteries, and one of the most intriguing ones is the Unidentified Flying Objects (UFOs). From quick sightings to jaw-dropping encounters, UFO sightings have fueled our curiosity and sparked endless possibilities. In this edition of Kaggle Stories, we invite you to join us on a celestial journey to decode the mystery of UFO sightings worldwide.

#### The Intrigue of the Unexplained

UFO sightings have been a source of wonder for decades, and they still puzzle us today. Are they advanced aircraft, extraterrestrial visitors, or simply a trick of the light? In this article, we explore a fascinating dataset spanning time and space to shed light on the patterns, anomalies, and mysteries that are hidden within the vast tapestry of UFO encounters.

#### Our Analytical Lens

Using data science tools, we take a closer look at eyewitness accounts and timestamps to uncover the underlying stories. What trends emerge when we chart the course of sightings over decades? Are there any hotspots that require our attention? Join us as we sift through the data, seeking signals amidst the noise, and turning speculation into insight.

#### Uniting Curiosity and Science

We believe this analysis is more than just a statistical exploration. It's a tribute to the power of human curiosity and the scientific method. Together, we embark on a data-driven quest to demystify the UFO phenomenon, one data point at a time. Whether you're an expert ufologist or simply curious about the topic, fasten your seatbelts as we take off into the realm where the unexplained meets the analytical.

---

## **Dataset Overview**

The dataset provides a comprehensive record of UFO sightings, offering a unique opportunity to uncover patterns, trends, and insights into these unexplained phenomena. By applying exploratory data analysis (EDA) techniques, we aim to gain a deeper understanding of the temporal, geographical, and categorical aspects of UFO sightings.

dataset link: [https://www.kaggle.com/datasets/willianoliveiragibin/ufo-sightings](https://www.kaggle.com/datasets/willianoliveiragibin/ufo-sightings)

#### **Objectives**

In this analysis, we have identified the following objectives:

1. **Temporal Patterns:** Investigate the temporal distribution of sightings to identify trends over the years, months, and hours.
    
2. **Geographic Insights:** Explore the geographical distribution of sightings to understand where these phenomena are more prevalent.
    
3. **Seasonal Analysis:** Examine how UFO sightings vary across different seasons.
    
4. **Country and Region Analysis:** Analyze the distribution of sightings across countries and regions to identify hotspots.
    
5. **UFO Shapes and Durations:** Investigate the relationship between UFO shapes and encounter durations.
    

```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
```

```python
data = pd.read_csv("ufo-sightings-transformed.csv")
data.head()
```

|  | Unnamed: 0 | Date\_time | date\_documented | Year | Month | Hour | Season | Country\_Code | Country | Region | Locale | latitude | longitude | UFO\_shape | length\_of\_encounter\_seconds | Encounter\_Duration | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 1949-10-10 20:30:00 | 4/27/2004 | 1949 | 10 | 20 | Autumn | USA | United States | Texas | San Marcos | 29.883056 | \-97.941111 | Cylinder | 2700.0 | 45 minutes | This event took place in early fall around 194... |

```python
data.info()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 80328 entries, 0 to 80327
Data columns (total 17 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   Unnamed: 0                   80328 non-null  int64  
 1   Date_time                    80328 non-null  object 
 2   date_documented              80328 non-null  object 
 3   Year                         80328 non-null  int64  
 4   Month                        80328 non-null  int64  
 5   Hour                         80328 non-null  int64  
 6   Season                       80328 non-null  object 
 7   Country_Code                 80069 non-null  object 
 8   Country                      80069 non-null  object 
 9   Region                       79762 non-null  object 
 10  Locale                       79871 non-null  object 
 11  latitude                     80328 non-null  float64
 12  longitude                    80328 non-null  float64
 13  UFO_shape                    78398 non-null  object 
 14  length_of_encounter_seconds  80328 non-null  float64
 15  Encounter_Duration           80328 non-null  object 
 16  Description                  80313 non-null  object 
dtypes: float64(3), int64(4), object(10)
memory usage: 10.4+ MB
```

```python
data.dropna(inplace=True)
data['Date_time'] = pd.to_datetime(data['Date_time'])
data.drop(columns=["Unnamed: 0"], inplace=True)
data.head(1)
```

|  | Date\_time | date\_documented | Year | Month | Hour | Season | Country\_Code | Country | Region | Locale | latitude | longitude | UFO\_shape | length\_of\_encounter\_seconds | Encounter\_Duration | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1949-10-10 20:30:00 | 4/27/2004 | 1949 | 10 | 20 | Autumn | USA | United States | Texas | San Marcos | 29.883056 | \-97.941111 | Cylinder | 2700.0 | 45 minutes | This event took place in early fall around 194... |

## **Exploratory Data Analysis (EDA)**

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

#### **Temporal Analysis**

Explore temporal patterns, such as whether there are certain months or hours when sightings are more frequent

```python
# Set Seaborn style and color palette
sns.set(style="whitegrid", palette="viridis")

# Extract and analyze the distribution of sightings over different years, months, and hours
yearly_counts = data['Year'].value_counts().sort_index()
monthly_counts = data['Month'].value_counts().sort_index()
hourly_counts = data['Hour'].value_counts().sort_index()

# Plot the temporal distributions using Seaborn with rotated x-axis labels
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

sns.barplot(x=yearly_counts.index, y=yearly_counts.values, ax=axes[0])
axes[0].set_title('Yearly Distribution of Sightings')
axes[0].set_ylabel('Number of Sightings')

sns.barplot(x=monthly_counts.index, y=monthly_counts.values, ax=axes[1])
axes[1].set_title('Monthly Distribution of Sightings')
axes[1].set_ylabel('Number of Sightings')

sns.barplot(x=hourly_counts.index, y=hourly_counts.values, ax=axes[2])
axes[2].set_title('Hourly Distribution of Sightings')
axes[2].set_xlabel('Hour of the Day')
axes[2].set_ylabel('Number of Sightings')

# Rotate x-axis labels for better readability
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Add a title for the entire plot
plt.suptitle('Temporal Distribution of UFO Sightings', y=1.02, fontsize=16)

plt.tight_layout()
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1700183869908/9e9ece37-7e24-4808-9c05-92deefdafc4f.png align="center")

#### **Geographical Analysis**

##### Explore the distribution of sightings across different countries, regions, or locales.

##### Visualize the geographical distribution using maps

```python
import geopandas as gpd
from shapely.geometry import Point

# Create a GeoDataFrame for plotting
geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
gdf = gpd.GeoDataFrame(data, geometry=geometry)

# Plot the geographical distribution using Seaborn and GeoPandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots(figsize=(15, 10))
sns.scatterplot(x='longitude', y='latitude', data=gdf,ax=ax, palette='viridis', s=50)
world.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.5)
plt.title('Geographical Distribution of Sightings')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1700183885069/8af707b8-21fe-4eb6-a120-76497dd9971b.png align="center")

#### **Seasonal Analysis**

##### Explore the distribution of sightings across different seasons

```python
sns.set(style="whitegrid", palette="husl")

# Analyze the distribution of sightings across different seasons
season_counts = data['Season'].value_counts()

# Plot the distribution of sightings by season using Seaborn
plt.figure(figsize=(15, 6))
sns.barplot(x=season_counts.index, y=season_counts.values)
plt.title('Distribution of Sightings by Season')
plt.xlabel('Season')
plt.ylabel('Number of Sightings')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1700183905734/58d16879-fdfb-4d47-bea3-df4ac87cb7df.png align="center")

##### Explore the distribution of sightings across different countries and regions.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style and color palette
sns.set(style="whitegrid", palette="colorblind")

# Analyze the distribution of sightings across different countries and regions
country_counts = data['Country'].value_counts()[:10]
region_counts = data['Region'].value_counts()[:10]

# Plot the distribution of sightings by country and region using Seaborn
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

sns.barplot(x=country_counts.index, y=country_counts.values, ax=axes[0], palette="viridis")
axes[0].set_title('Distribution of Sightings by Country')
axes[0].set_ylabel('Number of Sightings')

sns.barplot(x=region_counts.index, y=region_counts.values, ax=axes[1], palette="viridis")
axes[1].set_title('Distribution of Sightings by Region')
axes[1].set_xlabel('Region')
axes[1].set_ylabel('Number of Sightings')

plt.tight_layout()
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1700183919950/fcef1847-80f1-4a75-b7be-c6024e29d239.png align="center")

#### **Encounter Duration Analysis**

##### Explore the average encounter duration in different countries

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style and color palette
sns.set(style="whitegrid", palette="pastel")

# Analyze the average encounter duration by country
avg_duration_by_country = data.groupby('Country')['length_of_encounter_seconds'].mean().sort_values()[:10]

# Plot the average encounter duration by country using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_duration_by_country.values, y=avg_duration_by_country.index, palette="viridis")
plt.title('Average Encounter Duration by Country')
plt.xlabel('Average Encounter Duration (seconds)')
plt.ylabel('Country')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1700183938222/bade2d0e-5c8f-4249-969a-56ef5679759a.png align="center")

#### **UFO Shape and Encounter Duration Relationship**

##### Explore whether there's a relationship between the reported UFO shape and encounter duration

```python
# Analyze the relationship between UFO shape and encounter duration
plt.figure(figsize=(12, 6))
sns.boxplot(x='UFO_shape', y='length_of_encounter_seconds', data=data)
plt.title('Relationship between UFO Shape and Encounter Duration')
plt.xlabel('UFO Shape')
plt.ylabel('Encounter Duration (seconds)')
plt.xticks(rotation=45)
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1700183964911/8d70efb8-a7cb-469c-9e70-f74ea629d11a.png align="center")

##### Explore how the distribution of UFO shapes has changed over the years

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style and color palette
sns.set(style="whitegrid", palette="viridis")

# Create a pivot table for the distribution of UFO shapes over time
shape_over_time = data.pivot_table(index='Year', columns='UFO_shape', aggfunc='size', fill_value=0)

# Plot the distribution of UFO shapes over time using Seaborn with different colors for each shape
plt.figure(figsize=(15, 8))
for shape in shape_over_time.columns:
    sns.lineplot(x=shape_over_time.index, y=shape_over_time[shape], label=shape, linewidth=2.5)

plt.title('Distribution of UFO Shapes Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Sightings')
plt.legend(title='UFO Shape', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1700183995768/2dbeae61-9f04-4bbc-8fce-b3614f970f46.png align="center")

#### **Description Word Cloud**

##### Generate a word cloud to visualize the most common words in the descriptions.

```python
from wordcloud import WordCloud

# Combine all descriptions into a single string
all_descriptions = ' '.join(data['Description'].dropna())

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of UFO Sightings Descriptions')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1700184046959/f5116902-0394-45cd-b030-a0ae5e64fe24.png align="center")

## **Report on UFO Sightings Data Analysis**

This report presents a comprehensive analysis of a dataset containing information about UFO sightings. The dataset includes details such as the date and time of the sighting, location, UFO shape, and duration of the encounter.

### **Exploratory Data Analysis (EDA)**

#### **Temporal Distribution**

#### **Yearly Distribution of Sightings**

* The bar plot shows a gradual increase in the yearly count of UFO sightings from 1993 to 2011. There were sudden peaks in 2012 and 2014, followed by a significant drop.
    

#### **Monthly Distribution of Sightings**

* The bar plot depicts a bell-curve pattern, indicating that peak occurrences happen in July, suggesting a seasonal variation in UFO sightings.
    

#### **Hourly Distribution of Sightings**

* The bar plot shows that sightings are most frequent from 5 pm to 11 pm, highlighting the hourly distribution of UFO sightings.
    

#### **Geographical Distribution**

* The scatter plot on the world map reveals that there is a concentration of sightings in coastal areas of countries, indicating potential geographic patterns.
    

#### **Season Analysis**

##### **Distribution of Sightings Across Different Seasons**

* The bar plot shows that summer exhibits the highest occurrences of sightings, followed by autumn. Spring and winter have comparatively lower frequencies.
    

#### **Country and Region Analysis**

###### **Distribution of Sightings Across Top 10 Countries and Regions**

* The USA has the most UFO sightings, recording nearly 70,000 cases, followed by Canada, with approximately 8,000 cases. Among regions, all the top 10 are from the USA, and California has almost 10,000 cases, ranking first.
    

#### **UFO Shapes**

###### **UFO Shape and Encounter Duration Relationship**

* An analysis of UFO shapes reveals that sphere-shaped UFOs have the longest encounter durations, followed by the "Unknown" (others) category.
    

#### **Average Encounter Duration by Country**

* The horizontal bar plot shows the average encounter duration for the top 10 countries. Argentina, Saint Vincent and the Grenadines, São Tomé and Príncipe, Libya, and Uruguay have the longest average durations, around 60 seconds. Azerbaijan ranks second, with an average duration of 56 seconds.
    

---

### **Conclusion**

We've reached the end of our exciting journey through the fascinating dataset of UFO sightings! As we wrap up, we find ourselves at a crossroads of speculation and analysis. We've uncovered some intriguing patterns and anomalies that have shed some light on the mysterious world of unidentified flying objects. But what do these findings really mean, and where do we go from here?

#### Insights from UFO Sightings

Thanks to the data, we now have a much better understanding of the unseen phenomena that have graced our skies! From clusters of sightings in specific regions to temporal trends that are hard to explain, our insights have illuminated the breadth and complexity of the UFO narrative.

#### Further Exploration

The journey doesn't end here. As we conclude this chapter of Kaggle Stories, consider delving into the dataset yourself. What additional patterns might you uncover? What questions remain unanswered? The dataset is open for exploration, and your analytical lens may reveal new facets of the UFO phenomenon.

#### Community Engagement

Engage with the Kaggle Stories community. Comment on your thoughts, interpretations, or additional analyses you've conducted. Let's continue the conversation on UFO sightings, data science, and the endless possibilities that lie beyond our terrestrial horizons.

In the twilight of our analysis, the call to action is clear: let's continue to decode the skies, one dataset at a time. Join us in the ongoing exploration, and together, we may bring new perspectives to the age-old mystery of unidentified flying objects.

---

## **By the way…**

#### Call to action

*Hi, Everydaycodings— I’m building a newsletter that covers deep topics in the space of engineering. If that sounds interesting,* [***subscribe***](https://neuralrealm.hashnode.dev/newsletter) *and don’t miss anything. If you have some thoughts you’d like to share or a topic suggestion, reach out to me via* [***LinkedIn***](https://www.linkedin.com/in/kumar-saksham1891/) *or* [***X***](https://twitter.com/everydaycodings).

#### References

*And if you’re interested in diving deeper into these concepts, here are some great starting points:*

* [**Kaggle Stories**](https://neuralrealm.hashnode.dev/series/kaggle-stories) *\-* Each episode of Kaggle Stories takes you on a journey behind the scenes of a Kaggle notebook project, breaking down tech stuff into simple stories.
    
* [**Machine Learning**](https://neuralrealm.hashnode.dev/series/machine-learning) *\-* This series covers ML fundamentals & techniques to apply ML to solve real-world problems using Python & real datasets while highlighting best practices & limits.