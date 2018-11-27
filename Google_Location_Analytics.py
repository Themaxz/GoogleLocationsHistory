
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
def save(df,path,name):
    n = r'\{}'.format(name)
    df.to_csv(path+n, index=False)

def load(path,name):
    n = r'\{}'.format(name)
    return pd.read_csv(path+n,encoding='iso-8859-1')
path = r'C:\Users\thema\Documents\Google Locations'


# In[2]:

locations = load(path,'MyLocations2018.csv')


# In[5]:

from geopy.geocoders import Nominatim


# In[6]:

geolocator = Nominatim()


locations.Latitude = locations.Latitude.apply(lambda x: x/10**7)

locations.Longitude = locations.Longitude.apply(lambda x: x/10**7)

locations['Date'] =  pd.to_datetime(locations['Time'])


# In[11]:

locations.info()


locations.to_csv('GoogleLocations2018.csv')




locations['Year'] = locations['Date'].apply(lambda x: str(x.year))

locations['Month'] = locations['Date'].apply(lambda x: str(x.month))

locations['Day'] = locations['Date'].apply(lambda x: str(x.day))

locations['Day'] = locations['Day'].apply(lambda x:  '0'+x if len(x)==1 else x)

locations['MonthDay'] = locations['Month']+'_'+locations['Day']


locations.to_csv('GoogleLocations.csv')

DF2018 = locations[locations['Year']=='2018']

get_ipython().magic('matplotlib inline')

DAYS = pd.DataFrame({'Latitude':DF2018.groupby(['MonthDay']).Latitude.mean(),
                     'Longitude':DF2018.groupby(['MonthDay']).Longitude.mean(),
                     'VarianceLongitude':DF2018.groupby(['MonthDay']).Longitude.std(),
                     'VarianceLatitude':DF2018.groupby(['MonthDay']).Latitude.std()}).reset_index()


DAYS.to_csv('MeanLocationDays.csv')

DF2018.to_csv('Locations018.csv')

DAYS['TotalVariance'] = DAYS.VarianceLatitude + DAYS.VarianceLongitude

DAYS['Travel'] = DAYS['TotalVariance'].apply(lambda x: True if x>1 else False)

DAYS['Coordintes'] = DAYS['Latitude'].to_string()+', '+DAYS['Longitude'].to_string()

DAYS = DAYS[['MonthDay','Latitude','Longitude','TotalVariance','Travel']]

DAYS.to_csv('Days.csv')

DAYS['CoordinatesLT'] = DAYS['Latitude'].apply(lambda x: str(x))
DAYS['CoordinatesLG'] = DAYS['Longitude'].apply(lambda x: str(x))

DAYS['Coordinates'] = DAYS['CoordinatesLT'] +', ' + DAYS['CoordinatesLG'] 

DaysNotTravelling = DAYS[DAYS['Travel']==False].reset_index(drop=True)


DaysNotTravelling.Latitude.plot(kind='line',color='c', rot=0)


DaysNotTravelling.Longitude.plot(kind='line', color='r', rot=0)

DaysNotTravelling.Latitude.hist()


DaysNotTravelling.Longitude.hist()


#Clustering
from sklearn.cluster import KMeans


#number of clusters
k = 11
kmeans = KMeans(n_clusters=k)




data = DaysNotTravelling[['Latitude','Longitude']]


#
kmeans=kmeans.fit(data)


labels = kmeans.labels_



centroids = kmeans.cluster_centers_



colors = ['blue','red','green','black','cyan','yellow','purple','magenta','azure','grey','aqua']



from matplotlib import pyplot as plt


y = 0
for x in labels:
    plt.scatter(data.iloc[y,0], data.iloc[y,1],color=colors[x])
    y+=1
for x in range(k):
    lines = plt.plot(centroids[x,0],centroids[x,1],'kx')
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
t= 'Number of clusters (k) = ' + str(k)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()


centroids
places = []
countries = []
d = {}
for index, centroid in enumerate(centroids):
    pair = str (centroid[0]) +', '+ str (centroid[1])
    loc = geolocator.reverse((pair))
    places.append(loc)
    country = loc.address.split(',')[-1]
    countries.append(country)
    d[index] = country




DaysNotTravelling['Labels'] = labels



DaysNotTravelling['Labels'].hist()


DaysNotTravelling['Country'] = DaysNotTravelling['Labels'].apply(lambda x: d[x])

DaysNotTravelling['Country'].hist()




DaysNotTravelling.Country.value_counts()




DaysTravelling = DAYS[DAYS['Travel']==True].reset_index(drop=True)

DaysNotTravelling[DaysNotTravelling.TotalVariance == DaysNotTravelling.TotalVariance.max()]


locations['Weekday'] = locations['Date'].apply(lambda x: str(x.weekday()))

locations['hours'] = locations['Date'].apply(lambda x: str(x.hour))

travel = locations[locations['MonthDay'].isin(DaysTravelling['MonthDay'])]

travel[travel['MonthDay']=='3_12']['Latitude'].plot(kind='line', color='r', rot=0)


plt.Line2D(travel[travel['MonthDay']=='3_12']['Latitude'],travel[travel['MonthDay']=='3_12']['hours'])
plt.plot()




