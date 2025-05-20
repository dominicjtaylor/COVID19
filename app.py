import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import datetime
import time
from IPython.display import display
from datetime import date
from datetime import datetime
import streamlit as st
import pydeck as pdk
import altair as alt
from datetime import timedelta
import pydeck as pdk


st.title('COVID-19')
st.info('NOTICE: Click the Sidebar to change preferences. Once the Map is in view, you must click on it for accurate location.')

st.header('Daily COVID-19-related Deaths per Million')


df = pd.read_csv('https://raw.githubusercontent.com/tomwhite/covid-19-uk-data/master/data/covid-19-totals-uk.csv',error_bad_lines=False)

df_deaths = df[['Date', 'Tests', 'ConfirmedCases', 'Deaths']]
df_deaths = df_deaths.rename(columns={'ConfirmedCases': 'Confirmed Cases'})


df_daily = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv',error_bad_lines=False)

df_deaths = df_daily[['location', 'date', 'new_deaths_per_million']]
df_deaths = df_deaths.rename(columns={'location':'Country','date':'Date','new_deaths_per_million':'New Deaths per Million'})
df_deaths = df_deaths[['Date','Country','New Deaths per Million']]


by_entity = df_deaths.groupby(["Country"])

df_by_entity = pd.DataFrame(by_entity.size().reset_index())
df_by_entity = df_by_entity.drop(df_by_entity.columns[-1],axis=1)

list_names = df_deaths.groupby('Country')
names = []
for name, name_df in list_names:
    #print(name)
    names.append(name)
#print(names)

dic1 = {}
for x in names:
    dic1["{0}".format(x)]=by_entity.get_group(x)


st.sidebar.header('Daily COVID-19-related Deaths per Million')
input_country1 = st.sidebar.multiselect('Select the countries you wish to compare:',names,default=['United Kingdom'],key='box2')

subset_data1 = pd.DataFrame()
if len(input_country1) > 0:
    subset_data1 = df_deaths[df_deaths['Country'].isin(input_country1)]

by_sub1 = subset_data1.groupby(["Country"])
df_by_sub1 = pd.DataFrame(by_sub1.size().reset_index())
df_by_sub1 = df_by_sub1.drop(df_by_sub1.columns[-1],axis=1)


df_locus = pd.read_csv('https://gist.githubusercontent.com/tadast/8827699/raw/'
                       '3cd639fa34eec5067080a61c69e3ae25e3076abb/countries_codes_and_coordinates.csv')

df_locus = df_locus.rename(columns={'Alpha-3 code':'ISO','Latitude (average)':'Latitude','Longitude (average)':'Longitude'})
df_locus.drop(columns=["Alpha-2 code","Numeric code"], inplace=True)
df_locus['ISO'] = df_locus['ISO'].str.replace(r"[\"]", '')
df_locus['Latitude'] = df_locus['Latitude'].str.replace(r"[\"]", '')
df_locus['Longitude'] = df_locus['Longitude'].str.replace(r"[\"]", '')
df_locus['Latitude'] = df_locus['Latitude'].str.lstrip()
df_locus['Longitude'] = df_locus['Longitude'].str.lstrip()
df_locus['Latitude'] = pd.to_numeric(df_locus['Latitude'], downcast="float")
df_locus['Longitude'] = pd.to_numeric(df_locus['Longitude'], downcast="float")


dates = pd.date_range(start="2019-12-31",end=datetime.today()-timedelta(days=1)).to_list()
date = []
for i in dates:
    string = i.strftime("%Y-%m-%d")
    date.append(string)


list_country = df_locus.groupby('Country')
countries = []
for name, name_df in list_country:
    #print(name)
    countries.append(name)
#print(countries)

dic_country = {}
for x in countries:
    dic_country["{0}".format(x)]=list_country.get_group(x)

frames=[]
for i in range(0,len(df_by_sub1['Country'])):
    frames.append(dic_country[df_by_sub1['Country'][i]])

df_sel = pd.concat(frames)
df_sel1 = df_sel.drop(columns='ISO')

list = df_sel.groupby('Country')
count = []
for name, name_df in list:
    #print(name)
    count.append(name)
#print(count)

dic_1 = {}
for x in count:
    dic_1["{0}".format(x)]=list.get_group(x)


for n in count:
    latt = dic_1[n].loc[dic_1[n]['Country'] == n, 'Latitude'].iloc[0]
    lonn = dic_1[n].loc[dic_1[n]['Country'] == n, 'Longitude'].iloc[0]
#     st.write(latt)
    subset_data1.loc[subset_data1['Country']==n,'Latitude']=latt
    subset_data1.loc[subset_data1['Country']==n,'Longitude']=lonn
subset_data1['New Deaths per Million'] = subset_data1['New Deaths per Million'].fillna(0)
subset_data1['New Deaths per Million'] = subset_data1['New Deaths per Million'].replace(to_replace=0, method='ffill')
subset_data1 = subset_data1.reset_index()
subset_data1 = subset_data1.drop(columns='index')
#st.write('subset_data3:',pd.DataFrame(subset_data3.iloc[subset_data3[subset_data3['Date']=='2019-12-31'].index[0]]).transpose())
#st.write(subset_data3[subset_data3['Date']=='2019-12-31'])
#grou = subset_data3[subset_data3['Date']=='2019-12-31'].groupby(['Country'])
subset_data1['Radius']=''

for i in range(0,len(subset_data1)):
    subset_data1['Radius'].iloc[i]=subset_data1['New Deaths per Million'].iloc[i]*(1.2e6)

    
###########################
grouped1 = subset_data1.groupby(['Country'])
#st.write(grouped.iloc[0])
df_grouped1 = pd.DataFrame(grouped1.size().reset_index())
#st.write('df_grouped:',df_grouped)

namess1 = []
for name, name_df in grouped1:
    #print(name)
    namess1.append(name)
#st.write(namesss)

dic1 = {}
for x in namess1:
    dic1["{0}".format(x)]=grouped1.get_group(x)

###########################    
frame=[]
for i in range(0,len(df_grouped1['Country'])):
    frame.append(dic_country[df_grouped1['Country'][i]])

df_subset1 = pd.concat(frame)
df_subset1 = df_subset1.drop(columns='ISO')
#st.write('df_subset:',df_subset)

list11 = df_subset1.groupby('Country')
counting1 = []
for name, name_df in list11:
    #print(name)
    counting1.append(name)


dict_choice1 = {key: dic1[key] for key in dic1.keys() & set(dic_1.keys())}


xmin1 = st.sidebar.selectbox('Choose a start date:',date,key='box1.1')


#speed = 1/(st.slider('Speed of evolution',1,20))
                  
if st.button('Show Evolving Map',key='1.3'):
    datedate = datetime(2019,12,31)

    view = pdk.ViewState(latitude=54,longitude=-2,zoom=0,)

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=subset_data1[subset_data1['Date']=='2019-12-31'],
        get_position=['Longitude', 'Latitude'],
        pickable=False,
        opacity=0.1,
        stroked=True,
        filled=True,
        line_width_min_pixels=3,
        elevation_scale=4,
        get_radius='Radius',
        get_fill_color='[220, 0, 3]',
        get_line_color='[500,0,3]',
        tooltip="test test",
    )

    r = pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v9',
            initial_view_state=view,
            layers=[layer],
    )

    subheading = st.subheader("")

    map = st.pydeck_chart(r)
    
    for i in date:
        datedate += timedelta(days=1)
        layer.data = subset_data1[subset_data1['Date']==i]
        r.update()
        map.pydeck_chart(r)
        subheading.subheader("Daily Deaths per Million on : %s" % (datedate.strftime("%B %d, %Y")))
        time.sleep(0.1)


fig = plt.figure(figsize=(14,10))

for i in dict_choice1.values():
    plt.plot(pd.to_datetime(i['Date'], format = '%Y-%m-%d'),i['New Deaths per Million'],label=i['Country'].to_list()[0])
#plt.xlabel('Date',fontsize=16)
plt.ylabel('Daily Deaths per Million',fontsize=22)
plt.legend(fontsize=22)
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.xlim(xmin=xmin1)
plt.ylim(ymin=0)
st.pyplot()


############################################################################################################################
st.header('Daily COVID-19 Tests per Thousand')


#df = pd.read_excel('covid-testing-all-observations.xlsx')
df = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv',error_bad_lines=False)
df_tests = df[['Entity', 'Date', 'Daily change in cumulative total per thousand']]
#display(df_tests.head(3))


pd.options.mode.chained_assignment = None

df_tests["Entity"] = df_tests["Entity"].astype(str)
new = df_tests["Entity"].str.split('-',n=1,expand=True)
df_tests.drop(columns=["Entity"], inplace=True)
df_tests["Entity"]= new[0]
df_tests["Units"]= new[1]

df_tests.drop_duplicates(subset =["Date","Entity"], keep = 'first', inplace = True)
df_tests.dropna(subset=["Daily change in cumulative total per thousand"],inplace=True)
df_tests.dropna(subset=["Date"],inplace=True)
df_tests['Entity'] = df_tests['Entity'].str.rstrip()
df_tests = df_tests.rename(columns={'Entity':'Country'})


by_entity2 = df_tests.groupby(["Country"])

df_by_entity2 = pd.DataFrame(by_entity2.size().reset_index())
df_by_entity2 = df_by_entity2.drop(df_by_entity2.columns[-1],axis=1)

list_names = df_tests.groupby('Country')
names2 = []
for name, name_df in list_names:
    #print(name)
    names2.append(name)

dic2 = {}
for x in names2:
    dic2["{0}".format(x)]=by_entity2.get_group(x)
#print(dic)


st.sidebar.header('Daily COVID-19 Tests per Thousand')
input_country2 = st.sidebar.multiselect('Select the countries you wish to compare:',names2,default=['United Kingdom'],key='box1')

subset_data2 = pd.DataFrame()
if len(input_country2) > 0:
    subset_data2 = df_tests[df_tests['Country'].isin(input_country2)]

by_sub2 = subset_data2.groupby(["Country"])
df_by_sub2 = pd.DataFrame(by_sub2.size().reset_index())
df_by_sub2 = df_by_sub2.drop(df_by_sub2.columns[-1],axis=1)


frames2=[]
for i in range(0,len(df_by_sub2['Country'])):
    frames2.append(dic_country[df_by_sub2['Country'][i]])

df_sel2 = pd.concat(frames2)
df_sel2 = df_sel2.drop(columns='ISO')

list2 = df_sel2.groupby('Country')
count2 = []
for name, name_df in list2:
    #print(name)
    count2.append(name)

dic_2 = {}
for x in count2:
    dic_2["{0}".format(x)]=list2.get_group(x)

for n in count2:
    latt = dic_2[n].loc[dic_2[n]['Country'] == n, 'Latitude'].iloc[0]
    lonn = dic_2[n].loc[dic_2[n]['Country'] == n, 'Longitude'].iloc[0]
#     st.write(latt)
    subset_data2.loc[subset_data2['Country']==n,'Latitude']=latt
    subset_data2.loc[subset_data2['Country']==n,'Longitude']=lonn
subset_data2['Daily change in cumulative total per thousand'] = subset_data2['Daily change in cumulative total per thousand'].fillna(0)
subset_data2['Daily change in cumulative total per thousand'] = subset_data2['Daily change in cumulative total per thousand'].replace(to_replace=0, method='ffill')
subset_data2 = subset_data2.reset_index()
subset_data2 = subset_data2.drop(columns='index')
#st.write('subset_data3:',pd.DataFrame(subset_data3.iloc[subset_data3[subset_data3['Date']=='2019-12-31'].index[0]]).transpose())
#st.write(subset_data3[subset_data3['Date']=='2019-12-31'])
#grou = subset_data3[subset_data3['Date']=='2019-12-31'].groupby(['Country'])
subset_data2['Radius']=''

for i in range(0,len(subset_data2)):
    subset_data2['Radius'].iloc[i]=subset_data2['Daily change in cumulative total per thousand'].iloc[i]*(3e6)

    
###########################
grouped2 = subset_data2.groupby(['Country'])
#st.write(grouped.iloc[0])
df_grouped2 = pd.DataFrame(grouped2.size().reset_index())
#st.write('df_grouped:',df_grouped)

namess2 = []
for name, name_df in grouped2:
    #print(name)
    namess2.append(name)
#st.write(namesss)

dic2 = {}
for x in namess2:
    dic2["{0}".format(x)]=grouped2.get_group(x)

###########################    
frame2=[]
for i in range(0,len(df_grouped2['Country'])):
    frame2.append(dic_country[df_grouped2['Country'][i]])

df_subset2 = pd.concat(frame)
df_subset2 = df_subset2.drop(columns='ISO')
#st.write('df_subset:',df_subset)

list22 = df_subset2.groupby('Country')
counting2 = []
for name, name_df in list22:
    #print(name)
    counting2.append(name)


dict_choice2 = {key: dic2[key] for key in dic2.keys() & set(dic_2.keys())}

xmin2 = st.sidebar.selectbox('Choose a start date:',date,key='box2.1')


#speed = 1/(st.slider('Speed of evolution',1,20))
                  
if st.button('Show Evolving Map',key='2.3'):
    datedate = datetime(2019,12,31)

    view = pdk.ViewState(latitude=54,longitude=-2,zoom=0,)

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=subset_data2[subset_data2['Date']=='2019-12-31'],
        get_position=['Longitude', 'Latitude'],
        pickable=False,
        opacity=0.1,
        stroked=True,
        filled=True,
        line_width_min_pixels=3,
        elevation_scale=4,
        get_radius='Radius',
        get_fill_color='[220, 0, 3]',
        get_line_color='[500,0,3]',
        tooltip="test test",
    )

    r = pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v9',
            initial_view_state=view,
            layers=[layer],
    )

    subheading = st.subheader("")

    map = st.pydeck_chart(r)
    
    for i in date:
        datedate += timedelta(days=1)
        layer.data = subset_data2[subset_data2['Date']==i]
        r.update()
        map.pydeck_chart(r)
        subheading.subheader("Daily number of tests per thousand : %s" % (datedate.strftime("%B %d, %Y")))
        time.sleep(0.1)
    

fig = plt.figure(figsize=(12,8))

for i in dict_choice2.values():
    plt.plot(pd.to_datetime(i['Date'], format = '%Y-%m-%d'),i['Daily change in cumulative total per thousand'],label=i['Country'].to_list()[0])
#plt.xlabel('Date',fontsize=16)
plt.ylabel('Daily Tests per Thousand',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=16,rotation=45)
plt.yticks(fontsize=16)
plt.ylim(ymin=0)
plt.xlim(xmin=xmin2)
#plt.yticks(np.arange(0,max(DCTPT)+2,2))
st.pyplot()


############################################################################################################################
st.header('Daily Confirmed COVID-19 Cases')


df_cases = df_daily.copy()
df_cases = df_cases[['location', 'date', 'new_cases_per_million']]
df_cases = df_cases.rename(columns={'location':'Country','date':'Date','new_cases_per_million':'New Cases per Million'})
df_cases = df_cases[['Date','Country','New Cases per Million']]


by_entity_case = df_cases.groupby(["Country"])

df_by_entity_case = pd.DataFrame(by_entity_case.size().reset_index())
df_by_entity_case = df_by_entity_case.drop(df_by_entity_case.columns[-1],axis=1)

list_names3 = df_cases.groupby('Country')
names3 = []
for name, name_df in list_names3:
    #print(name)
    names3.append(name)
#print(names)

dic3 = {}
for x in names3:
    dic3["{0}".format(x)]=by_entity_case.get_group(x)


st.sidebar.header('Daily Confirmed COVID-19 Cases')
input_country3 = st.sidebar.multiselect('Select the countries you wish to compare:',names3,default=['United Kingdom'],key='box3')

subset_data3 = pd.DataFrame()
if len(input_country3) > 0:
    subset_data3 = df_cases[df_cases['Country'].isin(input_country3)]
subset_data3['Latitude']=""
subset_data3['Longitude']=""

by_sub3 = subset_data3.groupby(["Country"])
df_by_sub3 = pd.DataFrame(by_sub3.size().reset_index())
df_by_sub3 = df_by_sub3.drop(df_by_sub3.columns[-1],axis=1)


frames3=[]
for i in range(0,len(df_by_sub3['Country'])):
    frames3.append(dic_country[df_by_sub3['Country'][i]])

df_sel3 = pd.concat(frames3)
df_sel3 = df_sel3.drop(columns='ISO')

list3 = df_sel3.groupby('Country')
count3 = []
for name, name_df in list3:
    #print(name)
    count3.append(name)

dic_3 = {}
for x in count3:
    dic_3["{0}".format(x)]=list3.get_group(x)

for n in count3:
    latt = dic_3[n].loc[dic_3[n]['Country'] == n, 'Latitude'].iloc[0]
    lonn = dic_3[n].loc[dic_3[n]['Country'] == n, 'Longitude'].iloc[0]
#     st.write(latt)
    subset_data3.loc[subset_data3['Country']==n,'Latitude']=latt
    subset_data3.loc[subset_data3['Country']==n,'Longitude']=lonn
subset_data3['New Cases per Million'] = subset_data3['New Cases per Million'].fillna(0)
subset_data3['New Cases per Million'] = subset_data3['New Cases per Million'].replace(to_replace=0, method='ffill')
subset_data3 = subset_data3.reset_index()
subset_data3 = subset_data3.drop(columns='index')
#st.write('subset_data3:',pd.DataFrame(subset_data3.iloc[subset_data3[subset_data3['Date']=='2019-12-31'].index[0]]).transpose())
#st.write(subset_data3[subset_data3['Date']=='2019-12-31'])
#grou = subset_data3[subset_data3['Date']=='2019-12-31'].groupby(['Country'])
subset_data3['Radius']=''

for i in range(0,len(subset_data3)):
    subset_data3['Radius'].iloc[i]=subset_data3['New Cases per Million'].iloc[i]*(3e4)

    
###########################
grouped3 = subset_data3.groupby(['Country'])
#st.write(grouped.iloc[0])
df_grouped3 = pd.DataFrame(grouped3.size().reset_index())
#st.write('df_grouped:',df_grouped)

namess3 = []
for name, name_df in grouped3:
    #print(name)
    namess3.append(name)
#st.write(namesss)

dic3 = {}
for x in namess3:
    dic3["{0}".format(x)]=grouped3.get_group(x)

###########################    
frame=[]
for i in range(0,len(df_grouped3['Country'])):
    frame.append(dic_country[df_grouped3['Country'][i]])

df_subset3 = pd.concat(frame)
df_subset3 = df_subset3.drop(columns='ISO')
#st.write('df_subset:',df_subset)

list33 = df_subset3.groupby('Country')
counting3 = []
for name, name_df in list33:
    #print(name)
    counting3.append(name)


dict_choice3 = {key: dic3[key] for key in dic3.keys() & set(dic_3.keys())}

xmin3 = st.sidebar.selectbox('Choose a start date:',date,key='box3.1')


#speed = 1/(st.slider('Speed of evolution',1,20))
                  
if st.button('Show Evolving Map',key='3.3'):
    datedate = datetime(2019,12,31)

    view = pdk.ViewState(latitude=54,longitude=-2,zoom=0,)

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=subset_data3[subset_data3['Date']=='2019-12-31'],
        get_position=['Longitude', 'Latitude'],
        pickable=False,
        opacity=0.1,
        stroked=True,
        filled=True,
        line_width_min_pixels=3,
        elevation_scale=4,
        get_radius='Radius',
        get_fill_color='[220, 0, 3]',
        get_line_color='[500,0,3]',
        tooltip="test test",
    )

    r = pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v9',
            initial_view_state=view,
            layers=[layer],
    )

    subheading = st.subheader("")

    map = st.pydeck_chart(r)
    
    for i in date:
        datedate += timedelta(days=1)
        layer.data = subset_data3[subset_data3['Date']==i]
        r.update()
        map.pydeck_chart(r)
        subheading.subheader("Daily Cases per Million on : %s" % (datedate.strftime("%B %d, %Y")))
        time.sleep(0.1)


fig = plt.figure(figsize=(12,8))

for i in dict_choice3.values():
    plt.plot(pd.to_datetime(i['Date'], format = '%Y-%m-%d'),i['New Cases per Million'],label=i['Country'].to_list()[0])
#plt.xlabel('Date',fontsize=16)
plt.ylabel('Daily Cases per Million',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=16,rotation=45)
plt.yticks(fontsize=16)
plt.xlim(xmin=xmin3)
plt.ylim(ymin=0)

st.pyplot()

