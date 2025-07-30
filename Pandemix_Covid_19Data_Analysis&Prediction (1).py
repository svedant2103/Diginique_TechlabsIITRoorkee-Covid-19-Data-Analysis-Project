#!/usr/bin/env python
# coding: utf-8

# ** The objective of this capstone project is to analyze COVID-19 data from various sources, perform exploratory data analysis (EDA), and develop predictive models to forecast COVID-19 cases... **

# In[ ]:


# import all the necessary libraries for data analysis -->
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# STEP1 : DATA COLLECTION AND DATA PRE-PROCESSING --->

# # Data Collection:
# # Sources: Johns Hopkins University COVID-19 Data: Confirmed Cases,Deaths,& Recovered Cases
# # Load and Extract Data --> Load the datasets for confirmed cases, deaths, and recoveries, and preprocess them.
# # Load datasets ---->
# url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
# url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
# url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
# 
# # Read csv files
# df_confirmed = pd.read_csv(url_confirmed)
# df_deaths = pd.read_csv(url_deaths)
# df_recovered = pd.read_csv(url_recovered)
# 
# # Data Preprocessing
# # Melt the datasets to transform time series data into tabular format
# def preprocess_data(df, value_name):
#     df_melted = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
#                         var_name='Date', value_name=value_name)
#     df_melted['Date'] = pd.to_datetime(df_melted['Date'])
#     return df_melted
# 
# confirmed_melted = preprocess_data(df_confirmed, 'Confirmed')
# deaths_melted = preprocess_data(df_deaths, 'Deaths')
# recovered_melted = preprocess_data(df_recoveredrecovered, 'Recovered')

# In[ ]:


# Create a combined DataFrame --->
data = {
    'Date': pd.to_datetime(df_confirmed_global.index),
    'Confirmed': df_confirmed_global.values,
    'Deaths': df_deaths_global.values,
    'Recovered': df_recovered_global.values
}
df_combined = pd.DataFrame(data)

# Calculate active cases
df_combined['Active'] = df_combined['Confirmed'] - df_combined['Deaths'] - df_combined['Recovered']
df_combined.to_csv('covid19_combined_global.csv', index=False)


# In[ ]:


# read the combined dataset --->
df = pd.read_csv('covid19_combined_global.csv')
df.head(n=4)


# In[ ]:


# Transform the data -->
df.dtypes


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


# Data cleaning -->
df.isnull().sum()


# In[ ]:


# by observing there is no missing values --->


# STEP2 : EXPLORATORY DATA ANALYSIS (EDA) --->

# In[ ]:


# Perform EDA
df = pd.DataFrame(df)

# Merge 'Country/Region' from df_confirmed using the index as the key
df = pd.merge(df, df_confirmed[['Country/Region']], left_on='index', right_index=True)
df.head(n=4)

# Drop the original 'Country/Region' column and the index column
# df.drop(['Country/Region_x', 'index'], axis=1, inplace=True)
# df.rename(columns={'Country/Region_y': 'Country/Region'}, inplace=True)  # Rename for consistency
# df.head(n=4)

# Drop the 'Country/Region' column
# df.drop('Country/Region', axis=1, inplace=True)
# df.head(n=4)

# Drop the original 'Country/Region' column and the index column
df.drop(['Country/Region_x', 'index'], axis=1, inplace=True)
df.rename(columns={'Country/Region_y': 'Country/Region'}, inplace=True)  # Rename for consistency
df.head(n=4)


# In[ ]:


import matplotlib.pyplot as plt

# Plot time series
# Purpose: To visualize trends over time for confirmed, deaths, recovered, and active cases
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'], df_combined['Confirmed'], label='Confirmed', color='blue')
plt.plot(df_combined['Date'], df_combined['Deaths'], label='Deaths', color='red')
plt.plot(df_combined['Date'], df_combined['Recovered'], label='Recovered', color='green')
plt.plot(df_combined['Date'], df_combined['Active'], label='Active', color='orange')

plt.title('COVID-19 Global Trends')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().system('pip install plotly     # library for interactive & visuvally appealing library')


# In[ ]:


# Purpose: To visualize trends over time for confirmed, deaths, recovered, and active cases
import plotly.express as px

# Line chart for trends
fig = px.line(
    df_combined,
    x='Date',
    y=['Confirmed', 'Deaths', 'Recovered', 'Active'],
    labels={'value': 'Number of Cases', 'Date': 'Date'},
    title='COVID-19 Global Trends'
)
fig.update_layout(legend_title_text='Case Type', template='plotly_dark')
fig.show()


# In[ ]:


# Plot stacked area chart
# Purpose: To show the composition of confirmed cases (Active, Recovered, Deaths)
plt.figure(figsize=(12, 6))
plt.stackplot(
    df_combined['Date'],
    df_combined['Active'],
    df_combined['Recovered'],
    df_combined['Deaths'],
    labels=['Active', 'Recovered', 'Deaths'],
    colors=['orange', 'green', 'red']
)

plt.title('COVID-19 Cases Composition Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


# pie chart -->
fig = px.pie(
    df_combined,
    names=['Active', 'Recovered', 'Deaths'],
    values=df_combined[['Active', 'Recovered', 'Deaths']].sum(),
    title='COVID-19 Cases Composition Over Time',
    color_discrete_sequence=['orange', 'green', 'red']
)
fig.update_layout(legend_title_text='Case Type', template='plotly')
fig.show()


# In[ ]:


# Purpose: To compare total cases, deaths, recovered, and active cases on a specific date
# Data for the latest date
latest_data = df_combined.iloc[-1]
categories = ['Confirmed', 'Deaths', 'Recovered', 'Active']
values = [latest_data['Confirmed'], latest_data['Deaths'], latest_data['Recovered'], latest_data['Active']]

# Plot bar chart
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color=['blue', 'red', 'green', 'orange'])
plt.title(f'COVID-19 Global Cases on {latest_data["Date"].date()}')
plt.ylabel('Number of Cases')
plt.tight_layout()
plt.show()


# In[ ]:


# Calculate daily growth rate
# Purpose: To visualize the daily growth rate of confirmed cases.
df_combined['Daily Growth Rate (%)'] = df_combined['Confirmed'].pct_change() * 100

# Plot growth rate
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'], df_combined['Daily Growth Rate (%)'], color='purple', label='Daily Growth Rate')

plt.title('Daily Growth Rate of COVID-19 Confirmed Cases')
plt.xlabel('Date')
plt.ylabel('Growth Rate (%)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Calculate rates
df_combined['Death Rate (%)'] = (df_combined['Deaths'] / df_combined['Confirmed']) * 100
df_combined['Recovery Rate (%)'] = (df_combined['Recovered'] / df_combined['Confirmed']) * 100

# Plot death and recovery rates
# Purpose: To visualize the proportions of deaths and recoveries among confirmed cases.
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'], df_combined['Death Rate (%)'], label='Death Rate (%)', color='red')
plt.plot(df_combined['Date'], df_combined['Recovery Rate (%)'], label='Recovery Rate (%)', color='green')

plt.title('COVID-19 Death and Recovery Rates')
plt.xlabel('Date')
plt.ylabel('Rate (%)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


# Plot histogram
# Purpose: To show the distribution of active cases.
plt.figure(figsize=(10, 6))
plt.hist(df_combined['Active'], bins=20, color='orange', edgecolor='black')
plt.title('Distribution of Active COVID-19 Cases')
plt.xlabel('Active Cases')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


fig = px.histogram(
    df_combined,
    x='Active',
    nbins=20,
    labels={'Active': 'Active Cases'},
    title='Distribution of Active COVID-19 Cases',
    color_discrete_sequence=['orange']
)
fig.update_layout(template='plotly_dark')
fig.show()


# In[ ]:


# Purpose: To show the distribution of death cases.
plt.figure(figsize=(10, 6))
plt.hist(df_combined['Deaths'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Death COVID-19 Cases')
plt.xlabel('Death Cases')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


fig = px.histogram(
    df_combined,
    x='Deaths',
    nbins=20,
    labels={'Deaths': 'Death Cases'},
    title='Distribution of Death COVID-19 Cases',
    color_discrete_sequence=['green']
)
fig.update_layout(template='plotly_dark')
fig.show()


# In[ ]:


# Purpose: To show the distribution of Recovered cases.
plt.figure(figsize=(10, 6))
plt.hist(df_combined['Recovered'], bins=20, color='red', edgecolor='black')
plt.title('Distribution of Recovered COVID-19 Cases')
plt.xlabel('Recovered Cases')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


fig = px.histogram(
    df_combined,
    x='Recovered',
    nbins=20,
    labels={'Recovered': 'Recovered Cases'},
    title='Distribution of Recovered COVID-19 Cases',
    color_discrete_sequence=['yellow']
)
fig.update_layout(template='plotly_dark')
fig.show()


# In[ ]:


# Calculate 7-day rolling average for active cases
df_combined['Active 7-Day Avg'] = df_combined['Active'].rolling(window=7).mean()

# Plot rolling average
# Purpose: To smooth trends using a rolling average.
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'], df_combined['Active'], label='Active Cases', color='orange', alpha=0.5)
plt.plot(df_combined['Date'], df_combined['Active 7-Day Avg'], label='7-Day Rolling Avg', color='blue')

plt.title('7-Day Rolling Average of Active COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# STEP3: FEATURE ENGINEERING --->

# In[ ]:


# Add simple time-based features to help models understand temporal patterns.
# Add time-based features
df_combined['Quarter'] = df_combined['Date'].dt.quarter
df_combined['Month'] = df_combined['Date'].dt.month
df_combined['Day'] = df_combined['Date'].dt.day
df_combined['Day of Week'] = df_combined['Date'].dt.dayofweek


# In[ ]:


# calulate daily new cases --->
# Compute the difference in confirmed cases between consecutive days.
# Daily new cases
df_combined['New_Cases'] = df_combined['Confirmed'].diff().fillna(0)


# In[ ]:


# 7-day moving average --> Calculate a 7-day moving average to smooth the daily new cases.
df_combined['7_Day_Avg'] = df_combined['New_Cases'].rolling(window=7).mean().fillna(0)


# In[ ]:


# Daily growth rate --> Calculate the daily growth rate as a percentage.
df_combined['Growth_Rate (%)'] = df_combined['New_Cases'] / df_combined['Confirmed'].shift(1) * 100
df_combined['Growth_Rate (%)'] = df_combined['Growth_Rate (%)'].fillna(0)


# In[ ]:


# Save the engineered dataset
df_combined.to_csv('covid19_simple_engineered.csv', index=False)


# In[ ]:


print(df_combined.columns)


# 
# STEP4: MODAL DEVELOPMENT --->
# 
# 
# 
# 

# STEP5: MODAL EVALUATION --->
# 
# 

# In[ ]:


from sklearn.model_selection import cross_val_score

# Assuming 'model' is your LinearRegression model and 'X' and 'y' are your features and target
scores = cross_val_score(model, X, y, cv=2, scoring='neg_mean_squared_error')  # 5-fold cross-validation
rmse_scores = np.sqrt(-scores)  # Convert negative MSE scores to RMSE

print('RMSE scores for each fold:', rmse_scores)
print('Average RMSE:', rmse_scores.mean())


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Line plot of confirmed cases over time for a specific country
plt.figure(figsize=(12, 6))
country_data = data[data['Country/Region'] == 'India']
plt.plot(country_data['Date'], country_data['Confirmed'])
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title('COVID-19 Confirmed Cases in India')
plt.show()



# Correlation heatmap
plt.figure(figsize=(10, 8))
# Select only numerical columns for correlation calculation
numerical_data = data.select_dtypes(include=['float', 'int'])
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


# Line plot of death cases over time for a specific country -->

plt.figure(figsize=(12, 6))
country_data = data[data['Country/Region'] == 'India']
plt.plot(country_data['Date'], country_data['Deaths'])
plt.xlabel('Date')
plt.ylabel('Deaths Cases')
plt.title('COVID-19 Death Cases in India')
plt.show()


# In[ ]:


# Line plot of Recovered cases over time for a specific country -->

plt.figure(figsize=(12, 6))
country_data = data[data['Country/Region'] == 'India']
plt.plot(country_data['Date'], country_data['Recovered'])
plt.xlabel('Date')
plt.ylabel('Recovered Cases')
plt.title('COVID-19 Recovered Cases in India')
plt.show()


# In[ ]:


import plotly.express as px

fig = px.bar(data, x='Country/Region', y='Confirmed', color='Country/Region',opacity=0.8,
             title='Total Confirmed COVID-19 Cases by Country')
fig.show()


# In[ ]:


# Include features like geographical maps, time series plots, and trend analysis to enhance understanding --->
# 1.> time-series plots --->
# confirmed cases over time

import plotly.express as px

fig = px.line(data, x='Date', y='Confirmed', color='Country/Region',
              title='COVID-19 Confirmed Cases Over Time')
fig.show()


# In[ ]:


# deaths cases over time --->
import plotly.express as px

fig = px.line(data, x='Date', y='Deaths', color='Country/Region',
              title='COVID-19 Deaths Cases Over Time')
fig.show()


# In[ ]:


# Recovered cases over time ---->
import plotly.express as px

fig = px.line(data, x='Date', y='Recovered', color='Country/Region',
              title='COVID-19 Recovered Cases Over Time')
fig.show()


# In[ ]:


# 2.> Trend Analysis --->
# 7-Day Moving Average -->
data['7DayAvgConfirmed'] = data['Confirmed'].rolling(window=7).mean()

fig = px.line(data, x='Date', y='7DayAvgConfirmed', color='Country/Region',
              title='7-Day Average of COVID-19 Confirmed Cases')
fig.show()


# In[ ]:


# Growth Rate --->
data['GrowthRate'] = data['Confirmed'].pct_change() * 100

fig = px.line(data, x='Date', y='GrowthRate', color='Country/Region',
              title='Daily Growth Rate of COVID-19 Confirmed Cases')
fig.show()

