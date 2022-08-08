# Case Study 2: How Can a Wellness Technology Company Play It Smart?

## By: Amanda Wright
### Last updated: May 31, 2022
#### Submitted as part of Grow with Google Data Analytics Certificate

   # Ask
   
    
<b>Business task:</b> <br> <br>
To discover trends in smart device usage and use findings to inform Bellabeat marketing strategies.
    
<b>Key stakeholders:</b> 
* Bellabeat co-founder and Chief Creative Officer, Urška Sršen <br>
* Bellabeat co-founder and mathematician, Sando Mur <br>
* Bellabeat marketing analytics team <br>

# Prepare

<b>Data sources:</b> <br> <br>
The data is sourced from [Kaggle](https://www.kaggle.com/datasets/arashnic/fitbit?datasetId=1041311&searchQuery=sql). 33 Fitbit users responded to a survey distributed by Amazon Mechanical Turk. From 03/12/2016-05/12/2016, users consented to submitting their tracking data, including physical activity, heart rate, steps, and more.

<b>Limitations of the data:</b> <br><br>
The sample size is only 33 users and some tables do not include all users. An additional limitation is the lack of demographic data - age, gender, etc. Without gender specific data, we are unable to isolate specific findings for Bellabeat's target audience. In the future, using larger datasets would help further analysis.




As part of the prepare process, I first imported the necessary libraries, then imported each csv file as a Pandas data frame.


```python
# import libraries
import numpy as np # mathematical functions
import pandas as pd # data analysis/manipulation
from sklearn import linear_model # linear regressions, part of Pandas package
import pandasql as ps # sql
import matplotlib.pyplot as plt # creates visualizations
import seaborn as sns # creates visualizations
import datetime # manipulates datetimes
import scipy # runs statistical analysis

```

 To be able to manipulate the data, I created a copy of each dataframe to ensure that I would maintain the integrity of the original data. 


```python
# import each csv as a raw df
dailyActivity_merged_raw = pd.read_csv('/Users/amanda/Downloads/dailyActivity_merged.csv')
dailyActivity_merged_clean = dailyActivity_merged_raw.copy()

dailyCalories_merged_raw = pd.read_csv('/Users/amanda/Downloads/dailyCalories_merged.csv')
dailyCalories_merged_clean = dailyCalories_merged_raw.copy()

dailyIntensities_merged_raw = pd.read_csv('/Users/amanda/Downloads/dailyIntensities_merged.csv')
dailyIntensities_merged_clean = dailyIntensities_merged_raw.copy()

dailySteps_merged_raw = pd.read_csv('/Users/amanda/Downloads/dailySteps_merged.csv')
dailySteps_merged_clean = dailySteps_merged_raw.copy()

hourlyCalories_merged_raw = pd.read_csv('/Users/amanda/Downloads/hourlyCalories_merged.csv')
hourlyCalories_merged_clean = hourlyCalories_merged_raw.copy()

hourlyIntensities_merged_raw = pd.read_csv('/Users/amanda/Downloads/hourlyIntensities_merged.csv')
hourlyIntensities_merged_clean = hourlyIntensities_merged_raw.copy()

hourlySteps_merged_raw = pd.read_csv('/Users/amanda/Downloads/hourlySteps_merged.csv')
hourlySteps_merged_clean = hourlySteps_merged_raw.copy()

sleepDay_merged_raw = pd.read_csv('/Users/amanda/Downloads/sleepDay_merged.csv')
sleepDay_merged_clean = sleepDay_merged_raw.copy()

weightLogInfo_merged_raw = pd.read_csv('/Users/amanda/Downloads/weightLogInfo_merged.csv')
weightLogInfo_merged_clean = weightLogInfo_merged_raw.copy()

```

# Process

As part of the data cleaning process, I first looked at each table to check for any null values as well as duplicated values.  


```python
# create dict to easily loop through each csv file
clean_data_dict = {'dailyActivity_merged_clean': dailyActivity_merged_clean, 
                'dailyCalories_merged_clean': dailyCalories_merged_clean, 
                'dailyIntensities_merged_clean': dailyIntensities_merged_clean, 
                'dailySteps_merged_clean': dailySteps_merged_clean,
                'hourlyCalories_merged_clean': hourlyCalories_merged_clean, 
                'hourlyIntensities_merged_clean': hourlyIntensities_merged_clean, 
                'hourlySteps_merged_clean': hourlySteps_merged_clean, 
                'sleepDay_merged_clean': sleepDay_merged_clean, 
                'weightLogInfo_merged_clean': weightLogInfo_merged_clean}

# identify how data is organized
for (name, x) in clean_data_dict.items():
    print(name)
    display(x.info())
    display(x.head())
    display(x.duplicated())
```

    dailyActivity_merged_clean
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 940 entries, 0 to 939
    Data columns (total 15 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Id                        940 non-null    int64  
     1   ActivityDate              940 non-null    object 
     2   TotalSteps                940 non-null    int64  
     3   TotalDistance             940 non-null    float64
     4   TrackerDistance           940 non-null    float64
     5   LoggedActivitiesDistance  940 non-null    float64
     6   VeryActiveDistance        940 non-null    float64
     7   ModeratelyActiveDistance  940 non-null    float64
     8   LightActiveDistance       940 non-null    float64
     9   SedentaryActiveDistance   940 non-null    float64
     10  VeryActiveMinutes         940 non-null    int64  
     11  FairlyActiveMinutes       940 non-null    int64  
     12  LightlyActiveMinutes      940 non-null    int64  
     13  SedentaryMinutes          940 non-null    int64  
     14  Calories                  940 non-null    int64  
    dtypes: float64(7), int64(7), object(1)
    memory usage: 110.3+ KB



    None



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
      <th>Id</th>
      <th>ActivityDate</th>
      <th>TotalSteps</th>
      <th>TotalDistance</th>
      <th>TrackerDistance</th>
      <th>LoggedActivitiesDistance</th>
      <th>VeryActiveDistance</th>
      <th>ModeratelyActiveDistance</th>
      <th>LightActiveDistance</th>
      <th>SedentaryActiveDistance</th>
      <th>VeryActiveMinutes</th>
      <th>FairlyActiveMinutes</th>
      <th>LightlyActiveMinutes</th>
      <th>SedentaryMinutes</th>
      <th>Calories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1503960366</td>
      <td>4/12/2016</td>
      <td>13162</td>
      <td>8.50</td>
      <td>8.50</td>
      <td>0.0</td>
      <td>1.88</td>
      <td>0.55</td>
      <td>6.06</td>
      <td>0.0</td>
      <td>25</td>
      <td>13</td>
      <td>328</td>
      <td>728</td>
      <td>1985</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1503960366</td>
      <td>4/13/2016</td>
      <td>10735</td>
      <td>6.97</td>
      <td>6.97</td>
      <td>0.0</td>
      <td>1.57</td>
      <td>0.69</td>
      <td>4.71</td>
      <td>0.0</td>
      <td>21</td>
      <td>19</td>
      <td>217</td>
      <td>776</td>
      <td>1797</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1503960366</td>
      <td>4/14/2016</td>
      <td>10460</td>
      <td>6.74</td>
      <td>6.74</td>
      <td>0.0</td>
      <td>2.44</td>
      <td>0.40</td>
      <td>3.91</td>
      <td>0.0</td>
      <td>30</td>
      <td>11</td>
      <td>181</td>
      <td>1218</td>
      <td>1776</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1503960366</td>
      <td>4/15/2016</td>
      <td>9762</td>
      <td>6.28</td>
      <td>6.28</td>
      <td>0.0</td>
      <td>2.14</td>
      <td>1.26</td>
      <td>2.83</td>
      <td>0.0</td>
      <td>29</td>
      <td>34</td>
      <td>209</td>
      <td>726</td>
      <td>1745</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1503960366</td>
      <td>4/16/2016</td>
      <td>12669</td>
      <td>8.16</td>
      <td>8.16</td>
      <td>0.0</td>
      <td>2.71</td>
      <td>0.41</td>
      <td>5.04</td>
      <td>0.0</td>
      <td>36</td>
      <td>10</td>
      <td>221</td>
      <td>773</td>
      <td>1863</td>
    </tr>
  </tbody>
</table>
</div>



    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    935    False
    936    False
    937    False
    938    False
    939    False
    Length: 940, dtype: bool


    dailyCalories_merged_clean
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 940 entries, 0 to 939
    Data columns (total 3 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   Id           940 non-null    int64 
     1   ActivityDay  940 non-null    object
     2   Calories     940 non-null    int64 
    dtypes: int64(2), object(1)
    memory usage: 22.2+ KB



    None



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
      <th>Id</th>
      <th>ActivityDay</th>
      <th>Calories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1503960366</td>
      <td>4/12/2016</td>
      <td>1985</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1503960366</td>
      <td>4/13/2016</td>
      <td>1797</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1503960366</td>
      <td>4/14/2016</td>
      <td>1776</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1503960366</td>
      <td>4/15/2016</td>
      <td>1745</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1503960366</td>
      <td>4/16/2016</td>
      <td>1863</td>
    </tr>
  </tbody>
</table>
</div>



    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    935    False
    936    False
    937    False
    938    False
    939    False
    Length: 940, dtype: bool


    dailyIntensities_merged_clean
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 940 entries, 0 to 939
    Data columns (total 10 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Id                        940 non-null    int64  
     1   ActivityDay               940 non-null    object 
     2   SedentaryMinutes          940 non-null    int64  
     3   LightlyActiveMinutes      940 non-null    int64  
     4   FairlyActiveMinutes       940 non-null    int64  
     5   VeryActiveMinutes         940 non-null    int64  
     6   SedentaryActiveDistance   940 non-null    float64
     7   LightActiveDistance       940 non-null    float64
     8   ModeratelyActiveDistance  940 non-null    float64
     9   VeryActiveDistance        940 non-null    float64
    dtypes: float64(4), int64(5), object(1)
    memory usage: 73.6+ KB



    None



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
      <th>Id</th>
      <th>ActivityDay</th>
      <th>SedentaryMinutes</th>
      <th>LightlyActiveMinutes</th>
      <th>FairlyActiveMinutes</th>
      <th>VeryActiveMinutes</th>
      <th>SedentaryActiveDistance</th>
      <th>LightActiveDistance</th>
      <th>ModeratelyActiveDistance</th>
      <th>VeryActiveDistance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1503960366</td>
      <td>4/12/2016</td>
      <td>728</td>
      <td>328</td>
      <td>13</td>
      <td>25</td>
      <td>0.0</td>
      <td>6.06</td>
      <td>0.55</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1503960366</td>
      <td>4/13/2016</td>
      <td>776</td>
      <td>217</td>
      <td>19</td>
      <td>21</td>
      <td>0.0</td>
      <td>4.71</td>
      <td>0.69</td>
      <td>1.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1503960366</td>
      <td>4/14/2016</td>
      <td>1218</td>
      <td>181</td>
      <td>11</td>
      <td>30</td>
      <td>0.0</td>
      <td>3.91</td>
      <td>0.40</td>
      <td>2.44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1503960366</td>
      <td>4/15/2016</td>
      <td>726</td>
      <td>209</td>
      <td>34</td>
      <td>29</td>
      <td>0.0</td>
      <td>2.83</td>
      <td>1.26</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1503960366</td>
      <td>4/16/2016</td>
      <td>773</td>
      <td>221</td>
      <td>10</td>
      <td>36</td>
      <td>0.0</td>
      <td>5.04</td>
      <td>0.41</td>
      <td>2.71</td>
    </tr>
  </tbody>
</table>
</div>



    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    935    False
    936    False
    937    False
    938    False
    939    False
    Length: 940, dtype: bool


    dailySteps_merged_clean
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 940 entries, 0 to 939
    Data columns (total 3 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   Id           940 non-null    int64 
     1   ActivityDay  940 non-null    object
     2   StepTotal    940 non-null    int64 
    dtypes: int64(2), object(1)
    memory usage: 22.2+ KB



    None



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
      <th>Id</th>
      <th>ActivityDay</th>
      <th>StepTotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1503960366</td>
      <td>4/12/2016</td>
      <td>13162</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1503960366</td>
      <td>4/13/2016</td>
      <td>10735</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1503960366</td>
      <td>4/14/2016</td>
      <td>10460</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1503960366</td>
      <td>4/15/2016</td>
      <td>9762</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1503960366</td>
      <td>4/16/2016</td>
      <td>12669</td>
    </tr>
  </tbody>
</table>
</div>



    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    935    False
    936    False
    937    False
    938    False
    939    False
    Length: 940, dtype: bool


    hourlyCalories_merged_clean
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22099 entries, 0 to 22098
    Data columns (total 3 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   Id            22099 non-null  int64 
     1   ActivityHour  22099 non-null  object
     2   Calories      22099 non-null  int64 
    dtypes: int64(2), object(1)
    memory usage: 518.1+ KB



    None



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
      <th>Id</th>
      <th>ActivityHour</th>
      <th>Calories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1503960366</td>
      <td>4/12/2016 12:00:00 AM</td>
      <td>81</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1503960366</td>
      <td>4/12/2016 1:00:00 AM</td>
      <td>61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1503960366</td>
      <td>4/12/2016 2:00:00 AM</td>
      <td>59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1503960366</td>
      <td>4/12/2016 3:00:00 AM</td>
      <td>47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1503960366</td>
      <td>4/12/2016 4:00:00 AM</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>



    0        False
    1        False
    2        False
    3        False
    4        False
             ...  
    22094    False
    22095    False
    22096    False
    22097    False
    22098    False
    Length: 22099, dtype: bool


    hourlyIntensities_merged_clean
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22099 entries, 0 to 22098
    Data columns (total 4 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Id                22099 non-null  int64  
     1   ActivityHour      22099 non-null  object 
     2   TotalIntensity    22099 non-null  int64  
     3   AverageIntensity  22099 non-null  float64
    dtypes: float64(1), int64(2), object(1)
    memory usage: 690.7+ KB



    None



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
      <th>Id</th>
      <th>ActivityHour</th>
      <th>TotalIntensity</th>
      <th>AverageIntensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1503960366</td>
      <td>4/12/2016 12:00:00 AM</td>
      <td>20</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1503960366</td>
      <td>4/12/2016 1:00:00 AM</td>
      <td>8</td>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1503960366</td>
      <td>4/12/2016 2:00:00 AM</td>
      <td>7</td>
      <td>0.116667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1503960366</td>
      <td>4/12/2016 3:00:00 AM</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1503960366</td>
      <td>4/12/2016 4:00:00 AM</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



    0        False
    1        False
    2        False
    3        False
    4        False
             ...  
    22094    False
    22095    False
    22096    False
    22097    False
    22098    False
    Length: 22099, dtype: bool


    hourlySteps_merged_clean
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22099 entries, 0 to 22098
    Data columns (total 3 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   Id            22099 non-null  int64 
     1   ActivityHour  22099 non-null  object
     2   StepTotal     22099 non-null  int64 
    dtypes: int64(2), object(1)
    memory usage: 518.1+ KB



    None



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
      <th>Id</th>
      <th>ActivityHour</th>
      <th>StepTotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1503960366</td>
      <td>4/12/2016 12:00:00 AM</td>
      <td>373</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1503960366</td>
      <td>4/12/2016 1:00:00 AM</td>
      <td>160</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1503960366</td>
      <td>4/12/2016 2:00:00 AM</td>
      <td>151</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1503960366</td>
      <td>4/12/2016 3:00:00 AM</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1503960366</td>
      <td>4/12/2016 4:00:00 AM</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



    0        False
    1        False
    2        False
    3        False
    4        False
             ...  
    22094    False
    22095    False
    22096    False
    22097    False
    22098    False
    Length: 22099, dtype: bool


    sleepDay_merged_clean
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 413 entries, 0 to 412
    Data columns (total 5 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   Id                  413 non-null    int64 
     1   SleepDay            413 non-null    object
     2   TotalSleepRecords   413 non-null    int64 
     3   TotalMinutesAsleep  413 non-null    int64 
     4   TotalTimeInBed      413 non-null    int64 
    dtypes: int64(4), object(1)
    memory usage: 16.3+ KB



    None



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
      <th>Id</th>
      <th>SleepDay</th>
      <th>TotalSleepRecords</th>
      <th>TotalMinutesAsleep</th>
      <th>TotalTimeInBed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1503960366</td>
      <td>4/12/2016 12:00:00 AM</td>
      <td>1</td>
      <td>327</td>
      <td>346</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1503960366</td>
      <td>4/13/2016 12:00:00 AM</td>
      <td>2</td>
      <td>384</td>
      <td>407</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1503960366</td>
      <td>4/15/2016 12:00:00 AM</td>
      <td>1</td>
      <td>412</td>
      <td>442</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1503960366</td>
      <td>4/16/2016 12:00:00 AM</td>
      <td>2</td>
      <td>340</td>
      <td>367</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1503960366</td>
      <td>4/17/2016 12:00:00 AM</td>
      <td>1</td>
      <td>700</td>
      <td>712</td>
    </tr>
  </tbody>
</table>
</div>



    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    408    False
    409    False
    410    False
    411    False
    412    False
    Length: 413, dtype: bool


    weightLogInfo_merged_clean
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 67 entries, 0 to 66
    Data columns (total 8 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Id              67 non-null     int64  
     1   Date            67 non-null     object 
     2   WeightKg        67 non-null     float64
     3   WeightPounds    67 non-null     float64
     4   Fat             2 non-null      float64
     5   BMI             67 non-null     float64
     6   IsManualReport  67 non-null     bool   
     7   LogId           67 non-null     int64  
    dtypes: bool(1), float64(4), int64(2), object(1)
    memory usage: 3.9+ KB



    None



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
      <th>Id</th>
      <th>Date</th>
      <th>WeightKg</th>
      <th>WeightPounds</th>
      <th>Fat</th>
      <th>BMI</th>
      <th>IsManualReport</th>
      <th>LogId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1503960366</td>
      <td>5/2/2016 11:59:59 PM</td>
      <td>52.599998</td>
      <td>115.963147</td>
      <td>22.0</td>
      <td>22.650000</td>
      <td>True</td>
      <td>1462233599000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1503960366</td>
      <td>5/3/2016 11:59:59 PM</td>
      <td>52.599998</td>
      <td>115.963147</td>
      <td>NaN</td>
      <td>22.650000</td>
      <td>True</td>
      <td>1462319999000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1927972279</td>
      <td>4/13/2016 1:08:52 AM</td>
      <td>133.500000</td>
      <td>294.317120</td>
      <td>NaN</td>
      <td>47.540001</td>
      <td>False</td>
      <td>1460509732000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2873212765</td>
      <td>4/21/2016 11:59:59 PM</td>
      <td>56.700001</td>
      <td>125.002104</td>
      <td>NaN</td>
      <td>21.450001</td>
      <td>True</td>
      <td>1461283199000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2873212765</td>
      <td>5/12/2016 11:59:59 PM</td>
      <td>57.299999</td>
      <td>126.324875</td>
      <td>NaN</td>
      <td>21.690001</td>
      <td>True</td>
      <td>1463097599000</td>
    </tr>
  </tbody>
</table>
</div>



    0     False
    1     False
    2     False
    3     False
    4     False
          ...  
    62    False
    63    False
    64    False
    65    False
    66    False
    Length: 67, dtype: bool


I then examined each table individually, checking to see how many distinct users were in each table. I also took care to change dates into the data type <i>datetime</i>. For tables with dates, I added a DayOfWeek column, where 0 represented Monday and 6 represented Sunday. I took this step to be able to analyze data on the level of day of the week. 


```python
# dailyActivity_merged_clean

one_q1 = """
SELECT COUNT(DISTINCT Id)
FROM dailyActivity_merged_clean
"""
ps.sqldf(one_q1)

dailyActivity_merged_clean['ActivityDate'] = dailyActivity_merged_clean['ActivityDate'].astype('datetime64')
dailyActivity_merged_clean['ActivityDayOfWeek'] = dailyActivity_merged_clean['ActivityDate']\
    .apply(datetime.datetime.weekday)

day = dailyActivity_merged_clean['ActivityDayOfWeek']
dailyActivity_merged_clean = dailyActivity_merged_clean.drop(columns=['ActivityDayOfWeek'])
dailyActivity_merged_clean.insert(loc = 2, column = 'ActivityDayOfWeek', value = day)

#dailyActivity_merged_clean.head(10)
```


```python
# dailyCalories_merged_clean
two_q1 = """
SELECT COUNT(DISTINCT Id)
FROM dailyCalories_merged_clean
"""
ps.sqldf(two_q1)

dailyCalories_merged_clean['ActivityDay'] = dailyCalories_merged_clean['ActivityDay'].astype('datetime64')
```


```python
# dailyIntensities_merged_clean
three_q1 = """
SELECT COUNT(DISTINCT Id)
FROM dailyIntensities_merged_clean
"""

ps.sqldf(three_q1)

dailyIntensities_merged_clean['ActivityDay'] = dailyIntensities_merged_clean['ActivityDay'].astype('datetime64')
```


```python
# dailySteps_merged_clean
four_q1 = """
SELECT COUNT(DISTINCT Id)
FROM dailySteps_merged_clean
"""

ps.sqldf(four_q1)

dailySteps_merged_clean['ActivityDay'] = dailySteps_merged_clean['ActivityDay'].astype('datetime64')
```

For hourly data, I separated the date from the hour, allowing me to be able to easier analyze data on an hourly basis. 


```python
# hourlyCalories_merged_clean
five_q1 = """
SELECT COUNT(DISTINCT Id)
FROM hourlyCalories_merged_clean
"""

ps.sqldf(five_q1)

hourlyCalories_merged_clean['ActivityHour'] = hourlyCalories_merged_clean['ActivityHour'].astype('datetime64')

hourlyCalories_merged_clean['ActivityHourTime'] = hourlyCalories_merged_clean['ActivityHour']\
    .apply(datetime.datetime.time)
hourlyCalories_merged_clean['ActivityDate'] = hourlyCalories_merged_clean['ActivityHour']\
    .apply(datetime.datetime.date)

hourCalories = hourlyCalories_merged_clean['ActivityHourTime']
hourlyCalories_merged_clean = hourlyCalories_merged_clean.drop(columns=['ActivityHourTime'])
hourlyCalories_merged_clean.insert(loc = 2, column = 'ActivityHourTime', value = hourCalories)
dateCalories = hourlyCalories_merged_clean['ActivityDate']
hourlyCalories_merged_clean = hourlyCalories_merged_clean.drop(columns=['ActivityDate'])
hourlyCalories_merged_clean.insert(loc = 3, column = 'ActivityDate', value = dateCalories)

#hourlyCalories_merged_clean.head(30)
```


```python
# hourlyIntensities_merged_clean
six_q1 = """
SELECT COUNT(DISTINCT Id)
FROM hourlyIntensities_merged_clean
"""

ps.sqldf(six_q1)

hourlyIntensities_merged_clean['ActivityHour'] = hourlyIntensities_merged_clean['ActivityHour'].astype('datetime64')

hourlyIntensities_merged_clean['ActivityHourTime'] = hourlyIntensities_merged_clean['ActivityHour']\
    .apply(datetime.datetime.time)
hourlyIntensities_merged_clean['ActivityDate'] = hourlyIntensities_merged_clean['ActivityHour']\
    .apply(datetime.datetime.date)
hourlyIntensities_merged_clean['ActivityDate'].astype('datetime64')


hour = hourlyIntensities_merged_clean['ActivityHourTime']
hourlyIntensities_merged_clean = hourlyIntensities_merged_clean.drop(columns=['ActivityHourTime'])
hourlyIntensities_merged_clean.insert(loc = 2, column = 'ActivityHourTime', value = hour)
date = hourlyIntensities_merged_clean['ActivityDate']
hourlyIntensities_merged_clean = hourlyIntensities_merged_clean.drop(columns=['ActivityDate'])
hourlyIntensities_merged_clean.insert(loc = 3, column = 'ActivityDate', value = date)

# hourlyIntensities_merged_clean.head(24)
```


```python
# hourlySteps_merged_clean
seven_q1 = """
SELECT COUNT(DISTINCT Id)
FROM hourlySteps_merged_clean
"""

ps.sqldf(seven_q1)

hourlySteps_merged_clean['ActivityHour'] = hourlySteps_merged_clean['ActivityHour'].astype('datetime64')

hourlySteps_merged_clean['ActivityHourTime'] = hourlySteps_merged_clean['ActivityHour']\
    .apply(datetime.datetime.time)
hourlySteps_merged_clean['ActivityDate'] = hourlySteps_merged_clean['ActivityHour']\
    .apply(datetime.datetime.date)

hourSteps = hourlySteps_merged_clean['ActivityHourTime']
hourlySteps_merged_clean = hourlySteps_merged_clean.drop(columns=['ActivityHourTime'])
hourlySteps_merged_clean.insert(loc = 2, column = 'ActivityHourTime', value = hourSteps)
date = hourlySteps_merged_clean['ActivityDate']
hourlySteps_merged_clean = hourlySteps_merged_clean.drop(columns=['ActivityDate'])
hourlySteps_merged_clean.insert(loc = 3, column = 'ActivityDate', value = date)
#hourlySteps_merged_clean.head(30)
```

Note: The sleepDay_merged table has only 24 users. 


```python
# sleepDay_merged_clean
eight_q1 = """
SELECT COUNT(DISTINCT Id)
FROM sleepDay_merged_clean
"""

ps.sqldf(eight_q1)

sleepDay_merged_clean['SleepDay'] = sleepDay_merged_clean['SleepDay'].astype('datetime64')
sleepDay_merged_clean['SleepDayOfWeek'] = sleepDay_merged_clean['SleepDay']\
    .apply(datetime.datetime.weekday)

sleepDay = sleepDay_merged_clean['SleepDayOfWeek']
sleepDay_merged_clean = sleepDay_merged_clean.drop(columns=['SleepDayOfWeek'])
sleepDay_merged_clean.insert(loc = 2, column = 'SleepDayOfWeek', value = sleepDay)
# sleepDay_merged_clean.head(10)

```

Note: The weightLogInfo_merged table only has 8 users.


```python
# weightLogInfo_merged_clean
nine_q1 = """
SELECT COUNT(DISTINCT Id)
FROM weightLogInfo_merged_clean
"""

ps.sqldf(nine_q1)

weightLogInfo_merged_clean['Date'] = weightLogInfo_merged_clean['Date'].astype('datetime64')
```

# Analyze

As the first steps in analyzing the data, I began by viewing a summary of statistics of the dailyActivity_merged table just to get a sense of what was in the data. 


```python
# summary of statistics of dailyActivity_merged_clean

dailyActivity_merged_clean.describe()
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
      <th>Id</th>
      <th>ActivityDayOfWeek</th>
      <th>TotalSteps</th>
      <th>TotalDistance</th>
      <th>TrackerDistance</th>
      <th>LoggedActivitiesDistance</th>
      <th>VeryActiveDistance</th>
      <th>ModeratelyActiveDistance</th>
      <th>LightActiveDistance</th>
      <th>SedentaryActiveDistance</th>
      <th>VeryActiveMinutes</th>
      <th>FairlyActiveMinutes</th>
      <th>LightlyActiveMinutes</th>
      <th>SedentaryMinutes</th>
      <th>Calories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.400000e+02</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
      <td>940.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.855407e+09</td>
      <td>2.918085</td>
      <td>7637.910638</td>
      <td>5.489702</td>
      <td>5.475351</td>
      <td>0.108171</td>
      <td>1.502681</td>
      <td>0.567543</td>
      <td>3.340819</td>
      <td>0.001606</td>
      <td>21.164894</td>
      <td>13.564894</td>
      <td>192.812766</td>
      <td>991.210638</td>
      <td>2303.609574</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.424805e+09</td>
      <td>1.942379</td>
      <td>5087.150742</td>
      <td>3.924606</td>
      <td>3.907276</td>
      <td>0.619897</td>
      <td>2.658941</td>
      <td>0.883580</td>
      <td>2.040655</td>
      <td>0.007346</td>
      <td>32.844803</td>
      <td>19.987404</td>
      <td>109.174700</td>
      <td>301.267437</td>
      <td>718.166862</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.503960e+09</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.320127e+09</td>
      <td>1.000000</td>
      <td>3789.750000</td>
      <td>2.620000</td>
      <td>2.620000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.945000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>127.000000</td>
      <td>729.750000</td>
      <td>1828.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.445115e+09</td>
      <td>3.000000</td>
      <td>7405.500000</td>
      <td>5.245000</td>
      <td>5.245000</td>
      <td>0.000000</td>
      <td>0.210000</td>
      <td>0.240000</td>
      <td>3.365000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>199.000000</td>
      <td>1057.500000</td>
      <td>2134.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.962181e+09</td>
      <td>5.000000</td>
      <td>10727.000000</td>
      <td>7.712500</td>
      <td>7.710000</td>
      <td>0.000000</td>
      <td>2.052500</td>
      <td>0.800000</td>
      <td>4.782500</td>
      <td>0.000000</td>
      <td>32.000000</td>
      <td>19.000000</td>
      <td>264.000000</td>
      <td>1229.500000</td>
      <td>2793.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.877689e+09</td>
      <td>6.000000</td>
      <td>36019.000000</td>
      <td>28.030001</td>
      <td>28.030001</td>
      <td>4.942142</td>
      <td>21.920000</td>
      <td>6.480000</td>
      <td>10.710000</td>
      <td>0.110000</td>
      <td>210.000000</td>
      <td>143.000000</td>
      <td>518.000000</td>
      <td>1440.000000</td>
      <td>4900.000000</td>
    </tr>
  </tbody>
</table>
</div>



Then, I wanted to analyze what were the most active days of the week.


```python
# Average steps, calories, active minutes by day of the week

q1 = """
SELECT ActivityDayOfWeek, 
    AVG(TotalSteps), 
    AVG(Calories), 
    AVG(VeryActiveMinutes) + AVG(FairlyActiveMinutes) + AVG(LightlyActiveMinutes) AS AverageTotalActiveMinutes
FROM dailyActivity_merged_clean
GROUP BY ActivityDayOfWeek
"""
average_per_day = ps.sqldf(q1)
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# average steps plot
avg_steps_plot = sns.barplot(data = average_per_day, x = 'ActivityDayOfWeek', y = 'AVG(TotalSteps)')
plt.xticks([0, 1, 2, 3, 4, 5, 6], days_of_week, rotation = 45)
avg_steps_plot.set_xlabel(xlabel = '')
avg_steps_plot.set_ylabel(ylabel = 'Steps')
plt.title(label = 'Average Total Steps per Day')
plt.show()

# average activity plot
avg_activity_plot = sns.barplot(data = average_per_day, x = 'ActivityDayOfWeek', y = 'AverageTotalActiveMinutes')
plt.xticks([0, 1, 2, 3, 4, 5, 6], days_of_week, rotation = 45)
avg_activity_plot.set_xlabel(xlabel = '')
avg_activity_plot.set_ylabel(ylabel = 'Active Minutes')
plt.title(label = 'Average Active Minutes per Day')
plt.show()
```


    
![png](output_25_0.png)
    



    
![png](output_25_1.png)
    


Then I looked at what were the most active hours of the day. On both measures, steps and activity, the most active hours of the day are from 5:00 - 7:00 PM. 
<br><br>
<i>A note on the total active minutes: this metric is pulled from the hourlyIntensities_merged table, where we are given <b>TotalIntensity</b>. This is measured as a weighted sum of the number of active minutes per hour, where eaach type of activity (light, fair, very) gets weighted and added together to create TotalIntensity.</i>


```python
# Average average intensity by hour -- what % of the hour they were active

q2 = """
SELECT AVG(TotalIntensity) AS avg_total_intensity, 
    ActivityHourTime AS time
FROM hourlyIntensities_merged_clean
GROUP BY ActivityHourTime
"""

avg_activity_per_hour = ps.sqldf(q2)

avg_activity_per_hour_plot = sns.barplot(data = avg_activity_per_hour, x = 'time', y = 'avg_total_intensity')
avg_activity_per_hour_plot.set_xlabel(xlabel = 'Time')
avg_activity_per_hour_plot.set_ylabel(ylabel = 'Active Minutes')
plt.xticks(rotation = 90)
plt.title(label = 'Average Active Minutes per Hour')

```




    Text(0.5, 1.0, 'Average Active Minutes per Hour')




    
![png](output_27_1.png)
    



```python
# Average steps by hour

q3 = """
SELECT AVG(StepTotal) AS avg_steps, 
    ActivityHourTime AS time
FROM hourlySteps_merged_clean
GROUP BY ActivityHourTime
"""

avg_steps_per_hour = ps.sqldf(q3)

avg_steps_per_hour_plot = sns.barplot(data = avg_steps_per_hour, x = 'time', y = 'avg_steps')
avg_steps_per_hour_plot.set_xlabel(xlabel = 'Time')
avg_steps_per_hour_plot.set_ylabel(ylabel = 'Steps')
plt.title(label = 'Average Steps per Hour')
plt.xticks(rotation = 90)
plt.show()
```


    
![png](output_28_0.png)
    


Next, I looked at the amount of sleep users recorded, discovering that users get the most sleep on Sunday. 


```python
# Average amount of time asleep per day 

q4 = """
SELECT AVG(TotalMinutesAsleep), AVG(TotalTimeInBed), SleepDayOfWeek
FROM sleepDay_merged_clean
GROUP BY SleepDayOfWeek
"""

avg_sleep_per_day = ps.sqldf(q4)

avg_sleep_per_day_plot = sns.barplot(data=avg_sleep_per_day, x='SleepDayOfWeek', y='AVG(TotalMinutesAsleep)')
avg_sleep_per_day_plot.set_xlabel(xlabel='')
avg_sleep_per_day_plot.set_ylabel(ylabel='Minutes Asleep')
plt.xticks([0, 1, 2, 3, 4, 5, 6], days_of_week, rotation = 45)
plt.title(label='Average Minutes Asleep per Day')
plt.show()
```


    
![png](output_30_0.png)
    


Next, I wanted to look at the differences between users who track their sleep and those who don't- hopefully to better understand how we can encourage users to utilize every function of our device. <br><br>

I first looked at the average steps per day and average active minutes per day for each user and gave them an identifier of whether they tracked their sleep or not. I also looked at activity on the group level (group 0 being those who do not track their sleep, group 1 being those who track their sleep). 


```python
# Average activity per user and whether they track sleep on the user level

q6 = """
    WITH steps AS (
        SELECT Id AS steps_id, 
            AVG(TotalSteps) AS average_steps_per_day,
            AVG(VeryActiveMinutes) + AVG(FairlyActiveMinutes) + AVG(LightlyActiveMinutes) AS average_active_minutes_per_day
        FROM dailyActivity_merged_clean
        GROUP BY steps_id
    ),

    sleep_users AS (
        SELECT Id AS sleep_id
        FROM sleepDay_merged_clean
        GROUP BY sleep_id
    )

    SELECT *,
        CASE WHEN sleep_users.sleep_id IS NOT NULL 
            THEN 1 
            ELSE 0 
            END AS is_sleep_user
    FROM steps 
    LEFT JOIN sleep_users
        ON steps.steps_id = sleep_users.sleep_id
    ORDER BY is_sleep_user DESC
    
"""

average_activity_is_sleep_user = ps.sqldf(q6)
# average_activity_is_sleep_user
```


```python
# average activity between sleep and non sleep trackers on group level

average_activity_is_sleep_user.groupby(['is_sleep_user']).mean()[['average_steps_per_day', 'average_active_minutes_per_day']]

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
      <th>average_steps_per_day</th>
      <th>average_active_minutes_per_day</th>
    </tr>
    <tr>
      <th>is_sleep_user</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7860.025621</td>
      <td>235.810904</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7391.490324</td>
      <td>221.070073</td>
    </tr>
  </tbody>
</table>
</div>



I then looked at the difference in inactivity between the groups. To measure inactivity, I looked at the 50th percentile, which is 3.00 TotalIntensity for the hour. I also wanted to exclude time asleep, so I counted an inactive hour as one that occurred after 06:00 AM and had a TotalIntensity score of less than or equal to 3.00. <br> <br>

<b>Average Inactive Hours per day, Non-sleep trackers: 5.014352529451177</b>
<br>
<b>Average Inactive Hours per day, Sleep trackers: 6.221028268099687</b>


```python
# average inactive hours per day, separated by sleep user category
# inactive hours are daytime hours (after 06:00) where user has <= 3.00 in TotalIntensity (50th percentile)

q23 = """
    WITH intensity AS (
        SELECT d.Id AS intensity_id,
            CAST(COUNT(DISTINCT(d.ActivityDay)) AS float64) AS days_recorded,
            COALESCE(
                (SUM(CASE WHEN CAST(h.TotalIntensity AS float64) <= 3.00 
                THEN 1
                ELSE 0
                END) / CAST(COUNT(DISTINCT(d.ActivityDay)) AS float64)), 0)  
                AS avg_hours_per_day_inactive
        FROM dailyIntensities_merged_clean AS d
        LEFT JOIN hourlyIntensities_merged_clean AS h
            ON d.Id = h.Id
            AND DATE(d.ActivityDay) = DATE(h.ActivityDate)
        WHERE (CAST(h.ActivityHourTime AS float64) > 6.0 OR h.ActivityHourTime IS NULL)
        GROUP BY intensity_id
    ),
    
    sleep_users AS (
        SELECT Id AS sleep_id
        FROM sleepDay_merged_clean
        GROUP BY sleep_id
    ),
    
    is_sleep_user AS (
        SELECT *,
            CASE WHEN sleep_users.sleep_id IS NOT NULL 
                THEN 1 
                ELSE 0 
                END AS is_sleep_user
        FROM intensity
        LEFT JOIN sleep_users
            ON intensity.intensity_id = sleep_users.sleep_id
    )
    
    SELECT is_sleep_user.intensity_id AS id,
        avg_hours_per_day_inactive,
        is_sleep_user.is_sleep_user
    FROM is_sleep_user
    GROUP BY id,
        is_sleep_user.is_sleep_user
    

"""

avg_hours_perday_inactive_df = ps.sqldf(q23)
# display(avg_hours_perday_inactive_df)


# average inactive hours per day and std 
average_inactive_hours_non_trackers = avg_hours_perday_inactive_df[(avg_hours_perday_inactive_df['is_sleep_user'] == 0)].mean()
std_inactive_hours_non_trackers = avg_hours_perday_inactive_df[(avg_hours_perday_inactive_df['is_sleep_user'] == 0)].std()


# display(average_inactive_hours_non_trackers)
# display(std_inactive_hours_non_trackers)

average_inactive_hours_trackers = avg_hours_perday_inactive_df[(avg_hours_perday_inactive_df['is_sleep_user'] == 1)].mean()
std_inactive_hours_trackers = avg_hours_perday_inactive_df[(avg_hours_perday_inactive_df['is_sleep_user'] == 1)].std()

# display(average_inactive_hours_trackers)
# display(std_inactive_hours_trackers)


print('Average Inactive Hours per day, Non-sleep trackers:', average_inactive_hours_non_trackers['avg_hours_per_day_inactive'])
print('Standard Deviation Inactive Hours per day, Non-sleep trackers:', std_inactive_hours_non_trackers['avg_hours_per_day_inactive'])

print('Average Inactive Hours per day, Sleep trackers:', average_inactive_hours_trackers['avg_hours_per_day_inactive'])
print('Standard Deviation Inactive Hours per day, Sleep trackers:', std_inactive_hours_trackers['avg_hours_per_day_inactive'])


```

    Average Inactive Hours per day, Non-sleep trackers: 5.014352529451177
    Standard Deviation Inactive Hours per day, Non-sleep trackers: 2.260731477458942
    Average Inactive Hours per day, Sleep trackers: 6.221028268099687
    Standard Deviation Inactive Hours per day, Sleep trackers: 3.464922019201963


Lastly, I wanted to analyze the impact of sleep quality on health, as measured here by calories burned. I defined sleep quality as the percentage of time in bed that the user is asleep. The higher the percentage means that the user spent more time asleep and less time falling asleep or getting out of bed in the morning.


```python
# relationship between sleep quality and calories

q27 = """
    SELECT c.Id AS id,
        c.ActivityDay AS date,
        Calories AS calories,
    (CAST (s.TotalMinutesAsleep AS float64) / s.TotalTimeInBed) AS sleep_quality
    FROM dailyCalories_merged_clean AS c
    JOIN sleepDay_merged_clean AS s
        ON c.Id = s.Id 
        AND c.ActivityDay = s.SleepDay 
    GROUP BY c.Id,
        c.ActivityDay
    
"""
sleep_vs_calories_df = ps.sqldf(q27)
sleep_vs_calories_plot = sns.regplot(
    data=sleep_vs_calories_df,
    x='sleep_quality',
    y='calories')

# correlation between x and y
x_vars = np.array(sleep_vs_calories_df['sleep_quality'])
y_vars = np.array(sleep_vs_calories_df['calories'])

pearson_r = np.corrcoef(x_vars, y_vars)
print(pearson_r)
# correlation coefficient is 0.29 -- slightly positively correlated

x = pd.Series(sleep_vs_calories_df['sleep_quality'])
y = pd.Series(sleep_vs_calories_df['calories'])
pearson = x.corr(y)
spearman = x.corr(y, method='spearman')
kendall = x.corr(y, method='kendall')

print(f'Pearson: {pearson}')
print(f'Spearman: {spearman}')
print(f'Kendall: {kendall}')



```

    [[1.         0.29486181]
     [0.29486181 1.        ]]
    Pearson: 0.29486181130488903
    Spearman: 0.23243976238899228
    Kendall: 0.15398289607122934



    
![png](output_37_1.png)
    



```python
# linear regression in scipy sleep_quality and calories
result = scipy.stats.linregress(x, y)
slope = result.slope
intercept = result.intercept
r_value = result.rvalue
p_value = result.pvalue
std_err = result.stderr

print(f'Slope: {slope}')
print(f'Intecept: {intercept}')
print(f'R value(Correlation Coefficient): {r_value}')
print(f'P value: {p_value}')
print(f'Standard Error: {std_err}')

if p_value < 0.001:
    print('p < 0.001, results are considered statistically significant')
elif p_value < 0.05:
        print('p < 0.05, results are considered statistically significant')
else:
    print('p > 0.05, results are not considered statistically significant')
```

    Slope: 2562.140555753254
    Intecept: 41.17860136004947
    R value(Correlation Coefficient): 0.29486181130488903
    P value: 1.1422373205471258e-09
    Standard Error: 411.0580611919415
    p < 0.001, results are considered statistically significant


# Share

I noticed that not all users tracked their sleep. I wanted to further examine this to get a better understanding of why that may be happening and how Bellabeat can use that information for their own product marketing. 

<br>
My hypothesis was that not all users are tracking their sleep because they are charging their devices overnight. To look into this theory, I had to explore the difference during the day to see if those who did track their sleep were potentially charging their devices during the day instead. 

<br> 
To support this theory, I found that:

* on average, those who track their sleep were <i> less </i> active during the day than those who did not track their sleep. 
    * Those who did track their sleep got an average of 7391 steps per day, whereas those who did not track their sleep got an average of 7860 steps per day. 
    * Those who did track their sleep had an average of 221 active minutes per day, compared to the 236 active minutes per day of the other group.	

* on average, those who track their sleep had <i> more </i> inactive hours during the day than those who did not track their sleep. 
    * Those who track their sleep had an average of 6 inactive hours per day, whereas the other group had an average of 5 inactive hours per day. 

This supports the idea that users who track their sleep are charging their devices during the day, which is what accounts for how they are less active than the group who does not track their sleep. 

<br>
<b> Why does this matter? </b> <br>
The data shows that there is a slight positive correlation between quality of sleep and calories burned. In other words, the better your quality of sleep, the more calories you burn. This information would be helpful to our users in tracking their calories and other metrics of health. <br>

<b>Some marketing strategies:</b>
1. Advertise the longer battery life of the Bellabeat Leaf and its impact on sleep tracking. 
    * One reason users of other fitness devices may not track their sleep is because they are charging their device overnight. Because of this, they are missing out on valuable data about their sleep which can impact other health and wellness determinants. With the Leaf's 6-month battery life, users can track their sleep all night every night, no charging required! 

2. Push notifications during peak activity periods. 
    * We know that the most active day of the week is Saturday and the most active time of day is between 5:00-7:00PM. By sending push notifications during these times, we can increase app engagement and increase engagement with Bellabeat membership programs.  
