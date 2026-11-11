"""
Objectives
After completing this lab you will be able to:

Explore and perform feature engineering on a real-world data set
Build a classifier pipeline and optimize it using grid search cross validation
Evaluate your model by interpreting various performance metrics and visualizations
Implement a different classifier by updating your pipeline
Use an appropriate set of parameters to search over in each case
"""

!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install seaborn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


# Load the data
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

df.count()

"""
Date             145460
Location         145460
MinTemp          143975
MaxTemp          144199
Rainfall         142199
Evaporation       82670
Sunshine          75625
WindGustDir      135134
WindGustSpeed    135197
WindDir9am       134894
WindDir3pm       141232
WindSpeed9am     143693
WindSpeed3pm     142398
Humidity9am      142806
Humidity3pm      140953
Pressure9am      130395
Pressure3pm      130432
Cloud9am          89572
Cloud3pm          86102
Temp9am          143693
Temp3pm          141851
RainToday        142199
RainTomorrow     142193
dtype: int64
Sunshine and cloud cover seem like important features, but they have a lot of missing values, far too many to impute their missing values.
"""

"""
Drop all rows with missing values¶
To try to keep things simple we'll drop rows with missing values and see what's left
"""

df = df.dropna()
df.info()

"""
<class 'pandas.DataFrame'>
Index: 56420 entries, 6049 to 142302
Data columns (total 23 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Date           56420 non-null  str    
 1   Location       56420 non-null  str    
 2   MinTemp        56420 non-null  float64
 3   MaxTemp        56420 non-null  float64
 4   Rainfall       56420 non-null  float64
 5   Evaporation    56420 non-null  float64
 6   Sunshine       56420 non-null  float64
 7   WindGustDir    56420 non-null  str    
 8   WindGustSpeed  56420 non-null  float64
 9   WindDir9am     56420 non-null  str    
 10  WindDir3pm     56420 non-null  str    
 11  WindSpeed9am   56420 non-null  float64
 12  WindSpeed3pm   56420 non-null  float64
 13  Humidity9am    56420 non-null  float64
 14  Humidity3pm    56420 non-null  float64
 15  Pressure9am    56420 non-null  float64
 16  Pressure3pm    56420 non-null  float64
 17  Cloud9am       56420 non-null  float64
 18  Cloud3pm       56420 non-null  float64
 19  Temp9am        56420 non-null  float64
 20  Temp3pm        56420 non-null  float64
 21  RainToday      56420 non-null  str    
 22  RainTomorrow   56420 non-null  str    
dtypes: float64(16), str(7)
memory usage: 12.0 MB
Since we still have 56k observations left after dropping missing values, we may not need to impute any missing values.
"""

df.columns

"""
Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RainTomorrow'],
      dtype='str')
"""

"""
Data leakage considerations
Consider the descriptions above for the columns in the data set. Are there any practical limitations to being able to predict whether it will rain tomorrow given the available data?

Points to note - 1
List some of the features that would be inefficient in predicting tomorrow's rainfall.
Hint: Consider features that rely on the entire duration of today for their evaluation.

If we adjust our approach and aim to predict today’s rainfall using historical weather data up to and including yesterday, then we can legitimately utilize all of the available features. This shift would be particularly useful for practical applications, such as deciding whether you will bike to work today.

With this new target, we should update the names of the rain columns accordingly to avoid confusion.
"""
df = df.rename(
  columns = { 'RainToday' : 'RainYesterday', 'RainTomorrow' : 'RainToday' }
)

"""
Data Granularity
Would the weather patterns have the same predictability in vastly different locations in Australia? I would think not.
The chance of rain in one location can be much higher than in another. Using all of the locations requires a more complex model as it needs to adapt to local weather patterns.
Let's see how many observations we have for each location, and see if we can reduce our attention to a smaller region.

Location selection
You could do some research to group cities in the Location column by distance, which I've done for you behind the scenes.
I found that Watsonia is only 15 km from Melbourne, and the Melbourne Airport is only 18 km from Melbourne.
Let's group these three locations together and use only their weather data to build our localized prediction model.
Because there might still be some slight variations in the weather patterns we'll keep Location as a categorical variable.
"""

df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()

"""
<class 'pandas.DataFrame'>
Index: 7557 entries, 64191 to 80997
Data columns (total 23 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Date           7557 non-null   str    
 1   Location       7557 non-null   str    
 2   MinTemp        7557 non-null   float64
 3   MaxTemp        7557 non-null   float64
 4   Rainfall       7557 non-null   float64
 5   Evaporation    7557 non-null   float64
 6   Sunshine       7557 non-null   float64
 7   WindGustDir    7557 non-null   str    
 8   WindGustSpeed  7557 non-null   float64
 9   WindDir9am     7557 non-null   str    
 10  WindDir3pm     7557 non-null   str    
 11  WindSpeed9am   7557 non-null   float64
 12  WindSpeed3pm   7557 non-null   float64
 13  Humidity9am    7557 non-null   float64
 14  Humidity3pm    7557 non-null   float64
 15  Pressure9am    7557 non-null   float64
 16  Pressure3pm    7557 non-null   float64
 17  Cloud9am       7557 non-null   float64
 18  Cloud3pm       7557 non-null   float64
 19  Temp9am        7557 non-null   float64
 20  Temp3pm        7557 non-null   float64
 21  RainYesterday  7557 non-null   str    
 22  RainToday      7557 non-null   str    
dtypes: float64(16), str(7)
memory usage: 1.6 MB

We still have 7557 records, which should be enough to build a reasonably good model.
You could always gather more data if needed by partioning the data into similar locations or simplyby updating it from the source to include a larger time frame.

"""

"""
Extracting a seasonality feature
Now consider the Date column. We expect the weather patterns to be seasonal, having different predictablitiy levels in winter and summer for example.
There may be some variation with Year as well, but we'll leave that out for now. Let's engineer a Season feature from Date and drop Date afterward, since
it is most likely less informative than season. An easy way to do this is to define a function that assigns seasons to given months, then use that function to
transform the Date column.
"""
# Create a function to map dates to seasons

def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

# Exercise 1: Map the dates to seasons and drop the Date column
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

df=df.drop(columns=['Date'])
df

"""
Looks like we have a good set of features to work with.

Let's go ahead and build our model.

But wait, let's take a look at how well balanced our target is.
"""

"""
Exercise 2. Define the feature and target dataframes
"""
# Write your response.
y = df['RainToday']
X = df.drop([ 'RainToday', 'Evaporation', 'Sunshine', 'Rainfall', 'MaxTemp', 'WindGustSpeed', 'WindGustDir', 'Temp3pm', 'WindDir3pm', 'WindSpeed3pm', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm' ], axis=1)


"""
Exercise 3. How balanced are the classes?
Display the counts of each class.

Complete the following code:
"""
y.value_counts()
"""
RainToday
No     5766
Yes    1791
Name: count, dtype: int64
"""
