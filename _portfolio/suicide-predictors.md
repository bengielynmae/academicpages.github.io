---
title: "Machine Learning Identifies Top Predictors of Suicide Attempts"
excerpt: "Using demographic data from a survey in reddit, the top predictors of suicide attempts are identified. Machine learning tries to predict a result of complex social, psychological, and biological interactions.<br/><br><img src='/images/suicide-predictors/cover.png' width='800' height='600'>"
collection: portfolio
---

<h2>Overview</h2>
<p>This was a final project output for our <b>Machine Learning</b> course under Prof. Chris Monterola in the M.Sc. Data Science program. This study confirms that machine learning can be used on mental health data (if made available). It is capable of identifying predictors of suicide attempts and other critical suicide risk such as self-harm. This was presented to class in August 2019.</p>

<img src='/images/suicide-predictors/poster.png' width='800' height='1900'><br><br>


# Identifying Suicide Attempt Predictors using Machine Learning
### An extended study
Prepared by Bengielyn Danao

## Executive Summary
Suicide and suicide attempts are very challenging to predict because it is an end result of complex social, psychological, and biological interactions. Statistically speaking, it is also rare in terms of reported instances. Machine learning is now put forward as a tool that could improve the accuracy of predicting suicide and its predictors. An existing study from the Department of Psychiatry, University of Texas Health Science Center used information from 144 subjects and got an accuracy of 72%. Similarly, this study has used demographic data from a survey of **496 respondents**. Using the **Random Forest Classifier**, an accuracy of 81% has been obtained and the model identified **depression** as the highest predictor of suicide attempts. 

## Data Source

The demographic data was collected from a survey of subscribers of the subreddit `/r/ForeverAlone`. The survey questions served as the variables and are listed below:  

* Gender (male of female)
* Sexual orientation (bisexual, lesbian, gay)
* Age
* Income level
* Race / ethnicity
* Bodyweight description (underweight, normal, overweight, obese)
* Virginity (yes / no)
* Legality of prostitution in the area (yes / no)
* No. of friends in real life
* Social fear / anxiety (yes / no)
* Are you depressed? (yes / no)
* What kind of help do you want from others? (string)
* Have you attempted suicide? (yes / no)
* Employment status (undemployed, employed, student)
* Job title (string)
* Education level
* What have you done to improve yourself? (string) 

There were a total of `496` user participants of this survey and the dataset is available in `Kaggle` or in this [link](https://lionbridge.ai/datasets/12-free-demographic-datasets-for-machine-learning/).


## Sample data

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
      <th>time</th>
      <th>gender</th>
      <th>sexuallity</th>
      <th>age</th>
      <th>income</th>
      <th>race</th>
      <th>bodyweight</th>
      <th>virgin</th>
      <th>prostitution_legal</th>
      <th>pay_for_sex</th>
      <th>friends</th>
      <th>social_fear</th>
      <th>depressed</th>
      <th>what_help_from_others</th>
      <th>attempt_suicide</th>
      <th>employment</th>
      <th>job_title</th>
      <th>edu_level</th>
      <th>improve_yourself_how</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5/17/2016 20:04:18</td>
      <td>Male</td>
      <td>Straight</td>
      <td>35</td>
      <td>$30,000 to $39,999</td>
      <td>White non-Hispanic</td>
      <td>Normal weight</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>wingman/wingwoman, Set me up with a date</td>
      <td>Yes</td>
      <td>Employed for wages</td>
      <td>mechanical drafter</td>
      <td>Associate degree</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5/17/2016 20:04:30</td>
      <td>Male</td>
      <td>Bisexual</td>
      <td>21</td>
      <td>$1 to $10,000</td>
      <td>White non-Hispanic</td>
      <td>Underweight</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>wingman/wingwoman, Set me up with a date, date...</td>
      <td>No</td>
      <td>Out of work and looking for work</td>
      <td>-</td>
      <td>Some college, no degree</td>
      <td>join clubs/socual clubs/meet ups</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5/17/2016 20:04:58</td>
      <td>Male</td>
      <td>Straight</td>
      <td>22</td>
      <td>$0</td>
      <td>White non-Hispanic</td>
      <td>Overweight</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>10.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>I don't want help</td>
      <td>No</td>
      <td>Out of work but not currently looking for work</td>
      <td>unemployed</td>
      <td>Some college, no degree</td>
      <td>Other exercise</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5/17/2016 20:08:01</td>
      <td>Male</td>
      <td>Straight</td>
      <td>19</td>
      <td>$1 to $10,000</td>
      <td>White non-Hispanic</td>
      <td>Overweight</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>8.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>date coaching</td>
      <td>No</td>
      <td>A student</td>
      <td>student</td>
      <td>Some college, no degree</td>
      <td>Joined a gym/go to the gym</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5/17/2016 20:08:04</td>
      <td>Male</td>
      <td>Straight</td>
      <td>23</td>
      <td>$30,000 to $39,999</td>
      <td>White non-Hispanic</td>
      <td>Overweight</td>
      <td>No</td>
      <td>No</td>
      <td>Yes and I have</td>
      <td>10.0</td>
      <td>No</td>
      <td>Yes</td>
      <td>I don't want help</td>
      <td>No</td>
      <td>Employed for wages</td>
      <td>Factory worker</td>
      <td>High school graduate, diploma or the equivalen...</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## Data Cleaning

Columns that were not deemed determinate of the target variable (suicide attempt) were dropped and column names were renamed as needed. 


```python
drop_col = ['time', 'prostitution_legal', 'pay_for_sex', 'job_title']
clean_dum = dummy.drop(drop_col, axis=1)
```


```python
clean_dum = clean_dum.rename(columns={'sexuallity':'sexuality', 'friends':'# of friends', 'what_help_from_others':'want_help', 'edu_level':'education', 'improve_yourself_how':'willing_to_improve'})
```

The variable `"what help from others"` was also changed to binary. The phrase `"I don't want help"` was filtered and was categorized as a negative outcome (non-help) and all the rest be positive outcomes. The same was done for the column `"improve yourself how"` using the keyword `"None"` to identify negative outcomes and all the rest as positive outcomes. 


```python
clean_dum['willing_to_improve'] = clean_dum['willing_to_improve'].str.lower()
```


```python
clean_dum.loc[clean_dum['willing_to_improve'] == 'none', 'willing_to_improve'] = 0
clean_dum.loc[clean_dum['willing_to_improve'] != 0, 'willing_to_improve'] = 1
```


```python
clean_dum['want_help'] = clean_dum['want_help'].str.lower()
```


```python
no_list = ["i don't want help", 'im on my own', 
           'there is no way that they can help. they only give useless advice like "just be more confident".',
           "i don't want any help. i can't even talk about it.", 
           "i don't want help, like, to be seen & treated like a normal person",
           'i lost faith and hope', "i don't want help, kill me", "i don't want help, more general stuff"]

for item in no_list:
    clean_dum.loc[clean_dum['want_help'] == item, 'want_help'] = 0
```


```python
clean_dum.loc[clean_dum['want_help'] != 0, 'want_help'] = 1
```

The **ordinal categorical variables** (education, bodyweight, and income level) were mapped accordingly while the **nominal categorical variables** were also regrouped as shown below then one-hot encoded as necessary. 


```python
clean_dum['education'] = clean_dum['education'].str.lower()
clean_dum['gender'] = clean_dum['gender'].str.lower()
clean_dum['sexuality'] = clean_dum['sexuality'].str.lower()
clean_dum['race'] = clean_dum['race'].str.lower()
clean_dum['bodyweight'] = clean_dum['bodyweight'].str.lower()
clean_dum['employment'] = clean_dum['employment'].str.lower()
```


```python
hs = ['high school graduate, diploma or the equivalent (for example: ged)', 'trade/technical/vocational training',
      'some high school, no diploma']
college = ['associate degree', 'some college, no degree', 'bachelor’s degree', ]
post_grad = ["master’s degree", 'doctorate degree', 'professional degree']

for deg in hs:
    clean_dum.loc[clean_dum['education'] == deg, 'education'] = 1
    
for deg in college:
    clean_dum.loc[clean_dum['education'] == deg, 'education'] = 2
    
for deg in post_grad:
    clean_dum.loc[clean_dum['education'] == deg, 'education'] = 3
```


```python
unemployed = ['out of work and looking for work', 'out of work but not currently looking for work',
              'unable to work', 'retired', 'a homemaker']
employed = ['employed for wages', 'military', 'self-employed']

for x in unemployed:
    clean_dum.loc[clean_dum['employment'] == x, 'employment'] = 'unemployed'
    
for x in employed:
    clean_dum.loc[clean_dum['employment'] == x, 'employment'] = 'employed'
    
clean_dum.loc[clean_dum['employment'] == 'a student', 'employment'] = 'student'
```


```python
# hispanic = ['hispanic (of any race)']
european = ['caucasian', 'turkish', 'european']
# asian = ['asian']
# indian = ['indian']
african = ['black', 'north african']
mixed = ['mixed race', 'helicopterkin', 'mixed', 'multi', 'white non-hispanic', 
         'white and asian', 'mixed white/asian', 'half asian half white', 
         'first two answers. gender is androgyne, not male; sexuality is asexual, not bi.']
american = ['native american', 'native american mix', 'white and native american']
middle_eastern = ['middle eastern', 'half arab', 'pakistani']
```


```python
for val in european:
    clean_dum.loc[clean_dum['race'] == val, 'race'] = 'european'
    
for val in african:
    clean_dum.loc[clean_dum['race'] == val, 'race'] = 'african'

for val in mixed:
    clean_dum.loc[clean_dum['race'] == val, 'race'] = 'mixed race'

for val in american:
    clean_dum.loc[clean_dum['race'] == val, 'race'] = 'american'

for val in middle_eastern:
    clean_dum.loc[clean_dum['race'] == val, 'race'] = 'middle eastern'
    
clean_dum.loc[clean_dum['race'] == 'hispanic (of any race)', 'race'] = 'hispanic'
clean_dum.loc[clean_dum['race'] == 'asian', 'race'] = 'asian'
clean_dum.loc[clean_dum['race'] == 'indian', 'race'] = 'indian'
```


```python
low = ['$1 to $10,000', '$0', '$30,000 to $39,999', '$20,000 to $29,999', '$10,000 to $19,999']
mid = ['$50,000 to $74,999', '$75,000 to $99,999', '$40,000 to $49,999']
high = ['$150,000 to $174,999', '$125,000 to $149,999', '$100,000 to $124,999', '$174,999 to $199,999', 
        '$200,000 or more']
```


```python
for num in low:
    clean_dum.loc[clean_dum['income'] == num, 'income'] = 'low'

for num in mid:
    clean_dum.loc[clean_dum['income'] == num, 'income'] = 'mid'
    
for num in high:
    clean_dum.loc[clean_dum['income'] == num, 'income'] = 'high'
```


```python
bodyweight = {
'underweight': 1,
'normal weight': 2,
'overweight': 3,
'obese':4}
clean_dum['bodyweight'] = clean_dum['bodyweight'].map(bodyweight)
```


```python
income_map = {
'low': 1,
'mid': 2,
'high': 3}

clean_dum['income'] = clean_dum['income'].map(income_map)
```


```python
virgin_map = {
'Yes': 1,
'No': 0}

clean_dum['virgin'] = clean_dum['virgin'].map(virgin_map)
```


```python
social_fear = {
'Yes': 1,
'No': 0}

clean_dum['social_fear'] = clean_dum['social_fear'].map(social_fear)
```


```python
suicide_attempt = {
'Yes': 1,
'No': 0}

clean_dum['attempt_suicide'] = clean_dum['attempt_suicide'].map(suicide_attempt)
```


```python
depressed_map = {
'Yes': 1,
'No': 0}

clean_dum['depressed'] = clean_dum['depressed'].map(depressed_map)
```


```python
new_dummy = pd.get_dummies(clean_dum)
```

```python
new_dummy.head()
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
      <th>age</th>
      <th>income</th>
      <th>bodyweight</th>
      <th>virgin</th>
      <th># of friends</th>
      <th>social_fear</th>
      <th>depressed</th>
      <th>want_help</th>
      <th>attempt_suicide</th>
      <th>education</th>
      <th>...</th>
      <th>race_american</th>
      <th>race_asian</th>
      <th>race_european</th>
      <th>race_hispanic</th>
      <th>race_indian</th>
      <th>race_middle eastern</th>
      <th>race_mixed race</th>
      <th>employment_employed</th>
      <th>employment_student</th>
      <th>employment_unemployed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>10.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>8.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>10.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



The `new_dummy` is now the final, cleaned dataframe that will be used in the model. But for now, let us look more into our data. 

## Exploratory Data Analysis


```python
a_ = df_drop.groupby(['gender','sexuality'])['age'].count().reset_index().pivot(columns='sexuality', index='gender')
```


```python
a_.fillna(0)
```

```python
fig, ax = plt.subplots(1,2, figsize=(16,7))
total = df_drop.shape[0]
sns.countplot(x='gender', data=df_drop, ax=ax[0])

ax[0].set_title('Gender Distribution')
ax[0].set_axisbelow(True)
ax[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[0].set_ylabel('count')

# percent labels
for p in ax[0].patches:
    height = p.get_height()
    ax[0].text(p.get_x()+p.get_width()/2,
               height + 2,
               f'{height/total:.0%}',
               ha="center",
               weight='semibold',
               fontsize=16) 

total = df_drop.shape[0]
sns.countplot(x='sexuality', data=df_drop, ax=ax[1])

ax[1].set_title('Sexuality Distribution')
ax[1].set_axisbelow(True)
ax[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[1].set_ylabel('count')

plt.tight_layout()
```


![png](/images/suicide-predictors/gender-sex.png)



```python
male_data = df_drop[df_drop.gender=='Male']
female_data = df_drop[df_drop.gender=='Female']

plt.figure(figsize=(10,8))
# plt.hist(df_drop['age']);
plt.hist([male_data.age, female_data.age], label=['Male', 'Female'])
plt.legend(loc='upper right')

plt.title("Age Distribution")
plt.ylabel('count of users')
plt.xlabel('age');
plt.show()
```


![png](/images/suicide-predictors/age.png)


Majority of the respondents are `male` and `straight` and aged from `15-30` 


```python
fig, ax = plt.subplots(1,2, figsize=(16,7))
total = df_drop.shape[0]
sns.countplot(x='willing_to_improve', data=df_drop, ax=ax[0])

ax[0].set_title('willingness to improve')
ax[0].set_axisbelow(True)
ax[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[0].set_ylabel('count')

# percent labels
for p in ax[0].patches:
    height = p.get_height()
    ax[0].text(p.get_x()+p.get_width()/2,
               height + 2,
               f'{height/total:.0%}',
               ha="center",
               weight='semibold',
               fontsize=16) 

total = df_drop.shape[0]
sns.countplot(x='want_help', data=df_drop, ax=ax[1])

ax[1].set_title('wants help')
ax[1].set_axisbelow(True)
ax[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[1].set_ylabel('count')

# percent labels
for p in ax[1].patches:
    height = p.get_height()
    ax[1].text(p.get_x()+p.get_width()/2,
               height + 2,
               f'{height/total:.0%}',
               ha="center",
               weight='semibold',
               fontsize=16) 

plt.tight_layout()
```


![png](/images/suicide-predictors/improve-help.png)


Almost half of the users say they do not want help but 70% are actually willing to improve themselves. This could imply that a significant number of them are uncomfortable of exteranl help or reaching out to someone else. 


```python
fig, ax = plt.subplots(1,2, figsize=(16,7))
total = df_drop.shape[0]
sns.countplot(x='depressed', data=df_drop, ax=ax[0])

ax[0].set_title('depression')
ax[0].set_axisbelow(True)
ax[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[0].set_ylabel('count')

# percent labels
for p in ax[0].patches:
    height = p.get_height()
    ax[0].text(p.get_x()+p.get_width()/2,
               height + 2,
               f'{height/total:.0%}',
               ha="center",
               weight='semibold',
               fontsize=16) 

total = df_drop.shape[0]
sns.countplot(x='social_fear', data=df_drop, ax=ax[1])

ax[1].set_title('social fear / anxiety')
ax[1].set_axisbelow(True)
ax[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[1].set_ylabel('count')

# percent labels
for p in ax[1].patches:
    height = p.get_height()
    ax[1].text(p.get_x()+p.get_width()/2,
               height + 2,
               f'{height/total:.0%}',
               ha="center",
               weight='semibold',
               fontsize=16) 

plt.tight_layout()
```


![png](/images/suicide-predictors/depression-anxiety.png)


Among the participants, more than half identify themselves as depressed and with social fear - which are strong predictors of suicide attempts according to previous studies. 


```python
fig, ax = plt.subplots(1,2, figsize=(16,7))
total = df_drop.shape[0]
sns.countplot(x='education', data=df_drop, ax=ax[0])

ax[0].set_title('education')
ax[0].set_axisbelow(True)
ax[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[0].set_ylabel('count')

# percent labels
for p in ax[0].patches:
    height = p.get_height()
    ax[0].text(p.get_x()+p.get_width()/2,
               height + 2,
               f'{height/total:.0%}',
               ha="center",
               weight='semibold',
               fontsize=16) 

total = df_drop.shape[0]
sns.countplot(x='employment', data=df_drop, ax=ax[1])

ax[1].set_title('employment')
ax[1].set_axisbelow(True)
ax[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[1].set_ylabel('count')

# percent labels
for p in ax[1].patches:
    height = p.get_height()
    ax[1].text(p.get_x()+p.get_width()/2,
               height + 2,
               f'{height/total:.0%}',
               ha="center",
               weight='semibold',
               fontsize=16) 

plt.tight_layout()
```


![png](/images/suicide-predictors/educ-employment.png)


More than 75% of the respondents are employed and studying. More than half have / are taking up college and post-graduate degrees. 


```python
fig, ax = plt.subplots(1,2, figsize=(16,7))
total = df_drop.shape[0]
sns.countplot(x='race', data=df_drop, ax=ax[0])

ax[0].set_title('race')
ax[0].set_axisbelow(True)
ax[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[0].set_ylabel('count')

# percent labels
for p in ax[0].patches:
    height = p.get_height()
    ax[0].text(p.get_x()+p.get_width()/2,
               height + 2,
               f'{height/total:.0%}',
               ha="center",
               weight='semibold',
               fontsize=16) 

total = df_drop.shape[0]
sns.countplot(x='income', data=df_drop, ax=ax[1])

ax[1].set_title('income')
ax[1].set_axisbelow(True)
ax[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax[1].set_ylabel('count')

# percent labels
for p in ax[1].patches:
    height = p.get_height()
    ax[1].text(p.get_x()+p.get_width()/2,
               height + 2,
               f'{height/total:.0%}',
               ha="center",
               weight='semibold',
               fontsize=16) 

plt.tight_layout()
```


![png](/images/suicide-predictors/race-income.png)


Majority of the respondents consider themselves as `low income` earners. 


```python
df_friend = df_drop[['# of friends', 'attempt_suicide']]
```

```python
df_friend_yes = df_friend[df_friend['attempt_suicide']=='Yes']
df_friend_no = df_friend[df_friend['attempt_suicide']=='No']
```


```python
df.loc[df['friends']<=50,:].groupby('attempt_suicide')['friends'].plot.hist(legend=True, 
                                                                                 density=True);
```


![png](/images/suicide-predictors/friends.png)



```python
df.groupby('social_fear')['friends'].plot.hist(legend=True)
```


![png](/images/suicide-predictors/social-fear.png)


It can be seen from the last two plots above that most people who have attempted suicide have less number of friends compared to those who don't and people with social fear have less friends. 

Now let use machine learning techniques to predict which of these variables heavily identify the likelihood of suicide attempt. Let us then pick the best technique that results in the highest accuracy. 

## Modelling (Classification)


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import model_selection, preprocessing
```

### PCC


```python
num=(new_dummy.groupby('attempt_suicide').size()/new_dummy.groupby('attempt_suicide').size().sum())**2
print("Proportion Chance Criterion = {}%".format(100*num.sum()))
print("1.25*Proportion Chance Criterion = {}%".format(1.25*100*num.sum()))
```

    Proportion Chance Criterion = 70.3220116293343%
    1.25*Proportion Chance Criterion = 87.90251453666787%



```python
data = new_dummy.drop('attempt_suicide', axis=1)
y = new_dummy['attempt_suicide']
```

Let us now divide our dataset into training and test set with the `attempted suicide` as target label and using `15 features`. 


```python
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=143)
```

It can be observed that our data is significantly unbalanced to non-attempts (at 80-20). This means that this class has more observations than the other one and could result in a bias prediction. 


```python
new_dummy['attempt_suicide'].value_counts()
```




    0    384
    1     85
    Name: attempt_suicide, dtype: int64



So we `oversample` the unbalanced data using `SMOTE` which generate synthetic data points based on the existing dataset. 


```python
df_oversampled = SMOTE(random_state=1136)
```


```python
X_res, y_res = df_oversampled.fit_resample(X_train, y_train)
```

## Using GridSearch
The best model and hyperparameters were looked into using GridSearch. 


```python
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
```


```python
knn = KNeighborsClassifier(n_jobs=-1)
logres1 = LogisticRegression(penalty='l1', max_iter=1000,
                       solver='liblinear', n_jobs=-1)
logres2 = LogisticRegression(penalty='l2', max_iter=1000, n_jobs=-1)
linsvc1 = LinearSVC(penalty='l1', dual=False, max_iter=10000)
linsvc2 = LinearSVC(penalty='l2', max_iter=10000)
svc_rbf = SVC(kernel='rbf')
svc_poly = SVC(kernel='poly', degree=3)
dectree = DecisionTreeClassifier()
ranfor = RandomForestClassifier()
gradboost = GradientBoostingClassifier()
```


```python
kneighbors = range(1, 31)
C_list = np.logspace(-5, 5, num=10)
gamma_list = np.logspace(-5, 5, num=10)
depth_list = range(3,10)
learn_rate = np.logspace(-2, 0.5, num=10)
est_list = [150, 250, 350, 550, 750]
min_samples_leaf = [2, 3, 4]
max_features = [.5, .3, .2]

classifiers = [('kNN', knn, {'n_neighbors':kneighbors}),
              ('Logistic Regression (L1)', logres1, {'C':C_list}),
              ('Logistic Regression (L2)', logres2, {'C':C_list}),
              ('Linear SVM (L1)', linsvc1, {'C':C_list}),
              ('Linear SVM (L2)', linsvc2, {'C':C_list}),
              ('NonLinear SVM (RBF)', svc_rbf, {'C':C_list, 'gamma':gamma_list}),
              ('Decision Tree (DT)', dectree, {'max_depth':depth_list}),
              ('Random Forest (RF)', ranfor, {'max_depth':depth_list, 'n_estimators':est_list}),
              ('Gradient Boosting (GBM)', gradboost, {'max_depth':depth_list, 'learning_rate':learn_rate})
             ]
```


```python
models = {}
for cls in classifiers: 
    gs_cv = model_selection.GridSearchCV(cls[1], param_grid=cls[2], n_jobs=-1, scoring='accuracy')
    gs_cv.fit(X_res, y_res)
    models[cls[0]] = gs_cv
```

```python
cols = ['Machine Learning Method', 'Test Accuracy',
        'Best Parameter', 'Top Predictor Variable']
df_results = pd.DataFrame(columns=cols)

for i, m in enumerate(models):
    try:
        top_predictor = data.columns[
            np.argmax(np.abs(models[m].best_estimator_.coef_))]

    except AttributeError:
        top_predictor = np.nan
    
    df_results.loc[i] = [m, 
                 models[m].best_estimator_.score(X_test, y_test),
                 models[m].best_params_ ,
                 top_predictor]
```


```python
df_results
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
      <th>Machine Learning Method</th>
      <th>Test Accuracy</th>
      <th>Best Parameter</th>
      <th>Top Predictor Variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kNN</td>
      <td>0.703390</td>
      <td>{'n_neighbors': 2}</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression (L1)</td>
      <td>0.694915</td>
      <td>{'C': 599.4842503189421}</td>
      <td>race_american</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression (L2)</td>
      <td>0.694915</td>
      <td>{'C': 7742.636826811277}</td>
      <td>race_american</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Linear SVM (L1)</td>
      <td>0.694915</td>
      <td>{'C': 3.593813663804626}</td>
      <td>race_american</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Linear SVM (L2)</td>
      <td>0.703390</td>
      <td>{'C': 0.2782559402207126}</td>
      <td>race_middle eastern</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NonLinear SVM (RBF)</td>
      <td>0.771186</td>
      <td>{'C': 3.593813663804626, 'gamma': 0.2782559402...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Decision Tree (DT)</td>
      <td>0.728814</td>
      <td>{'max_depth': 7}</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Random Forest (RF)</td>
      <td>0.805085</td>
      <td>{'max_depth': 7, 'n_estimators': 150}</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Gradient Boosting (GBM)</td>
      <td>0.805085</td>
      <td>{'learning_rate': 0.03593813663804628, 'max_de...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Random Forest Classifier
This model is actually a powerful one as it uses many decision trees to make a prediction. These individual decision trees are uncorrelated and spits out a prediction each. The prediction with the most "votes" will then be the final prediction of the Random Forest model. This is conceptually great by the notion of `wisdom of crowds`. The individual decision trees "protect" themselves fromt he mistakes of the other trees and in return, spits out a more accurate collective prediction. 


```python
gs_cv.best_estimator_
```

```python
df_features = pd.DataFrame(gs_cv.best_estimator_.feature_importances_, index=data.columns).sort_values(0, ascending=True)
```


```python
df_features[df_features[0]>0].plot(kind='barh', legend=False);
```


![png](/images/suicide-predictors/top-predictors.png)


The prediction model identified `depression` is the highest predictor of suicide attempts followed by `education`. This is consistent with previous studies saying majority of suicide tendencies are found to experience anxiety disorders. Several studies also suggest that academic work and stresses are associated with suicide attempts as well.

## Validation


```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
```


```python
rfc = RandomForestClassifier()
rfc.fit(X_res,y_res)
rfc_predict = rfc.predict(X_test)
```


```python
roc_auc_score(y_test, rfc_predict)
```

    0.5460992907801419


```python
rfc_cv_score = cross_val_score(ranfor, X_res, y_res, cv=5, scoring='roc_auc')
```

    === Confusion Matrix ===
    [[87  7]
     [20  4]]
    
    
    === Classification Report ===
                  precision    recall  f1-score   support
    
               0       0.81      0.93      0.87        94
               1       0.36      0.17      0.23        24
    
       micro avg       0.77      0.77      0.77       118
       macro avg       0.59      0.55      0.55       118
    weighted avg       0.72      0.77      0.74       118
    
    
    
    === All AUC Scores ===
    [0.7956302  0.98023187 0.99762188 0.99598692 0.98498811]
    
    
    === Mean AUC Score ===
    Mean AUC Score - Random Forest:  0.9508917954815695


It can be observed that the precision and recall for `class 1` (suicide attempts) is low and this is actually what we want to accurately predict. These corresponds to the actual positives that were correctly predicted and the percent of positives that were predicted over the total of actual positives. Therefore, the high AUC score here is misleading as this could possibly account for the non-attempts instead.  

This could mean that even with oversampling, the data is still heavily biased to the non-attempts. A more sophisticated and efficient way could be used to address this issue in unbalanced data. 

## Conclusion and Recommendation

This study is subject to methodological limitations – data from the reddit survey included information about suicide attempts, demographics, and psychological status that was examined using simple questions and scales only. This might have affected the performance of the prediction models. The dataset to be used next could also be more representative of the different classes to make sure to remove bias. Additionally, the stigma around suicide leads to underreporting which critically affects data collection which is subsequently crucial to predicting suicide or suicide attempts. 

The study confirms that a machine learning model on mental health data (if made available) is capable of identifying predictors of suicide attempts and other critical suicide risk such as self-harm. Better approaches and algorithms could be tried using a bigger and more comprehensive dataset to further improve results. 


<h2>Document</h2>

The presentation deck for this project can be viewed [here](/files/suicide-predictors.pdf).
