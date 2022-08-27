# 6301-Project

# Credit Line Increase Model Card

### Basic Information

* **Person or organization developing model**: Group 23 - Bernard Low, Kerry McKeever, Lejla Skahic, Nigel Tinotenda Nyajeka 
* **Model date**: August 2022
* **Model version**: 1.0
* **License**: MIT
* **Model implementation code**: [https://github.com/lskahic/6301-Project/blob/main/DNSC_6301_Example_Project.ipynb](DNSC_6301_Example_Project.ipynb)

### Intended Use
* **Primary intended uses**: This model is an *example* probability of default classifier, with an *example* use case for determining eligibility for a credit line increase.
* **Primary intended users**: Students in GWU DNSC 6301 bootcamp.
* **Out-of-scope use cases**: Any use beyond an educational example is out-of-scope.

### Training Data

* Data dictionary: 

| Name | Modeling Role | Measurement Level| Description|
| ---- | ------------- | ---------------- | ---------- |
|**ID**| ID | int | unique row indentifier |
| **LIMIT_BAL** | input | float | amount of previously awarded credit |
| **SEX** | demographic information | int | 1 = male; 2 = female
| **RACE** | demographic information | int | 1 = hispanic; 2 = black; 3 = white; 4 = asian |
| **EDUCATION** | demographic information | int | 1 = graduate school; 2 = university; 3 = high school; 4 = others |
| **MARRIAGE** | demographic information | int | 1 = married; 2 = single; 3 = others |
| **AGE** | demographic information | int | age in years |
| **PAY_0, PAY_2 - PAY_6** | inputs | int | history of past payment; PAY_0 = the repayment status in September, 2005; PAY_2 = the repayment status in August, 2005; ...; PAY_6 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; ...; 8 = payment delay for eight months; 9 = payment delay for nine months and above |
| **BILL_AMT1 - BILL_AMT6** | inputs | float | amount of bill statement; BILL_AMNT1 = amount of bill statement in September, 2005; BILL_AMT2 = amount of bill statement in August, 2005; ...; BILL_AMT6 = amount of bill statement in April, 2005 |
| **PAY_AMT1 - PAY_AMT6** | inputs | float | amount of previous payment; PAY_AMT1 = amount paid in September, 2005; PAY_AMT2 = amount paid in August, 2005; ...; PAY_AMT6 = amount paid in April, 2005 |
| **DELINQ_NEXT**| target | int | whether a customer's next payment is delinquent (late), 1 = late; 0 = on-time |

* **Source of training data**: GWU Blackboard, email `jphall@gwu.edu` for more information
* **How training data was divided into training and validation data**: 50% training, 25% validation, 25% test
* **Number of rows in training and validation data**:
  * Training rows: 15,000
  * Validation rows: 7,500

### Test Data
* **Source of test data**: GWU Blackboard, email `jphall@gwu.edu` for more information
* **Number of rows in test data**: 7,500
* **State any differences in columns between training and test data**: None

### Model details
* **Columns used as inputs in the final model**: 'LIMIT_BAL',
       'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
       'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
* **Column(s) used as target(s) in the final model**: 'DELINQ_NEXT'
* **Type of model**: Decision Tree 
* **Software used to implement the model**: Python, scikit-learn
* **Version of the modeling software**: 
Python version: 3.7.13
sklearn version: 1.0.2
* **Hyperparameters or other settings of your model**: 
```

License
MIT License

Copyright (c) 2021 jphall@gwu.edu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Pythons imports
[2]
1s
from sklearn import tree                                  # sklearn tree model for decision trees
from sklearn.model_selection import train_test_split      # for partitioning data
from sklearn.model_selection import cross_val_score       # for cross validation
from sklearn.metrics import roc_auc_score, accuracy_score # to assess decision tree perforamce

# to upload local files
import io
from google.colab import files             

import numpy as np                                   # array, vector, matrix calculations
import pandas as pd                                  # dataFrame handling

from matplotlib import pyplot as plt                 # plotting
import seaborn as sns                                # slightly better plotting  

SEED = 12345                                         # ALWAYS use a random seed for better reproducibility
[3]
0s
# print version information 
import sys
import sklearn
version = ".".join(map(str, sys.version_info[:3]))
print('Python version:', version)
print('sklearn version:', sklearn.__version__)
Python version: 3.7.13
sklearn version: 1.0.2
Upload training data
[4]
23s
 # special google collab command to upload a file from computer
uploaded = files.upload()

[ ]
type(uploaded) # what kind of Python object did we just create?
dict
[7]
0s
uploaded.keys() # what is stored in that Python object?
dict_keys(['credit_line_increase.csv'])
[6]
0s
# read uploaded data into a pandas dataframe
data = pd.read_csv(io.StringIO(uploaded['credit_line_increase.csv'].decode('utf-8')))
Data Dictionary
Name	Modeling Role	Measurement Level	Description
ID	ID	int	unique row indentifier
LIMIT_BAL	input	float	amount of previously awarded credit
SEX	demographic information	int	1 = male; 2 = female
RACE	demographic information	int	1 = hispanic; 2 = black; 3 = white; 4 = asian
EDUCATION	demographic information	int	1 = graduate school; 2 = university; 3 = high school; 4 = others
MARRIAGE	demographic information	int	1 = married; 2 = single; 3 = others
AGE	demographic information	int	age in years
PAY_0, PAY_2 - PAY_6	inputs	int	history of past payment; PAY_0 = the repayment status in September, 2005; PAY_2 = the repayment status in August, 2005; ...; PAY_6 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; ...; 8 = payment delay for eight months; 9 = payment delay for nine months and above
BILL_AMT1 - BILL_AMT6	inputs	float	amount of bill statement; BILL_AMNT1 = amount of bill statement in September, 2005; BILL_AMT2 = amount of bill statement in August, 2005; ...; BILL_AMT6 = amount of bill statement in April, 2005
PAY_AMT1 - PAY_AMT6	inputs	float	amount of previous payment; PAY_AMT1 = amount paid in September, 2005; PAY_AMT2 = amount paid in August, 2005; ...; PAY_AMT6 = amount paid in April, 2005
DELINQ_NEXT	target	int	whether a customer's next payment is delinquent (late), 1 = late; 0 = on-time
Basic Data Analysis
[ ]
data.shape # (rows,columns)
(30000, 26)
[8]
0s
data.columns # names of columns
Index(['ID', 'LIMIT_BAL', 'SEX', 'RACE', 'EDUCATION', 'MARRIAGE', 'AGE',
       'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
       'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'DELINQ_NEXT'],
      dtype='object')
[ ]
data.isnull().any() # check for missing values
ID             False
LIMIT_BAL      False
SEX            False
RACE           False
EDUCATION      False
MARRIAGE       False
AGE            False
PAY_0          False
PAY_2          False
PAY_3          False
PAY_4          False
PAY_5          False
PAY_6          False
BILL_AMT1      False
BILL_AMT2      False
BILL_AMT3      False
BILL_AMT4      False
BILL_AMT5      False
BILL_AMT6      False
PAY_AMT1       False
PAY_AMT2       False
PAY_AMT3       False
PAY_AMT4       False
PAY_AMT5       False
PAY_AMT6       False
DELINQ_NEXT    False
dtype: bool
[ ]
data.describe() # basic descriptive statistics

Double-click (or enter) to edit

[ ]
_ = data[data.columns].hist(bins=50, figsize=(15, 15)) # display histograms

[9]
0s
# Pearson correlation matrix
corr = data.corr() 
corr

Double-click (or enter) to edit

[ ]
# correlation heatmap
plt.figure(figsize=(10, 10))
_ = sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)

Train decision tree
[11]
0s
# assign basic modeling roles
# do not put demographic variables into a financial model!
y_name = 'DELINQ_NEXT'
X_names = ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
[12]
0s
# partition data for honest assessment
train_X, valid_test_X, train_y, valid_test_y = train_test_split(data[X_names], data[y_name], test_size=0.5, random_state=SEED) # split off training data
valid_X, test_X, valid_y, test_y = train_test_split(valid_test_X, valid_test_y, test_size=0.5, random_state=SEED) # split remainder into validation and test

# summarize 
print('Training data: %i rows and %i columns' % (train_X.shape[0], train_X.shape[1] + 1))
print('Validation data: %i rows and %i columns' % (valid_X.shape[0], valid_X.shape[1] + 1))
print('Testing data: %i rows and %i columns' % (test_X.shape[0], test_X.shape[1] + 1))

# housekeeping
del valid_test_X 
del valid_test_y
Training data: 15000 rows and 20 columns
Validation data: 7500 rows and 20 columns
Testing data: 7500 rows and 20 columns
[13]
4s
# train decision tree 
# with validation-based early stopping

max_depth = 12
candidate_models = {}

for depth in range(0, max_depth):

  clf = tree.DecisionTreeClassifier(max_depth = depth + 1, random_state=SEED)
  clf.fit(train_X, train_y)

  train_phat = clf.predict_proba(train_X)[:, 1]
  valid_phat = clf.predict_proba(valid_X)[:, 1]

  train_auc = roc_auc_score(train_y, train_phat) #auc = area under the curve be around .6 to .8
  valid_auc = roc_auc_score(valid_y, valid_phat)

  cv_scores = cross_val_score(clf, valid_X, valid_y, scoring='roc_auc', cv=5)
  cv_std = np.std(cv_scores)

  candidate_models[depth + 1] = {}
  candidate_models[depth + 1]['Model'] = clf
  candidate_models[depth + 1]['Training AUC'] = train_auc
  candidate_models[depth + 1]['Validation AUC'] = valid_auc
  candidate_models[depth + 1]['5-Fold SD'] = cv_std
[14]
0s
# plot tree depth vs. training and validation AUC
# using simple pandas plotting and matplotlib

candidate_results = pd.DataFrame.from_dict(candidate_models, orient='index')
fig, ax = plt.subplots(figsize=(8, 8))
_ = candidate_results[['Training AUC', 'Validation AUC']].plot(title='Iteration Plot',
                                                               ax=ax)
_ = ax.set_xlabel('Tree Depth')
_ = ax.set_ylabel('AUC')

[15]
0s
# view same results as a table, using pandas iloc to remove first column of table

candidate_results.iloc[:, 1:]

[17]
# plot the tree for human interpretation

best_model = candidate_models[6]['Model']
fig = plt.figure(figsize=(400, 70))
_ = tree.plot_tree(best_model,
                   feature_names=X_names,
                   class_names=['On time', 'Delinquent'],
                   filled=True)
best_model.get_params()
{'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': 6,
 'max_features': None,
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'random_state': 12345,
 'splitter': 'best'}
```
### Quantitative Analysis
Metrics used to evaluate your final model (AUC and AIR)
○ State the final values, neatly -- as bullets or a table, of the metrics for all data:
training, validation, and test data
○ Provide any plots related to your data or final model -- be sure to label the plots!
#### Variable Histograms
![Variable Histograms](histograms.png)

○ Histograms showing the distribution of each variable in the data set.

#### Correlation Heatmap
![Correlation Heatmap](heatmap.png)

○ Heatmap shows a concerning correlation between race and predicted delinquency.


#### Initital Plot
![Initial Plot](initial_plot.png)

○ Initial plot comparing training and validation AUCs. Depth 6 provides best balance of fairness and accuracy, AUCs diverge greatly after that point.


#### Training Table
![Training Table](training_table.png)

○ Looking at the plot values in table form, the maximum Validation AUC does indeed occur at depth 6.


#### Test AUC
| Test AUC | 0.7438 |
| ---- | ------------- |

○ Test to see how well the model will do with completely new data. A result of 0.7438 falls within the ideal range of 0.6 to 0.9.


#### Initial AIR/AUC
| Race | AUC | AIR (to White) |
| ---- | --- | -------------- |
| White | 0.568 | - |
| Hispanic | 0.434 | 0.76 |
| Black | 0.465 | 0.82 |
| Asian | 0.568 | 1.00 |

○ For every 1000 white people that were granted credit line increaes, only 760 hispanics received the same. This is below the 80% threshold for bias, and thus a new cutoff needs to be used. An important thing to bear in mind is that while a higher cutoff provides greater accuracy, business needs means we should not stray too far from the standard of 15%.


#### Final AIR/AUC
| Race | AUC | AIR (to White) |
| ---- | --- | -------------- |
| White | 0.735 | - |
| Hispanic | 0.613 | 0.83 |
| Black | 0.626 | 0.85 |
| Asian | 0.739 | 1.00 |

○ Further calculation showed accuracy at an 18% cutoff provided balance between accuracy and business goals. Using the new cutoff, the ratio of Hispanics to Whites being approved for credit line increases above the 80% threshold.


#### Final Plot
![Final Plot](final_plot.png)

○ Final plot including Hispanic/White AIR for testing purposes due to them being identified as the race group that had the most concerning AIR. Depth 6 continues to provide the best balance between fairness and accuracy.

Ethical considerations
There are many potential negative impacts of using our model. The data used to train this model used demographic information, which inherently brings bias into the creation of this model tree. The variables used of race and gender, for exasmple, demonstrate the concept of disparate treatment, where we are actively making a business decision to include demographic information in our models. While we addressed that we are doing this for learning purposes and this is not a model that will be published, using demographic data incorporates ingrained biases from outdated structures that discriminated against people of color, for example. While this is inherently unfair for one, it is also illegal. The members of our group could be completely discredited or heavily fined for this decision. This data decision would make our software completely wrong.  
Additionally, by working in a group of like minded graduate students, our group was at risk of succumbing to the idea of confirmation bias, as well as group think. Although we came from different backgrounds, our similar knowledge levels and lack of experience in this field brought the risk of blind agreement with our partners without considering the consequences of certain decisions. This would also put our model at risk of being completely inconclusive and wrong.
 Real world risks. For similar reasons as described above, the mistakes made in the creation of the model would cause drastic real work risks. By including data that does not include the full picture of who pays back their loans, wrong conclusions are made as to who deserves these loans resulting in discriminatory outcomes. This means loans are not dealt to well deserving people, which in turn have drastic effects on their lives, but also our economy and beyond. 
Potential Uncertainties related to the impacts of our model: Math or software/ real world risks. The potential uncertainties related to the impacts of our model are numerous. First, it should be addressed that even if we correctly avoided using demographic data, it is quite impossible to avoid incorporating variables of bias as a whole. We cannot confirm that the variables are not introducing bias. Additionally, data as a whole cannot be blindly trusted and is not objective. Data records all our mistakes, biases, regrets. Data is not always accurate, and can be miscoded, corrupted, hacked or just wrong. As mentioned above, this leads to wrong conclusions and discriminatory outcomes. 
	Initially our task was to determine how certain metrics impact whether people receive a credit line increase or not. In inspecting this data, we determined that using certain variables causes the results to be inherently biased towards certain protected groups. In order to create a more fair, yet performing model, we utilized a decision tree and determined that using a tree depth of six creates the most holistic analysis where there is lower chance of those who receive loans to default on payment, but still diminishes bias against protected groups. We focused on the hispanic to white population AIR because the ratio was below the acceptable value of 80%. The 80% value means that for every 1000 people who get their credit line increased, only 800 hispanics would receive the same result, which is the standard accepted cutoff. By increasing this cutoff of probability of default from 15% to 18%, we received both a fairer and more accurate model. 
