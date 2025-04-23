import numpy as np # mathimatical operations specially on multi dimeention array 
import pandas as pd # data manapulaton and analysis 
from sklearn.model_selection import train_test_split  #  split the traning and testing data 
from sklearn.feature_extraction.text import TfidfVectorizer # Tfid.. is a technique to extract features from nlp in numerical format 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score # it is going to tell how much accurate the modle is 

df = pd.read_csv('c:/users/mueez/OneDrive/Desktop/Email Spam/mail_data.csv')
data = df.where((pd.notnull(df)),'') # if there is an empty space then replace it with '' by default it is Nan which can cause issue while programming
# print(data.head())
# print(data.info()) # Non-Null Count means 5572 entries that are not null
# print(data.shape) # tells the number of rows and columns
data.loc[data['Category'] == 'spam', 'Category'] = 0  # data.loc[rows_you_want_to_target, column_you_want_to_update]
data.loc[data['Category'] == 'ham', 'Category'] = 1

x = data['Message']
y = data['Category']


x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=3)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

feature_extraction = TfidfVectorizer(min_df= 1, stop_words='english', lowercase=True)
# It converts text data into numerical vectors
# min_df=1	Ignore words that appear in fewer than 1 document (i.e., keep all words).
# stop_words='english'	Removes common English words like “is”, “the”, “and”, etc.
# lowercase='True'	Converts all words to lowercase. (Note: should actually be lowercase=True without quotes — otherwise it’s a string, not a boolean. But might still work.)

X_train_features = feature_extraction.fit_transform(x_train)
X_test_features = feature_extraction.transform(x_test)

# fit learns the vocabulary  from training set 
# transform use the same vocabulary from training 

#Learns the vocabulary from your training data.
# Converts the training messages into a matrix of TF-IDF features.
# the one in ml slides it tells importance of that word in the document like 0.707 is the weightage of python in that sentence then in transform it checks that is there python in it if yes then it is going to mark its weigthage according to it and ignores the rest of the words

y_train = y_train.astype('int') # in category the 0 and 1 are still string e.g. '0' so we are converting them into string 
y_test = y_test.astype('int') 

model = LogisticRegression()
model.fit(X_train_features,y_train) # This trains the model using your training data.
prediction_on_trainning_data = model.predict(X_train_features)  # To check how well the model learned the training data. It is going to return [0,1,1 ...] 1st is spam second is ham ...
accuracy_on_training_data = accuracy_score(y_train,prediction_on_trainning_data) # check the accuracy
print(accuracy_on_training_data)

prediction_on_testing_data = model.predict(X_test_features) # to test how well it performs on unseen data
accuracy_on_testing_data = accuracy_score(y_test,prediction_on_testing_data)
print(accuracy_on_testing_data)

input = ["🎉 Congratulations! You've been selected to win a brand new iPhone 15. Click here to claim your prize now: http://fake-link.com"]
input_data_features = feature_extraction.transform(input)
prediction_on_input = model.predict(input_data_features)

if prediction_on_input[0] == 0:
    print('Spam')
else:
    print('Ham')
