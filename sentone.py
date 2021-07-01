 import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle


header_list = ["rev", "num","Start Date","query","user","tweets"]
df = pd.read_csv("trp.csv", names=header_list,encoding="latin1") 
df.drop(columns =["num","Start Date","query","user"],inplace = True)
df1=df.sample(frac=1)

c_tweets=[]
import re
for a_tweet in df1["tweets"]:
    a_tweet = re.sub(r'@\S+|https?://\S+', '',a_tweet)
    c_tweets.append(a_tweet)

df1["cleaned_tweets"] = c_tweets

df1.drop(columns = ["tweets"],inplace = True)

df1.reset_index(inplace = True) 
df1.drop(columns = ["index"],inplace = True)
df1.loc[df1["rev"]==4,"rev"]=1
df1.loc[df1["rev"]==0,"rev"]=0



X = df1["cleaned_tweets"] #assigning feature as X
tfidf_var = TfidfVectorizer() # storing classifier in a variable
X = tfidf_var.fit_transform(X) # tranforms each words into numerical vector form
y = df1["rev"]
X.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)

clf = LinearSVC()
fitted_clf=clf.fit(X_train,y_train)   

filename = 'linearsvc1600k001.sav'
pickle.dump(fitted_clf, open(filename, 'wb'))

y_pred = fitted_clf.predict(X_test)

def sentiment_s(text):
    vec = tfidf_var.transform([text])
    if clf.predict(vec)==1:
        return "i give positive opinion"
    else:
        return " i give negative opinion"


#sentiment("fuck this")