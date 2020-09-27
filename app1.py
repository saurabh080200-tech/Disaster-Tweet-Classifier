import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,precision_recall_curve
from sklearn.metrics import precision_score,recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def main():
    st.title("Disaster Tweets Classifier")
    st.markdown("### Is the Tweet Regarding Disaster Real or Not?? 💥")
    st.sidebar.title("Lets Check it out!!")

    choice=st.sidebar.selectbox("Choose an Option",("About","Describe","Classify"),key="choice")

    if choice=="About":
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.write("Twitter has become an important communication channel in times of emergency.")
        st.write("The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time.")
        st.write("Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies")
        st.write("But, it’s not always clear whether a person’s words are actually announcing a disaster.")
        st.write("with this web application we intend to check whether the tweet regarding the disaster is true or not")

    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv("train.csv")
        data.dropna(inplace=True)
        data.reset_index(drop=True,inplace=True)
        return data
    
    df=load_data()

    if choice=="Describe":
        if st.sidebar.checkbox("Show Raw Data",False):
            st.subheader("Disaster Tweets Dataset")
            st.write(df)
        
        st.sidebar.subheader("Show Random Tweet")
        random_tweet=st.sidebar.radio("Tweet",("Disastrous","Healthy"),key="random_tweet")

        if random_tweet=="Disastrous":
            filt=(df['target']==1)
            str1=df.loc[filt,"text"].sample(n=1).values
            st.sidebar.markdown(str1[0])
        else:
            filt=(df['target']==0)
            str1=df.loc[filt,"text"].sample(n=1).values
            st.sidebar.markdown(str1[0])
        
        st.sidebar.markdown("### Number of Tweets of Disastrous and Healthy")
        select=st.sidebar.selectbox("Visualization Type",("Barplot","Pie Chart"),key="select")

        df1=df.copy()
        df1['target']=df1['target'].replace({1:"Disastrous",0:"Healthy"})
        sentiment=df1['target'].value_counts()
        df3=pd.DataFrame({"target":sentiment.index,"Tweets":sentiment.values})

        if not st.sidebar.checkbox("Hide",True):
            st.text(" ")
            st.text(" ")
            st.markdown("### Number of Tweets of Disaster and Healthy")
            if select=="Barplot":
                fig=px.bar(df3,x="target",y="Tweets",color="Tweets",height=500,width=500)
                st.plotly_chart(fig)
            
            if select=="Pie Chart":
                fig=px.pie(df3,names="target",values="Tweets",opacity=1)
                st.plotly_chart(fig)

        st.sidebar.subheader("Word Cloud")
        word_tweet=st.sidebar.radio("Word Cloud for which Tweets",("Disastrous","Healthy"),key="word_tweet")

        if not st.sidebar.checkbox("Hide",True,key="shut"):
            st.text(" ")
            st.text(" ")
            st.subheader("Word Cloud for %s Tweet" %(word_tweet))
            
            df2=df1[df1['target']==word_tweet]
            words= " ".join(df2['text'])
            processed_words=" ".join([word for word in words.split() if "http" not in word and not word.startswith("@")])
            wordcloud=WordCloud(stopwords=STOPWORDS,background_color="white",height=440,width=640).generate(processed_words)
            plt.imshow(wordcloud)
            plt.xticks([])
            plt.yticks([])
            st.pyplot()
    
    if choice=="Classify":
        
        @st.cache(persist=True)
        def split(df):
            y=df['target']
            x=df['text']
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
            return x_train,x_test,y_train,y_test

        x_train,x_test,y_train,y_test=split(df)
        
        count_vector=CountVectorizer(stop_words="english")
        training_data=count_vector.fit_transform(x_train)
        testing_data=count_vector.transform(x_test)

        naive_bayes=MultinomialNB()
        naive_bayes.fit(training_data,y_train)
        pred=naive_bayes.predict(testing_data)

        y_test=y_test.values
        st.subheader("Our Model Result..")
        st.write("Precision",precision_score(y_test,pred).round(2))
        st.write("Recall",recall_score(y_test,pred).round(2))

        df4=pd.DataFrame({"Actual":y_test,"Predicted":pred})
        df4=df4.replace({1:"Disastrous",0:"Healthy"})
        
        st.text(" ")
        st.text(" ")
        st.subheader("Bar Plot for the Actual values")
        actual=df4['Actual'].value_counts()
        df5=pd.DataFrame({"Actual Values":actual.index,"Tweets":actual.values})
        fig=px.bar(df5,x="Actual Values",y="Tweets",color="Tweets",height=500,width=500)
        st.plotly_chart(fig)
        
        st.subheader("Bar Plot for the Predicted Values")
        predicted=df4['Predicted'].value_counts()
        df6=pd.DataFrame({"Predicted Values":predicted.index,"Tweets":predicted.values})
        fig1=px.bar(df6,x="Predicted Values",y="Tweets",color="Tweets",height=500,width=500)
        st.plotly_chart(fig1)

        if st.checkbox("Want to see if your tweet is disastrous or not??",False):
            st.markdown("Press Ctrl+Enter to Apply...")
            s=st.text_area("Enter Your Tweet 🐦",key="S")
            x=s.split()
            user_input=str(x)
            user_input=[user_input]
            user_test=count_vector.transform(user_input)
            user_pred=naive_bayes.predict(user_test)
            if user_pred[0]==1:
                st.markdown("### Disastrous")
            elif user_pred[0]==0 and len(s)>0:
                st.markdown("### Healthy")

if __name__=="__main__":
    main()