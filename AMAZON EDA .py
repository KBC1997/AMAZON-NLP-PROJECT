############################### PACKAGES REQUIRED ################
import pandas as pd
import numpy as np
import nltk                            # NATURAL LANGUAGE PREOCESSING TOOLKIT.
from nltk.corpus import stopwords
import re                              # REGULAR EXPRESSIONS.
import matplotlib.pyplot as plt
from wordcloud import WordCloud

## To import the data of saved reviews

lifebouy_view_df= pd.read_excel(r'C:\Users\KIRAN BC\Desktop\PROJECT 2 NLP\lifebuoy_reviews.xlsx')

lifebouy_view_df.isnull().sum()#754
lifebouy_view_df.dropna(inplace=True)
lifebouy_view_df.isnull().sum()
lifebouy_view_df.head#752

lifebuoy_view = lifebouy_view_df[lifebouy_view_df.columns[3]].tolist()
## Data cleaning

handwash_rev_string="".join(lifebuoy_view)
handwash_rev_string=re.sub("[^A-Za-z" "]+"," ",handwash_rev_string).lower()
handwash_rev_string=re.sub("[0-9" "]+"," ",handwash_rev_string)

handwash_rev_words=handwash_rev_string.split(" ")
lifebuoy_rev_words=handwash_rev_string.split(" ")

## Importing stopwords  

with open("D:\\DATA SCIENCE\\JAN162020\\ASSIGNMENTS\\TEXT MINING\\stop.txt","r") as sw:
    stopwords = sw.read()
stopwords=stopwords.split("\n")
handwash_reviews_words=[w for w in handwash_rev_words if w not in stopwords]
handwash_reviews_strings=" ".join(handwash_reviews_words)

#Sentence Tokenization
from nltk.tokenize import sent_tokenize
tokenized_text=sent_tokenize(handwash_reviews_strings)
print(tokenized_text)

#Word Tokenization
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(handwash_reviews_strings)
print(tokenized_word)

#Frequency Distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)  #<FreqDist with 1483 samples and 5136 outcomes>
fdist.most_common(5)
# Frequency Distribution Plot
fdist.plot(30,cumulative=False)
plt.show()

## word cloud all reviews

wordcloud_lifebouy=WordCloud(
        background_color='black',
        mode='RGB',
        width=1800,
        height=1600
        ).generate(handwash_reviews_strings)
plt.imshow(wordcloud_lifebouy)

# wc = WordCloud(background_color="white", max_words=200, width=400, height=400, random_state=1).generate(handwash_reviews_strings)
# plt.imshow(wc)

# positive words # Choose the path for +ve words stored in system

with open("D:\\DATA SCIENCE\\JAN162020\\ASSIGNMENTS\\TEXT MINING\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]

# negative words  Choose path for -ve words stored in system
with open("D:\\DATA SCIENCE\\JAN162020\\ASSIGNMENTS\\TEXT MINING\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
lifebouy_neg_in_neg = " ".join ([w for w in handwash_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(lifebouy_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
lifebouy_pos_in_pos = " ".join ([w for w in handwash_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(lifebouy_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)

## sentimental analysis
# Create quick lambda functions to find the polarity and subjectivity of each routine
# Terminal / Anaconda Navigator: conda install -c conda-forge textblob

from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

lifebouy_view_df['polarity'] = lifebouy_view_df['Reviews'].apply(pol)
lifebouy_view_df['subjectivity'] = lifebouy_view_df['Reviews'].apply(sub)
lifebouy_view_df

# creating a function to compute the negative, neutral and positive analysis

def getAnalysis(score):
    if score < 0:
        return '-1'
    elif score ==0:
        return '0'
    else:
        return '+1'

lifebouy_view_df['sentiment'] = lifebouy_view_df['polarity'].apply(getAnalysis)
print(lifebouy_view_df.sentiment.value_counts())
print(lifebouy_view_df.sentiment.value_counts(normalize=True) * 100)

#bar plot
fig, ax = plt.subplots(figsize=(5, 5))
lifebouy_view_df['sentiment'].value_counts(normalize=True).plot(kind='bar'); 
ax.set_xticklabels(['Positive', 'Neutral', 'Negative'])
ax.set_ylabel("Percentage")
plt.show()

lifebouy_view_df.to_csv(r'C:\Users\KIRAN BC\Desktop\PROJECT 2 NLP\lifebouy_754_ps.csv',index=False,header=True)  

## print all of the postive reviews by polarity 
j = 1
sortedDF = lifebouy_view_df.sort_values(by=['polarity'])
for i in range(0, sortedDF.shape[0]):
    if(sortedDF['sentiment'][i] == '+1'):
      print(str(j) + ')' + sortedDF['Reviews'][i])
      print()
      j= j+1

## print all of the negative reviews by polarity 
j = 1
sortedDF = lifebouy_view_df.sort_values(by=['polarity'],ascending='False')
for i in range(0, sortedDF.shape[0]):
    if(sortedDF['sentiment'][i] == '-1'):
      print(str(j) + ')' + sortedDF['Reviews'][i])
      print()
      j= j+1
     
from nltk.tokenize import word_tokenize, RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
def process_text(headlines):
    tokens = []
    for line in headlines:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stopwords]
        tokens.extend(toks)    
    return tokens

pos_lines = list(lifebouy_view_df[lifebouy_view_df.sentiment == '+1'].Reviews)
pos_tokens = process_text(pos_lines)
pos_freq = nltk.FreqDist(pos_tokens)
pos_freq.most_common(20)        
                                    
neg_lines = list(lifebouy_view_df[lifebouy_view_df.sentiment == '-1'].Reviews)
neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)
neg_freq.most_common(20)

#positive word frequency      
y_val = [x[1] for x in pos_freq.most_common()]
fig = plt.figure(figsize=(10,5))
plt.plot(y_val)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Positive)")
plt.show()   
#Negative word frequency
y_val = [x[1] for x in neg_freq.most_common()]
fig = plt.figure(figsize=(10,5))
plt.plot(y_val)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Negative)")
plt.show() 

#########
######## STOCK PRICE
########### For Three months may-june-july
stock=pd.read_csv(r"C:\Users\KIRAN BC\Desktop\PROJECT 2 NLP\MY CODE\01-01-2020-TO-10-08-2020HINDUNILVRALLN.csv")
stock.dtypes
type(stock)
stock.columns
stock_3months = stock.iloc[82:,]
stock_3months.plot()
stock_3months.hist()
######################## High Price vs Low Price

x = stock_3months['Date']
y1 = stock_3months['High Price']
y2 = stock_3months['Low Price']

# Plot Line1 (Left Y Axis)
fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
ax1.plot(x, y1, color='tab:red')
# Plot Line2 (Right Y Axis)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(x, y2, color='tab:blue')

# Decorations
# ax1 (left Y axis)
ax1.set_xlabel('Year', fontsize=20)
ax1.tick_params(axis='x', rotation=0, labelsize=12)
ax1.set_ylabel('high', color='tab:red', fontsize=20)
ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
ax1.grid(alpha=.4)
# ax2 (right Y axis)
ax2.set_ylabel("low", color='tab:blue', fontsize=20)
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_xticks(np.arange(0, len(x), 60))
ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
ax2.set_title("High vs Low: Plotting in Secondary Y Axis", fontsize=22)
fig.tight_layout()
plt.show()

######################## Open Price vs Close Price
x = stock_3months['Date']
y1 = stock_3months['Open Price']
y2 = stock_3months['Close Price']

# Plot Line1 (Left Y Axis)
fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
ax1.plot(x, y1, color='tab:red')
# Plot Line2 (Right Y Axis)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(x, y2, color='tab:blue')

# Decorations
# ax1 (left Y axis)
ax1.set_xlabel('Year', fontsize=20)
ax1.tick_params(axis='x', rotation=0, labelsize=12)
ax1.set_ylabel('open', color='tab:red', fontsize=20)
ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
ax1.grid(alpha=.4)
# ax2 (right Y axis)
ax2.set_ylabel("close", color='tab:blue', fontsize=20)
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_xticks(np.arange(0, len(x), 60))
ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
ax2.set_title("Open vs Close: Plotting in Secondary Y Axis", fontsize=22)
fig.tight_layout()
plt.show()



########## stock price vs date plotting.. ####

x= stock_3months['Date']
y= stock_3months['Close Price']
plt.figure(figsize=(30,10))
plt.plot(x,y ,color='blue')
plt.title('date vs close price',fontsize=28)
plt.xlabel(' stock Date',fontsize=28)
plt.ylabel('close price',fontsize=28)
plt.xticks(rotation=40)
plt.grid(linewidth=1)
plt.show()

######### turn over line graph
lifebouy_view_df.columns
stock_3months.columns
x= stock_3months['Date']
y= stock_3months['Turnover']
plt.figure(figsize=(30,10))
plt.plot(x,y ,color='blue')
plt.title('date vs turn over',fontsize=28)
plt.xlabel('stock Date',fontsize=28)
plt.ylabel('turn over',fontsize=28)
plt.xticks(rotation=40)
plt.grid(linewidth=1)
plt.show()

######### turn over line graph

######################## Open Price vs Close Price
x = stock_3months['Date']
y1 = stock_3months['No. of Trades']
y2 = stock_3months['% Dly Qt to Traded Qty']

# Plot Line1 (Left Y Axis)
fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
ax1.plot(x, y1, color='tab:red')
# Plot Line2 (Right Y Axis)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(x, y2, color='tab:blue')

# Decorations
# ax1 (left Y axis)
ax1.set_xlabel('Year', fontsize=20)
ax1.tick_params(axis='x', rotation=40, labelsize=12)
ax1.set_ylabel('No. of Trades', color='tab:red', fontsize=20)
ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
ax1.grid(alpha=.4)
# ax2 (right Y axis)
ax2.set_ylabel("% Dly Qt to Traded Qty", color='tab:blue', fontsize=20)
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_xticks(np.arange(0, len(x), 60))
ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
ax2.set_title("No. of Trades vs % Dly Qt to Traded Qty: Plotting in Secondary Y Axis", fontsize=22)
fig.tight_layout()
plt.show()

#######


