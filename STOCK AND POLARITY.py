import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

data= pd.read_csv(r"C:\Users\KIRAN BC\Desktop\PROJECT 2 NLP\lifebouy_754_ps.csv")
nse = pd.read_csv(r"C:\Users\KIRAN BC\Desktop\PROJECT 2 NLP\MY CODE\01-01-2020-TO-10-08-2020HINDUNILVRALLN.csv")

nse_3months= nse.iloc[82:,]
nse_3months.isnull().sum() ## no null values

sorted_data = data.sort_values(by=['Date'])   # 754 total
sorted_data.isnull().sum()
sorted_data = sorted_data.dropna()            # removing na values and null values 752 total

nse_3months.columns
nse_3months['ClosePrice'] = MinMaxScaler().fit_transform(nse_3months['Close Price'].values.reshape(-1,1))
minMax = MinMaxScaler()

nse_3months['Date'] = pd.to_datetime(nse_3months['Date'])

data.shape
data.info()

day_review = pd.DataFrame(columns= ['polarity','subjectivity','sentiment'])
day_review[['polarity','subjectivity','sentiment']] = data[['polarity','subjectivity','sentiment']].astype(float)
day_review.index = pd.to_datetime(data['Date'])
day_review = day_review.resample('D').mean().ffill()

day_review['Date'] = day_review.index
day_review.reset_index(inplace= True,drop=True)

nse_merge_all_review = pd.merge(nse_3months,day_review,on='Date')

 
###################################     plots #############################
nse_merge_all_review.columns

######################## ClosePrice vs polarity
x = nse_merge_all_review['Date']
y1 = nse_merge_all_review['ClosePrice']
y2 = nse_merge_all_review['polarity']

# Plot Line1 (Left Y Axis)
fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
ax1.plot(x, y1, color='tab:red')
# Plot Line2 (Right Y Axis)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(x, y2, color='tab:blue')
# Decorations
# ax1 (left Y axis)
ax1.set_xlabel('date', fontsize=20)
ax1.tick_params(axis='x', rotation=40, labelsize=12)
ax1.set_ylabel('ClosePrice', color='tab:red', fontsize=20)
ax1.tick_params(axis='y', rotation=40, labelcolor='tab:red' )
ax1.grid(alpha=.4)
# ax2 (right Y axis)
ax2.set_ylabel("polarity", color='tab:blue', fontsize=20)
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_xticks(np.arange(0, len(x), 60))
ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
ax2.set_title("closing price vs polarity : Plotting in Secondary Y Axis", fontsize=22)
fig.tight_layout()
plt.show()

######################   single plot --- polarity ################

x= nse_merge_all_review['Date']
y= nse_merge_all_review['polarity']
plt.figure(figsize=(30,10))
plt.plot(x,y ,color='blue')
plt.title('date vs polarity',fontsize=28)
plt.xlabel('stock Date',fontsize=28)
plt.ylabel('polarity',fontsize=28)
plt.xticks(rotation=40)
plt.grid(linewidth=1)
plt.show()


############### single plot --- CLOSING PRICE ALL ##########

x= nse['Date']
y= nse['Close Price']
plt.figure(figsize=(50,20))
plt.plot(x,y ,color='blue')
plt.title('date vs clsoing price',fontsize=28)
plt.xlabel('stock Date',fontsize=10)
plt.ylabel('closing price',fontsize=10)
plt.xticks(rotation=90)
plt.show()

##################### 

