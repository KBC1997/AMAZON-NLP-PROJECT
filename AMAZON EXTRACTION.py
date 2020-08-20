
#####################################    PACKAGES /MODEULES REQUIRED   #################################

import requests                        # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs    # Beautifulsoup is for web scrapping...used to scrap specific content 
import re                              # regular expressions 
import pandas as pd
  
##########################################  Extracting Reviews  ################################################

lifebuoy_review=[]
lifebouy_date=[]
for i in range(1,100):
    respons=requests.get('https://www.amazon.in/Lifebuoy-Alcohol-Based-Protection-Sanitizer/product-reviews/B0866JTZXN/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber='+str(i))
    soup=bs(respons.content, 'html.parser')
    lifebuoy_rev=soup.findAll('span', attrs={"class","a-size-base review-text review-text-content"})
    lifebouy_dt = soup.findAll("span",attrs={"class":"a-size-base a-color-secondary review-date"})
    for i in range(len(lifebuoy_rev)):
        ip=[]
        dp=[]
        ip.append(lifebuoy_rev[i].text)
        dp.append(lifebouy_dt[i].text)
        lifebuoy_review=list(lifebuoy_review+ip)
        lifebouy_date=list(lifebouy_date+dp)

###############################  creating dataframe of date and review  ######################################

type(lifebuoy_review)
lifebouy_df_all=pd.DataFrame()
lifebouy_df_all['review']=lifebuoy_review
lifebouy_df_all['date']=lifebouy_date

lifebouy_df_all.to_excel(r'C:\Users\KIRAN BC\Desktop\PROJECT 2 NLP\lifebouy_reviews.xlsx',index=True,header=True)

############# 