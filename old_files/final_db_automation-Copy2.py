#!/usr/bin/env python
# coding: utf-8

# # Completely Automated Univariate Disk Storage Forecasting Modeller

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose 
from pmdarima import auto_arima
import matplotlib.animation as animation
import warnings

warnings.filterwarnings('ignore')



# In[2]:


data=pd.read_csv("/home/stackup/day41_usage.txt",delimiter="\t")
rc=data.shape[0]
data.head(rc)

def flag_anomaly_moving_average(series, window, scale = 3):
    fa = False
    
    if len(series)<=window:
        return fa, None, None, None 
    
    series = pd.DataFrame(series)
    rolling_mean = series.rolling(window=window).mean()
    mae = mean_absolute_error(series[window:], rolling_mean[window:])
    deviation = np.std(series[window:] - rolling_mean[window:])
    lower_bond = rolling_mean - (mae + scale * deviation)
    upper_bond = rolling_mean + (mae + scale * deviation)
    lower_bond = pd.DataFrame(lower_bond)
    upper_bond = pd.DataFrame(upper_bond)
    ul = float(upper_bond.iloc[-1:,0])
    ll = float(lower_bond.iloc[-1:,0])
    lv = float(series.iloc[-1:,0])
    rolling_mean = float(rolling_mean.iloc[-1:,0])

    if lv < ll or lv > ul:
        fa = True
    
    return fa,ul,ll,rolling_mean
# In[3]:

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

boxname='Anygraf4'

fpts = 15

conf_per =0.85

deviation_scale = 1.96

deviation_window = 10

slope_window = 15

result=pd.DataFrame()

# In[4]:
k

data[boxname].iloc[5:rc]=pd.to_numeric(data[boxname].iloc[5:rc])


# In[5]:


df=pd.DataFrame()
df['Date']=data['Date'].iloc[5:rc]
df['Usage']=data[boxname].iloc[5:rc]
df['Usage']=df["Usage"]/1000

# In[6]:


df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df=df.iloc[:,1:]

# decomp only works when our datetime index is present in equal intervals of time, 
# so in our case it isn't, further optimazation of our dataset needs to be researched 

# def decomp(df):
#     decomposition = seasonal_decompose(df,two_sided=False,extrapolate_trend=1,model='additive')
#     decomposition.plot()
#     plt.show()
    

# In[7]:


from pandas.plotting import register_matplotlib_converters 


# In[8]:


register_matplotlib_converters()


# In[9]:


# days=data[boxname].iloc[5:rc].values
# f=plt.figure(figsize=(14,14))
# nor=f.add_subplot(211)
# plt.title("without day consideration")
# act=f.add_subplot(212)
# plt.title("with day consideration")
# nor.plot(days)
# nor.scatter(range(0,rc-5),days,color='r')
# act.plot(df)
# act.scatter(df.index,df['Usage'],color='r')


# In[10]:


def adftest(c):
    result=adfuller(c)
    if result[1]<=0.05:
        return True
    elif result[1]>0.05:
        return False
    else:
        return None

def datacompat(ar,k,cf):
    preds=[np.nan]*(k-1)
    preds=preds+ar[:cf+1]
    return preds


i=np.zeros(5)
i =i.tolist()


def linreg(b):
    n=len(b)
    s=0
    for i in range(n):
        s=s+np.power(i,2)
    denom=s-n*np.power(np.mean(range(0,n)),2)
    ns=0
    for i in range(n):
        ns=ns+i*float(b[i])
    
    numer=ns-n*np.mean(b)*np.mean(range(0,n))
    
    k=numer/denom
    
    return k

def normal_dist_exception(b,scale):

    if abs(b[-1]-np.mean(b)) > abs(np.std(b)*scale - np.mean(b)):
        return True

    else:
        return False


    # a=np.mean(b)-k*np.mean(range(0,n))
    # lr=[]
    # for i in range(int(n)):
    #     lr.append(a+k*i)
        
    # plt.title(label=comment)
    # #plt.xlabel("No of projects",color='b')
    # #plt.ylabel("Disk space used in GB",color='b')
    # plt.scatter(range(0,n),b,color='g',label="training points")
    # plt.plot(lr,color='r',label="best fit line")

# In[11]:



#--------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------#


def animate(_, result, i):
    i.append(0)

    if len(i) < 5:
        return None

    k = len(i) 

    print(k)
    #input("Press Enter to continue...")
    
    if k >= df.shape[0]:
        input("Press Enter to quit...")
        exit()

    d=[]
    d.append(0)
    for i in range(df.shape[0]-1):
        d.append(df['Usage'].iloc[i+1]-df['Usage'].iloc[i])

    df['Diff_Usage']=d


    # In[12]:


    u=[]
    for i in d:
        u.append(i-np.mean(d))
    #plt.scatter(range(0,len(u)),u,color='r')
    u=pd.Series(u)
    df['Diff_Zero']=u.values
    #plt.plot(u)


    # In[13]:

    #linf.fit(n)

    stationarity = False

    while stationarity != True:
        stationarity = adftest(u)

    if stationarity == True :
        print("It's stationary") 

    # In[14]:


    #df.iloc[:20,0].to_csv('c1.csv')

    # In[15]:


    #df.iloc[:,1:].plot(figsize=(14,10))
    #plt.scatter(df.index,df["Diff_Usage"],color='r')

    #plot_acf(df.iloc[:,:1]);
    #plot_pacf(df.iloc[:,:1]);


    # diff_no=0
    # while adftest(d)!=True:
    #     adftest(a)
    #     diff_no=diff_no+1
    #     

    # In[18]:


    train=df.iloc[:k,2:]
    test=df.iloc[k:,2:]

    Arima_model = auto_arima(train, start_p=0, start_q=0, max_p=9, max_q=9,d=0,
                             stationary=False, n_iter=200, information_criterion='aic',
                             alpha=0.05, seasonal=False, trace=True, error_action='ignore', 
                             supress_warnings=True, stepwise=True,trend='nc', transparams = False)


    # In[19]:


    #Arima_model


    # In[20]:


    initial=ARIMA(train, order=Arima_model.order)
    fitted=initial.fit()
    fc=fitted.forecast(fpts+5,alpha=conf_per)
    lb=[]
    temp1=float(df["Usage"][k-1])
    for i in fc[2]:
        temp1=i[0]+temp1+np.mean(d)
        lb.append(temp1)
    ub=[]
    temp2=float(df["Usage"][k-1])
    for i in fc[2]:
        temp2=i[1]+temp2+np.mean(d)
        ub.append(temp2)

    f=[]
    f.append(df['Usage'][k-1])
    temp=float(df["Usage"][k-1])
    for i in fc[0]:
        temp=i+temp+np.mean(d)
        f.append(temp)


    # In[40]:


    #print(Arima_model.summary())


    # In[21]:


    df["Diff_Usage"].shape[0]
    len(f)


    # In[22]:


    l=df.shape[0]
    cf=l-k


    # In[23]:


    range(k-1,k+cf+1)
    range(cf+2)


    # In[24]:
    
    

    ax.clear()
    plt.figure(figsize=(18,12))
    #plt.xticks(np.arange(0,l+1, 1.0))
    plt.xticks(rotation=45)
    ax.plot(df["Usage"][:k].values,label='historical consumption')
    #ax.scatter(range(k,l),df["Usage"][k:l].values,label='actual consumption',color='g',marker='.',s=150)
    #ax.plot(range(k,k+fpts),df.Usage[k:k+fpts].values,label='actual consumption',color='g')#,s=50)
    #ax.scatter(range(k,k+fpts),df.Usage[k:k+fpts].values,label='actual consumption',color='g',s=50)
    ax.scatter(range(k,k+fpts),f[1:fpts+1],label='predicted consumption',color='k',s=50)
    ax.plot(range(k-1,k+fpts+1),f[0:fpts+2], label='predicted consumption')
    ax.fill_between(range(k,k+fpts+1),lb[:fpts+1],ub[:fpts+1],color='k',alpha=0.15,label='confidence interval')
    ax.legend(loc='upper left',fontsize=18)
    #plt.plot(range(12,),df["Diff_Usage"].iloc[12:].values)

    if fpts < slope_window:
        raise Exception('Slope window greater than prediction window')

    #--------------------------------------LINREG CALL-------------------------------------------------------------------#    
    # if k>slope_window:
    #     if linreg(train.Diff_Zero[-slope_window:].to_list()) * linreg(f[:slope_window]) < 0:
    #         #ax.clear()
    #         ax.annotate('Not fit for Prediction', xy = (k/2,np.mean(f+df.Usage[:k].to_list())), size = 15, color = 'r')
            #return None

    #--------------------------------------LINREG CALL-------------------------------------------------------------------#


    if k > 10: 
        if normal_dist_exception(train.Diff_Zero[-deviation_window:].to_list(), deviation_scale):
            ax.annotate('Not fit for Prediction', xy = (k/2,np.mean(f+df.Usage[:k].to_list())), size = 15, color = 'r')


    
    # In[25]:


    #df["Usage"][k:l].values


    # In[26]:


    #df.index[k:k+cf]


    # In[27]:


    # In[28]:


    # preds  =datacompat(f,k,cf)
    # lowerb =datacompat(lb,k,cf)
    # upperb =datacompat(ub,k,cf) 


    # # In[29]:


 
    # result['Actual'] = df["Usage"][:l].values
    # result['Predictions'] = preds
    # result['Upper_bound'] = upperb
    # result['Lower_bound'] = lowerb
    # result.head()


    # # In[30]:


    # result['Dates']=df.index[:k+cf] 
    # result.reset_index(inplace=True)
    # result = result.set_index('Dates')
    # result=result.iloc[:,1:]


    # result['Predictions'] = result['Predictions'].astype('float')
    # result['Observed'] = result['Observed'].astype('float')
    # result['Predictions']=result['Predictions'].div(1000)
    # result['Observed']= result['Observed'].div(1000)


    # result=result.astype('float')
    # result=result.div(1000)


    # # In[32]:


    # result.iloc[k-1:,2:4]=result.iloc[k-1:,2:4].shift(periods=1)


    # ## All figures in GB

    # In[33]:


    # result.head(result.shape[0])


    # In[34]:


    # a=result.iloc[:k,0].plot(figsize=(15,11))
    # result.iloc[:,1].plot(color='#ffa500')
    # a.scatter(result.index,result.Predictions,color='k',s=30,label='Predicted Points')
    # #a.scatter(result[k:].index,result.Actual[k:],color='g',s=30,label='Actual Points')
    # a.fill_between(result.Lower_bound.index,result.iloc[:,2],result.iloc[:,3],color='k',alpha=0.15,label='Confidence Interval')
    # a.set_xlabel('Dates',fontsize=20,color='b')
    # a.set_ylabel('Consumption in GB',fontsize=20,color='b')
    # a.set_title('Disk Storage Consumption',fontsize=20,color='r')
    # a.legend(fontsize=16,loc='upper left')

if __name__ == "__main__":

    # Set up plot to call animate() function periodically

    ani = animation.FuncAnimation(fig, animate, fargs=(result, i), interval=1000)
    plt.show()

# In[35]:


# plt.figure(figsize=(15,10))
# plt.plot(result.Actual,label='Actual')
# plt.plot(result.Predictions,label='Predictions')
# plt.fill_between(result.index,result.Lower_bound,result.Upper_bound,color='k',alpha=0.10,label='Confidence Interval')
# plt.legend(fontsize=16,loc='upper left')

