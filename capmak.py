import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import pandas_datareader as web
#import capm_func  # importing plotly function python file

st.set_page_config(page_title="CAPM" , page_icon="chart_with_upward_trend",layout = "wide")

st.title("CAPITAL ASSET PRICING MODEL")

# INPUT FROM USER
col1 , col2 = st.columns([1,1])
with col1:
    stocks_list = st.multiselect("Choose MINIMUM 4 stock" , ('TSLA' , 'NFLX', 'AMZN', 'GOOGL','AAPL', 'NVDIA'),['AMZN','GOOGL','AAPL'])
with col2:
    year = st.number_input("Number of years",1,10)
# downloading data for sp500




# function to plot interactive plotly chart


# function to calculate daily returns


try:

    #capm_func
    import numpy as np
    import plotly.express as px
    def interactive_plot(df):
    fig = px.line()
    for i in df.columns[1:]:
        fig.add_scatter(x = df['Date'], y = df[i], name = i)
    fig.update_layout(width = 450,margin = dict(l = 20, r = 20, t = 50 ,b = 20) , legend = dict(orientation = 'h',yanchor = 'bottom',
         y = 1.02 , xanchor = 'right' , x = 1,)) 
    return fig

# function to normalize the prices based on the initial price
    def normalize(df_2):
        df = df_2.copy()
    for i in df.columns[1:]:
        df[i] = df[i]/df[i][0]
    return df

# function to calculate daily returns
    def daily_return(df):
        df_daily_return = df.copy()
    for  i in df.columns[1:]:
        for j in range(1,len(df)):
            df_daily_return[i][j] = ((df[i][j]-df[i][j-1])/df[i][j-1])*100
        df_daily_return[i][0] = 0
    return df_daily_return

# function to calculate beta
    def calculate_beta(stocks_daily_return , stock):
        rm = stocks_daily_return['sp500'].mean()*252

    b ,a = np.polyfit(stocks_daily_return['sp500'],stocks_daily_return[stock],1)
    return b,a






    end = datetime.date.today()
    start = datetime.date(datetime.date.today().year-year, datetime.date.today().month , datetime.date.today().day)
    SP500 = web.DataReader(['sp500'],'fred',start,end)
    #print(sp500.tail())
    stocks_df = pd.DataFrame()
    for stock in stocks_list:
        data = yf.download(stock, period=f'{year}y')
        stocks_df[f'{stock}'] = data['Close']


    stocks_df.reset_index(inplace = True)
    SP500.reset_index(inplace = True)
    SP500.columns = ['Date','sp500']
    stocks_df['Date'] = stocks_df['Date'].astype('datetime64[ns]')
    stocks_df['Date'] = stocks_df['Date'].apply(lambda x:str(x)[:10]) 
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    stocks_df = pd.merge(stocks_df,SP500,on='Date',how='inner')
    #print(stocks_df)
    col1 , col2 = st.columns([1,1])
    with col1:
        st.markdown('### Dataframe head')
        st.dataframe(stocks_df.head(), use_container_width= True)
    with col2:
        st.markdown('### Dataframe tail')
        st.dataframe(stocks_df.tail(), use_container_width= True)
    col1 , col2 = st.columns([1,1])
    with col1:
        st.markdown('### Price Of All The Stocks')
        st.plotly_chart(capm_func.interactive_plot(stocks_df))
    with col2:
        st.markdown('### Price Of All The Stocks (After Normalizing)')
        st.plotly_chart(capm_func.interactive_plot(capm_func.normalize(stocks_df)))

    stocks_daily_return = capm_func.daily_return(stocks_df)
    print(stocks_daily_return.head())

    beta = {}
    alpha = {}

    for i in stocks_daily_return.columns:
        if  i != 'Date' and i != 'sp500':
            b ,a = capm_func.calculate_beta(stocks_daily_return,i)

            beta[i] = b
            alpha[i] = a
    print(beta ,alpha) 

    beta_df = pd.DataFrame(columns=['Stock','Beta value'])
    beta_df['Stock'] = beta.keys()
    beta_df['Beta Value'] = [str(round(i,2)) for i in beta.values()]
    with col1:
        st.markdown('### Calculated Beta Value')
        st.dataframe(beta_df, use_container_width=True)
    rf = 0
    rm = stocks_daily_return['sp500'].mean()*252

    return_df = pd.DataFrame()
    return_value = []
    for stock , value in beta.items():
        return_value.append(str(round(rf + (value*(rm - rf)),2)))

    return_df['Stock'] = stocks_list

    return_df['Return Value'] = return_value

    with col2 :
        st.markdown('### Calculated Return using CAPM')

        st.dataframe(return_df, use_container_width=True)

    st.write(" ")
    st.write("")
    st.write(" ")
    st.write("-𝘮𝘢𝘥𝘦 𝘣𝘺 𝘈𝘬𝘢𝘴𝘩 𝘎𝘢𝘶𝘵𝘢𝘮")

except:
    st.write("TRY to change with patience !!")
