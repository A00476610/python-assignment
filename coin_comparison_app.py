import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# Title of the application
st.title('Coin Comparison App')

# CoinGecko API URL and headers
COINS_LIST_URL = 'https://api.coingecko.com/api/v3/coins/list'
COIN_MARKET_CHART_URL = 'https://api.coingecko.com/api/v3/coins/{id}/market_chart'
API_KEY = 'CG-TZe1VrxroETPEuLXxfBFTtEU'  # Replace with your actual API key
headers = {
    'accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# Cache coin list to reduce API calls
@st.cache_data
def get_coins_list():
    response = requests.get(COINS_LIST_URL, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error('Failed to fetch coins list')
        return pd.DataFrame()

coins_df = get_coins_list()

# Input for two cryptocurrencies
coin_name1 = st.text_input("Enter first cryptocurrency name", value="bitcoin").lower().strip()
coin_name2 = st.text_input("Enter second cryptocurrency name", value="ethereum").lower().strip()

# Time frame selection
time_frame = st.selectbox("Select time frame", options=["7", "30", "365", "1825"], format_func=lambda x: f"{int(x)//365} year" if int(x) > 30 else f"{x} days")

# Helper function to fetch coin data
def get_coin_data(coin_id, days):
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'  # Ensures we receive daily data for volume and market caps
    }
    response = requests.get(COIN_MARKET_CHART_URL.format(id=coin_id), params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['volume'] = pd.DataFrame(data['total_volumes'])[1]
        df['market_cap'] = pd.DataFrame(data['market_caps'])[1]
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        return df
    else:
        st.error(f'Failed to fetch data for {coin_id}')
        return pd.DataFrame()

# Display comparison if both coins exist
if coin_name1 in coins_df['id'].values and coin_name2 in coins_df['id'].values:
    data1 = get_coin_data(coin_name1, time_frame)
    data2 = get_coin_data(coin_name2, time_frame)

    if not data1.empty and not data2.empty:
        # Price Comparison Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data1.index, y=data1['price'], mode='lines', name=f'{coin_name1.capitalize()} Price'))
        fig.add_trace(go.Scatter(x=data2.index, y=data2['price'], mode='lines', name=f'{coin_name2.capitalize()} Price'))
        fig.update_layout(title='Price Comparison', yaxis_title='Price in USD', xaxis_title='Date')
        st.plotly_chart(fig)

        # Volume Comparison Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data1.index, y=data1['volume'], mode='lines', name=f'{coin_name1.capitalize()} Volume'))
        fig.add_trace(go.Scatter(x=data2.index, y=data2['volume'], mode='lines', name=f'{coin_name2.capitalize()} Volume'))
        fig.update_layout(title='Volume Comparison', yaxis_title='Volume', xaxis_title='Date')
        st.plotly_chart(fig)

        # Market Cap Comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data1.index, y=data1['market_cap'], mode='lines', name=f'{coin_name1.capitalize()} Market Cap'))
        fig.add_trace(go.Scatter(x=data2.index, y=data2['market_cap'], mode='lines', name=f'{coin_name2.capitalize()} Market Cap'))
        fig.update_layout(title='Market Cap Comparison', yaxis_title='Market Cap', xaxis_title='Date')
        st.plotly_chart(fig)

        # Display max/min price with dates for coin 1
        max_price1 = data1['price'].max()
        min_price1 = data1['price'].min()
        max_day1 = data1[data1['price'] == max_price1].index[0]
        min_day1 = data1[data1['price'] == min_price1].index[0]

        # Display max/min price with dates for coin 2
        max_price2 = data2['price'].max()
        min_price2 = data2['price'].min()
        max_day2 = data2[data2['price'] == max_price2].index[0]
        min_day2 = data2[data2['price'] == min_price2].index[0]

        # Show the max/min prices with days in the UI
        col1, col2 = st.columns(2)
        col1.metric(f"{coin_name1.capitalize()} Maximum Price", f"${max_price1:.2f}",
                    f"On {max_day1.strftime('%A, %B %d, %Y')}")
        col2.metric(f"{coin_name2.capitalize()} Maximum Price", f"${max_price2:.2f}",
                    f"On {max_day2.strftime('%A, %B %d, %Y')}")

        col1.metric(f"{coin_name1.capitalize()} Minimum Price", f"${min_price1:.2f}",
                    f"On {min_day1.strftime('%A, %B %d, %Y')}")
        col2.metric(f"{coin_name2.capitalize()} Minimum Price", f"${min_price2:.2f}",
                    f"On {min_day2.strftime('%A, %B %d, %Y')}")

