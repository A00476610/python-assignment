import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from datetime import datetime

# Title of the application
st.title('Stock Details App')

# CoinGecko API URL
COINS_LIST_URL = 'https://api.coingecko.com/api/v3/coins/list'
COIN_MARKET_CHART_URL = 'https://api.coingecko.com/api/v3/coins/{id}/market_chart'
API_KEY = 'CG-TZe1VrxroETPEuLXxfBFTtEU'  # Headers including the API key
headers = {
    'accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# Fetching the list of all available coins
@st.cache_data
def get_coins_list():
    response = requests.get(COINS_LIST_URL, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error('Failed to fetch coins list')
        return pd.DataFrame()

coins_df = get_coins_list()

# Text input for cryptocurrency selection with default value "bitcoin"
coin_name = st.text_input("Enter a cryptocurrency name", value="bitcoin").lower().strip()
if coin_name:
    # Check if the entered cryptocurrency is valid
    if coin_name in coins_df['id'].values:
        coin_id = coin_name
    else:
        st.error(f"Cryptocurrency {coin_name} does not exist.")
        st.stop()

# Fetching historical market data for the selected coin
def get_coin_data(coin_id, days):
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    response = requests.get(COIN_MARKET_CHART_URL.format(id=coin_id), params=params, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json()['prices'], columns=['timestamp', 'price'])
    else:
        st.error('Failed to fetch coin data')
        return pd.DataFrame()

def format_price(price):
    return f"${price:.6f}" if price < 0.01 else f"${price:.2f}"

# Displaying the data if a coin is selected
if coin_id:
    data = get_coin_data(coin_id, 365)
    if not data.empty:
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('date', inplace=True)
        data['daily_change_pct'] = data['price'].pct_change() * 100  # Calculate daily percentage change

        # Plotting the price data and daily changes with Matplotlib
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(data.index, data['price'], label='Price in USD')
        ax1.set_ylabel('Price in USD')
        ax1.set_title(f'Price Trend for {coin_id.capitalize()} Over the Last Year')
        ax1.legend()

        ax2.bar(data.index, data['daily_change_pct'], color='orange', label='Daily Change (%)')
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.set_ylabel('Daily Change (%)')
        ax2.set_xlabel('Date')
        ax2.legend()

        ax1.xaxis.set_major_locator(MonthLocator(interval=2))
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

        st.pyplot(fig)

        # Calculating and displaying max and min prices
        max_price = data['price'].max()
        min_price = data['price'].min()
        max_day = data[data['price'] == max_price].index[0]
        min_day = data[data['price'] == min_price].index[0]

        # UI enhancements for max and min display
        col1, col2 = st.columns(2)
        col1.metric("Maximum Price", format_price(max_price), f"On {max_day.strftime('%A (%Y-%m-%d')})")
        col2.metric("Minimum Price", format_price(min_price),
                   delta=f"On {min_day.strftime('%A (%Y-%m-%d')})", delta_color="inverse")

