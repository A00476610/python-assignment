import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import plotly.graph_objects as go
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Model loading is now a cached function to ensure it loads only once per session
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('digit_recognition_model.h5')

model = load_model()


def process_digit_image(input_img):
    if input_img.mode in ('RGBA', 'LA', 'P'):
        opaque_background = Image.new("RGB", input_img.size, "WHITE")
        opaque_background.paste(input_img, mask=input_img.getchannel('A') if input_img.mode == 'RGBA' else input_img)
        modified_img = opaque_background
    else:
        modified_img = input_img.convert('RGB')

    modified_img = ImageOps.grayscale(modified_img)
    modified_img = ImageOps.invert(modified_img)
    resize_ratio = min(28 / modified_img.width, 28 / modified_img.height)
    new_size = (int(modified_img.width * resize_ratio), int(modified_img.height * resize_ratio))
    modified_img = modified_img.resize(new_size, Image.Resampling.LANCZOS)

    white_canvas = Image.new('L', (28, 28), 'white')
    positioning = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    white_canvas.paste(modified_img, positioning)
    image_array = np.array(white_canvas).astype(np.float32) / 255.0
    image_array = image_array.reshape(-1, 28, 28, 1)
    return image_array

def predict(image):
    processed_image = process_digit_image(image)
    predictions = model.predict(processed_image)
    return np.argmax(predictions), np.max(predictions)

def image_classifier_app():
    st.title('Number Recognition App')
    st.write("Upload an image of a handwritten digit, and the model will predict the digit.")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Predict'):
            label, confidence = predict(image)
            st.write(f'Predicted Digit: {label} with confidence {confidence:.2f}')

def coin_comparison_app():
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
            st.error('Failed to fetch coins list because of multiple api requests, please wait few minutes')
            return pd.DataFrame()

    coins_df = get_coins_list()

    # Input for two cryptocurrencies
    coin_name1 = st.text_input("Enter first cryptocurrency name", value="maker").lower().strip()
    coin_name2 = st.text_input("Enter second cryptocurrency name", value="ethereum").lower().strip()

    # Time frame selection
    time_frame = st.selectbox("Select time frame", options=["7", "30", "365", "1825"],
                              format_func=lambda x: f"{int(x) // 365} year" if int(x) > 30 else f"{x} days")

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
            st.error(f'Failed to fetch data for {coin_id} because of multiple api requests, please wait few minutes')
            return pd.DataFrame()

    # Display comparison if both coins exist
    if coin_name1 in coins_df['id'].values and coin_name2 in coins_df['id'].values:
        data1 = get_coin_data(coin_name1, time_frame)
        data2 = get_coin_data(coin_name2, time_frame)

        if not data1.empty and not data2.empty:
            # Price Comparison Plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=data1.index, y=data1['price'], mode='lines', name=f'{coin_name1.capitalize()} Price'))
            fig.add_trace(
                go.Scatter(x=data2.index, y=data2['price'], mode='lines', name=f'{coin_name2.capitalize()} Price'))
            fig.update_layout(title='Price Comparison', yaxis_title='Price in USD', xaxis_title='Date')
            st.plotly_chart(fig)

            # Volume Comparison Plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=data1.index, y=data1['volume'], mode='lines', name=f'{coin_name1.capitalize()} Volume'))
            fig.add_trace(
                go.Scatter(x=data2.index, y=data2['volume'], mode='lines', name=f'{coin_name2.capitalize()} Volume'))
            fig.update_layout(title='Volume Comparison', yaxis_title='Volume', xaxis_title='Date')
            st.plotly_chart(fig)

            # Market Cap Comparison
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data1.index, y=data1['market_cap'], mode='lines',
                                     name=f'{coin_name1.capitalize()} Market Cap'))
            fig.add_trace(go.Scatter(x=data2.index, y=data2['market_cap'], mode='lines',
                                     name=f'{coin_name2.capitalize()} Market Cap'))
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


def stock_details_app():
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
            st.error('Failed to fetch coins list because of multiple api requests, please wait few minutes')
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
            st.error('Failed to fetch coin data because of multiple api requests, please wait few minutes')
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

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an App", ("Coin Comparison", "Stock Details", "Image Classifier"))
    if choice == "Stock Details":
        stock_details_app()
    elif choice == "Coin Comparison":
        coin_comparison_app()
    elif choice == "Image Classifier":
        image_classifier_app()


if __name__ == '__main__':
    main()
