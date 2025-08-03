import time
import requests
import csv
import os
from datetime import datetime
from pytz import timezone
from math import log, sqrt
from scipy.stats import norm
import numpy as np
from collections import deque

EASTERN = timezone("US/Eastern")
STOP_HOUR = 4
STOP_MINUTE = 5
STOP_SECOND = 0

# Deribit API endpoints
DERIBIT_INDEX_URL = "https://www.deribit.com/api/v2/public/get_index_price"
DERIBIT_OPTIONS_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"

TARGET_OPTIONS = [
    "BTC-3AUG25-110000-C",
    "BTC-3AUG25-112000-C",
    "BTC-3AUG25-114000-C"
]

UPDATE_INTERVAL = 5
ROLLING_WINDOW_MINUTES = 30
ROLLING_WINDOW_SIZE = (ROLLING_WINDOW_MINUTES * 60) // UPDATE_INTERVAL

# CSV setup
CSV_FILE = "Summer Research - Arb.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp_et", "deribit_btc_price", "30min_avg_btc", "forward_price"]
        for option in TARGET_OPTIONS:
            header.extend([
                f"{option}_market_price",
                f"{option}_our_price"
            ])
        writer.writerow(header)

# === DATA STRUCTURES ===
price_history = deque(maxlen=ROLLING_WINDOW_SIZE)


# === BLACK-SCHOLES CALCULATION ===
def bs_deribit_option_price(option_type, F, K, T, sigma):
    d1 = (log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'call':
        return norm.cdf(d1) - (K / F) * norm.cdf(d2)
    elif option_type == 'put':
        return (K / F) * norm.cdf(-d2) - norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# === API FUNCTIONS ===
def fetch_deribit_btc_price():
    try:
        response = requests.get(DERIBIT_INDEX_URL, params={"index_name": "btc_usd"})
        response.raise_for_status()
        return float(response.json()["result"]["index_price"])
    except Exception as e:
        print(f"Error fetching BTC price: {e}")
        return None


def fetch_deribit_options():
    try:
        response = requests.get(DERIBIT_OPTIONS_URL, params={
            "currency": "BTC",
            "kind": "option"
        })
        response.raise_for_status()
        return response.json().get("result", [])
    except Exception as e:
        print(f"Error fetching options: {e}")
        return None


# === TIME CALCULATIONS ===
def calculate_hours_until_expiration(expiration_date_str):
    try:
        date_part = expiration_date_str.split('-')[1]
        expiration_date = datetime.strptime(date_part, "%d%b%y")
        expiration_datetime = expiration_date.replace(hour=8, minute=0, second=0, tzinfo=timezone('UTC'))
        now = datetime.now(timezone('UTC'))
        return max(0, (expiration_datetime - now).total_seconds() / 3600)
    except Exception as e:
        print(f"Error calculating expiration: {e}")
        return None


# === MAIN LOOP ===
def main():
    print("Starting BTC Option Monitoring Algorithm")
    print(f"• Refresh interval: {UPDATE_INTERVAL} seconds")
    print(f"• Using {ROLLING_WINDOW_MINUTES}-minute rolling average for forward price")
    print(f"• Monitoring options: {', '.join(TARGET_OPTIONS)}")
    print(f"• Will automatically stop at {STOP_HOUR}:{STOP_MINUTE:02d}:{STOP_SECOND:02d} AM Eastern\n")

    while True:
        now = datetime.now(EASTERN)

        # Check if we've reached stop time
        if now.hour == STOP_HOUR and now.minute >= STOP_MINUTE:
            print(f"Reached stop time: {now.strftime('%H:%M:%S')} Eastern. Exiting.")
            break

        # Fetch current BTC price
        current_price = fetch_deribit_btc_price()
        if current_price is None:
            time.sleep(UPDATE_INTERVAL)
            continue

        # Update price history and calculate averages
        price_history.append(current_price)
        rolling_avg = np.mean(price_history) if price_history else current_price
        rolling_avg = round(rolling_avg, 10)
        forward_price = round(rolling_avg + 0, 10)

        # Fetch options data
        options_data = fetch_deribit_options()
        if options_data is None:
            time.sleep(UPDATE_INTERVAL)
            continue

        # Prepare data row for CSV
        timestamp_et = now.strftime('%Y-%m-%d %H:%M:%S')
        row_data = [
            timestamp_et,
            round(current_price, 10),
            rolling_avg,
            forward_price
        ]
        option_prices = {}

        # Process each target option
        for option in options_data:
            instrument_name = option.get("instrument_name")
            if instrument_name not in TARGET_OPTIONS:
                continue

            # Extract option details
            parts = instrument_name.split('-')
            option_type = 'call' if parts[3] == 'C' else 'put'
            strike = float(parts[2])

            # Get market data
            mark_price = option.get("mark_price")
            mark_price = round(float(mark_price), 10) if mark_price is not None else None
            iv = option.get("mark_iv", 0)
            iv = round(float(iv) / 100, 10) if iv else 0  # Convert to decimal

            # Calculate time to expiration
            hours_to_exp = calculate_hours_until_expiration(instrument_name)
            years_to_exp = round(hours_to_exp / (24 * 365), 10) if hours_to_exp else 0

            # Black-Scholes calculation
            our_price = None
            if years_to_exp > 0 and iv > 0:
                our_price = bs_deribit_option_price(option_type, forward_price, strike, years_to_exp, iv)
                our_price = round(our_price, 10) if our_price is not None else None

            # Store prices
            option_prices[instrument_name] = {
                "market": mark_price,
                "our": our_price
            }

        # Build CSV row
        for option in TARGET_OPTIONS:
            prices = option_prices.get(option, {"market": None, "our": None})
            row_data.extend([prices["market"], prices["our"]])

        # Write to CSV
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # Print status
        print(f"\n[{timestamp_et}] Updated:")
        print(f"BTC Price: {current_price} | 30-min Avg: {rolling_avg} | Forward: {forward_price}")
        for option in TARGET_OPTIONS:
            prices = option_prices.get(option, {"market": None, "our": None})
            print(f"{option}: Market={prices['market']} | Our={prices['our']}")

        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()
