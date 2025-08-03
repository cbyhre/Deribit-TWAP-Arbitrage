import asyncio
import csv
import os
import requests
from datetime import datetime, timezone
from pytz import timezone as tz
from playwright.async_api import async_playwright

STRIKES = [106000, 108000, 110000, 112000, 114000, 116000]
#Adjustable Expiration Date
EXPIRY = "3AUG25"
SYMBOLS = [f"BTC-{EXPIRY}-{strike}-C" for strike in STRIKES]
CSV_FILE = "Summer Research - comparison.csv"
DERIBIT_BOOK_URL = "https://www.deribit.com/api/v2/public/get_order_book"
DERIBIT_INDEX_URL = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
SLEEP_SECONDS = 30
EASTERN = tz("US/Eastern")
STOP_HOUR = 4
STOP_MINUTE = 9

# === Setup CSV ===
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "OKX_BTC_price", "Deribit_BTC_price"] +
            ["Deribit_" + s for s in SYMBOLS] +
            ["OKX_" + str(s) for s in STRIKES]
        )


# === Get Deribit BTC Price ===
def get_deribit_btc_price():
    try:
        r = requests.get(DERIBIT_INDEX_URL)
        return r.json()["result"]["index_price"]
    except:
        return None


# === Async BTC Price from OKX Page ===
async def get_okx_btc_price(page):
    try:
        price_div = await page.query_selector("div.index_last__T0kNQ")
        if price_div:
            price_text = await price_div.inner_text()
            return float(price_text.replace("$", "").replace(",", "").strip())
    except:
        pass
    return None


# === Async Scraper for OKX Call Mark Prices ===
async def scrape_okx_data():
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://www.okx.com/trade-option-chain/btc-usd", wait_until="networkidle")
            await page.wait_for_timeout(8000)

            btc_price = await get_okx_btc_price(page)

            strike_cells = await page.query_selector_all("td.strike")
            mark_price_cells = await page.query_selector_all("td.mark-price")

            strike_mark_map = {}

            for i, strike_td in enumerate(strike_cells):
                text = await strike_td.inner_text()
                clean_strike = text.replace(",", "").strip()

                if clean_strike in map(str, STRIKES):
                    call_index = i * 2  # call = even index
                    if call_index < len(mark_price_cells):
                        price_cell = mark_price_cells[call_index]
                        p_tags = await price_cell.query_selector_all("p")
                        if p_tags:
                            mark_price = await p_tags[0].inner_text()
                            mark_price_clean = float(mark_price.strip().replace(",", ""))  # normalize to 1 BTC
                            strike_mark_map[clean_strike] = round(mark_price_clean, 6)

            await browser.close()
            return btc_price, strike_mark_map

    except Exception as e:
        print(f"OKX scraping error: {e}")
        return None, {}


# === Get Deribit Option Prices ===
def get_deribit_prices(btc_price):
    deribit_prices = []
    for symbol in SYMBOLS:
        try:
            r = requests.get(DERIBIT_BOOK_URL, params={"instrument_name": symbol})
            result = r.json()["result"]
            mark_price = result.get("mark_price", 0)
            usd_price = round(mark_price * btc_price, 2)
            deribit_prices.append(usd_price)
        except:
            deribit_prices.append("N/A")
    return deribit_prices


# === Main Async Loop ===
async def main():
    while True:
        now_utc = datetime.now(timezone.utc)
        now_eastern = now_utc.astimezone(EASTERN)
        if now_eastern.hour == STOP_HOUR and now_eastern.minute >= STOP_MINUTE:
            print("Reached stop time — exiting.")
            break

        try:
            # Get Deribit BTC price first (synchronous)
            deribit_btc = get_deribit_btc_price()

            # Then get OKX data (async)
            okx_btc, okx_map = await scrape_okx_data()

            if None in (deribit_btc, okx_btc):
                raise ValueError("Could not get both BTC prices")

            # Get Deribit option prices using Deribit's BTC price
            deribit_prices = get_deribit_prices(deribit_btc)

            now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")
            row = [now_str, okx_btc, deribit_btc]

            # Add Deribit option prices
            row.extend(deribit_prices)

            # Add OKX option prices
            for strike in STRIKES:
                row.append(okx_map.get(str(strike), "N/A"))

            # Write to CSV
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"[{now_str}] Logged. OKX BTC: ${okx_btc:,.2f} | Deribit BTC: ${deribit_btc:,.2f}")
            await asyncio.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(SLEEP_SECONDS)


# === Run ===
if __name__ == "__main__":
    asyncio.run(main())
