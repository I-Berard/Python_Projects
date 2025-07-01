import ccxt.async_support as ccxt
import asyncio
import time
import random
from collections import defaultdict
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import requests
import statistics

# Add your Telegram bot token and chat ID
TELEGRAM_BOT_TOKEN = ''
TELEGRAM_CHAT_ID = ''

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'  # Optional: use Markdown for formatting
    }
    requests.post(url, json=payload)

def cmc(starting_rank, number_of_tokens):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    parameters = {
       'start':'1',
       'limit': str(number_of_tokens),
       'convert':'USD'
    }
    headers = {
       'Accepts': 'application/json',
       'X-CMC_PRO_API_KEY': '5fd80b86-e0d2-42b4-9afc-1653c232f672',
    }
    session = Session()
    session.headers.update(headers)
    
    try:
      response = session.get(url, params=parameters)
      data = json.loads(response.text)
      #print(data)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
      print(e)
    
    # exclude stablecoins
    cmc_coin_list = []
    for coin in data['data']:
        if starting_rank <= coin['cmc_rank'] < number_of_tokens and 'stablecoin' not in coin['tags']:
            cmc_coin_list.append(coin['symbol'])
    return cmc_coin_list
coins = cmc(100, 1500)

async def fetch_all_fees_and_statuses(exchanges, symbols):
    withdrawal_fees = {}
    trading_fees = defaultdict(lambda: defaultdict(dict))
    coin_statuses = defaultdict(lambda: defaultdict(dict))

    async def fetch_fees_and_statuses_for_exchange(exchange_id, exchange):
        try:
            currencies = await exchange.fetch_currencies()
            withdrawal_fees[exchange_id] = {
                coin: currencies[coin]['fee'] if coin in currencies and 'fee' in currencies[coin] else None
                for coin in coins
            }
            
            for coin in coins:
                if coin in currencies:
                    coin_statuses[exchange_id][coin] = {
                        'withdrawable': currencies[coin].get('withdraw', False),
                        'depositable': currencies[coin].get('deposit', False)
                    }
                else:
                    coin_statuses[exchange_id][coin] = {
                        'withdrawable': False,
                        'depositable': False
                    }
            
            for symbol in symbols:
                try:
                    fees = await exchange.fetch_trading_fee(symbol)
                    trading_fees[exchange_id][symbol] = fees
                except Exception as e:
                    print(f"Error fetching trading fees for {exchange_id} {symbol}: {str(e)}")
                    trading_fees[exchange_id][symbol] = {'maker': 0.001, 'taker': 0.001}
        except Exception as e:
            print(f"Error fetching fees and statuses for {exchange_id}: {str(e)}")
            withdrawal_fees[exchange_id] = {coin: None for coin in coins}
            coin_statuses[exchange_id] = {coin: {'withdrawable': False, 'depositable': False} for coin in coins}

    await asyncio.gather(*[fetch_fees_and_statuses_for_exchange(exchange_id, exchange) for exchange_id, exchange in exchanges.items()])
    return withdrawal_fees, dict(trading_fees), dict(coin_statuses)

async def fetch_order_book_with_retry(exchange, symbol, max_retries=3, initial_delay=1):
    for attempt in range(max_retries):
        try:
            return await exchange.fetch_order_book(symbol)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching order book for {exchange.id} {symbol}: {str(e)}")
                return None
            delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
            # print(f"Retrying {exchange.id} {symbol} in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

async def fetch_order_books(exchanges, symbols):
    order_books = defaultdict(dict)
    latencies = defaultdict(list)

    async def fetch_order_book_for_symbol(exchange_id, exchange, symbol):
        try:
            start_time = time.time()
            order_book = await fetch_order_book_with_retry(exchange, symbol)
            if order_book is None:
                return
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies[exchange_id].append(latency)
            order_books[symbol][exchange_id] = order_book
        except Exception as e:
            print(f"Error fetching order book for {exchange_id} {symbol}: {str(e)}")

    await asyncio.gather(*[fetch_order_book_for_symbol(exchange_id, exchange, symbol) 
                           for exchange_id, exchange in exchanges.items() 
                           for symbol in symbols])
    return dict(order_books), dict(latencies)

def estimate_slippage(order_book, amount, side='buy'):
    """
    Estimate slippage based on the order book depth.
    """
    total_amount = 0
    weighted_price = 0
    target_amount = amount

    if side == 'buy':
        for price, size in order_book['asks']:
            if total_amount < target_amount:
                fill_amount = min(size, target_amount - total_amount)
                total_amount += fill_amount
                weighted_price += price * fill_amount
            else:
                break
    else:  # sell
        for price, size in order_book['bids']:
            if total_amount < target_amount:
                fill_amount = min(size, target_amount - total_amount)
                total_amount += fill_amount
                weighted_price += price * fill_amount
            else:
                break

    if total_amount > 0:
        average_price = weighted_price / total_amount
        if side == 'buy':
            slippage = (average_price - order_book['asks'][0][0]) / order_book['asks'][0][0]
        else:
            slippage = (order_book['bids'][0][0] - average_price) / order_book['bids'][0][0]
        return slippage
    return 0

def calculate_profit_with_fees_and_slippage(buy_exchange, sell_exchange, buy_price, sell_price, amount, withdrawal_fees, trading_fees, symbol, buy_order_book, sell_order_book):
    try:
        # Estimate slippage
        buy_slippage = estimate_slippage(buy_order_book, amount, 'buy')
        sell_slippage = estimate_slippage(sell_order_book, amount, 'sell')

        # Adjust prices for slippage
        adjusted_buy_price = buy_price * (1 + buy_slippage)
        adjusted_sell_price = sell_price * (1 - sell_slippage)

        buy_fee_rate = trading_fees.get(buy_exchange, {}).get(symbol, {}).get('taker', 0.001)
        if buy_fee_rate is None:
            print(f"Warning: Using default taker fee for {buy_exchange} {symbol}")
            buy_fee_rate = 0.001
        buy_fee = amount * buy_fee_rate
        coins_bought = (amount - buy_fee) / adjusted_buy_price

        coin = symbol.split('/')[0]
        withdrawal_fee = withdrawal_fees.get(buy_exchange, {}).get(coin, 0) or 0
        coins_after_withdrawal = coins_bought - withdrawal_fee

        sell_fee_rate = trading_fees.get(sell_exchange, {}).get(symbol, {}).get('maker', 0.001)
        if sell_fee_rate is None:
            print(f"Warning: Using default maker fee for {sell_exchange} {symbol}")
            sell_fee_rate = 0.001
        sale_amount_before_fee = coins_after_withdrawal * adjusted_sell_price
        sell_fee = sale_amount_before_fee * sell_fee_rate
        sale_amount = sale_amount_before_fee - sell_fee

        profit = sale_amount - amount
        profit_percentage = (profit / amount) * 100
        return profit, profit_percentage, buy_slippage, sell_slippage
    except Exception as e:
        print(f"Error in calculate_profit_with_fees_and_slippage: {str(e)}")
        print(f"Inputs: buy_exchange={buy_exchange}, sell_exchange={sell_exchange}, buy_price={buy_price}, sell_price={sell_price}, amount={amount}")
        print(f"Symbol: {symbol}")
        print(f"Trading fees: {trading_fees.get(buy_exchange, {}).get(symbol, 'N/A')} (buy), {trading_fees.get(sell_exchange, {}).get(symbol, 'N/A')} (sell)")
        return None, None, None, None

def find_arbitrage_opportunities(all_order_books, initial_capital, withdrawal_fees, trading_fees, coin_statuses, min_profit_percentage):
    opportunities = []
    for symbol, order_books in all_order_books.items():
        exchange_names = list(order_books.keys())
        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                buy_exchange = exchange_names[i]
                sell_exchange = exchange_names[j]
                
                if not order_books[buy_exchange]['asks'] or not order_books[sell_exchange]['bids']:
                    continue

                coin = symbol.split('/')[0]
                
                buy_price = order_books[buy_exchange]['asks'][0][0]
                sell_price = order_books[sell_exchange]['bids'][0][0]
                
                profit, profit_percentage, buy_slippage, sell_slippage = calculate_profit_with_fees_and_slippage(
                    buy_exchange, sell_exchange, buy_price, sell_price, 
                    initial_capital, withdrawal_fees, trading_fees, symbol,
                    order_books[buy_exchange], order_books[sell_exchange]
                )
                
                if profit_percentage > min_profit_percentage:
                    opportunities.append({
                        'symbol': symbol,
                        'buy_exchange': buy_exchange,
                        'sell_exchange': sell_exchange,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'profit': profit,
                        'profit_percentage': profit_percentage,
                        'buy_withdrawable': coin_statuses[buy_exchange][coin]['withdrawable'],
                        'sell_depositable': coin_statuses[sell_exchange][coin]['depositable'],
                        'buy_slippage': buy_slippage,
                        'sell_slippage': sell_slippage
                    })
    
    return opportunities

async def main():
    exchanges = {
            'binance': ccxt.binance({
                'apiKey': '',
                'secret': '',
            }),
            'huobi': ccxt.huobi({
                'apiKey': '',
                'secret': ''
            }),
            'kucoin': ccxt.kucoin({
                'apiKey': '',
                'secret': '',
                'password': ''
            }),
            'bitget': ccxt.bitget({
                'apiKey': '',
                'secret': '',
                'password': ''
            }),
            'bigone': ccxt.bigone({
                'apiKey': '',
                'secret': '',
            }),
            'whitebit': ccxt.whitebit({
                'apiKey': '',
                'secret': '',
            }),
        
    }
    
    coins = cmc(100, 1500)
    symbols = [f"{coin}/USDT" for coin in coins]
    initial_capital = 5000  # $5000 initial capital
    min_profit_percentage = 0.1  # 0.1% minimum profit
    
    # Fetch all fees and statuses once at the start
    withdrawal_fees, trading_fees, coin_statuses = await fetch_all_fees_and_statuses(exchanges, symbols)
    
    try:
        while True:
            start_time = time.time()
            all_order_books, latencies = await fetch_order_books(exchanges, symbols)
            fetch_time = time.time() - start_time

            opportunities = find_arbitrage_opportunities(all_order_books, initial_capital, withdrawal_fees, trading_fees, coin_statuses, min_profit_percentage)
            
            if opportunities:
                print("Arbitrage opportunities found:")
                for opportunity in opportunities:
                    if opportunity['buy_withdrawable'] and opportunity['sell_depositable']:
                        message = (f"Arbitrage Opportunity:\n"
                                   f"Symbol: {opportunity['symbol']}\n"
                                   f"Buy from {opportunity['buy_exchange']} at {opportunity['buy_price']}\n"
                                   f"Sell on {opportunity['sell_exchange']} at {opportunity['sell_price']}\n"
                                   f"Potential profit: ${opportunity['profit']:.2f} ({opportunity['profit_percentage']:.2f}%)\n"
                                   f"Estimated buy slippage: {opportunity['buy_slippage']:.2%}\n"
                                   f"Estimated sell slippage: {opportunity['sell_slippage']:.2%}")
                        
                        # Send the message to Telegram
                        send_telegram_message(message)
    
                        print(f"Symbol: {opportunity['symbol']}")
                        print(f"Buy from {opportunity['buy_exchange']} at {opportunity['buy_price']}")
                        print(f"Sell on {opportunity['sell_exchange']} at {opportunity['sell_price']}")
                        print(f"Potential profit: ${opportunity['profit']:.2f} ({opportunity['profit_percentage']:.2f}%)")
                        print(f"Estimated buy slippage: {opportunity['buy_slippage']:.2%}")
                        print(f"Estimated sell slippage: {opportunity['sell_slippage']:.2%}")
                        print("---")
                    else:
                        continue

            await asyncio.sleep(10)  # Wait for 10 seconds before checking again
    finally:
        # Close all exchange instances
        await asyncio.gather(*[exchange.close() for exchange in exchanges.values()])

if __name__ == "__main__":
    asyncio.run(main())