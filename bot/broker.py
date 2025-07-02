import ccxt
import time

class BinanceDemo:
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.exchange.set_sandbox_mode(True)
        self.exchange.load_markets()
        self.exchange.check_required_credentials()
        self.exchange.load_time_difference()

    def place_order(self, symbol, side, amount, price=None, order_type='market'):
        try:
            if order_type == 'market':
                print(f"📈 Enviando orden de mercado: {side.upper()} {amount} {symbol}")
                return self.exchange.create_market_order(symbol, side, amount)
            elif order_type == 'limit' and price:
                print(f"📉 Enviando orden límite: {side.upper()} {amount} {symbol} @ {price}")
                return self.exchange.create_limit_order(symbol, side, amount, price)
            else:
                print("⚠️ Tipo de orden inválido o precio faltante.")
                return None
        except Exception as e:
            print(f"❌ Error al ejecutar orden: {e}")
            return None

    def get_balance(self, asset='USDT'):
        try:
            balance = self.exchange.fetch_balance()
            return balance['total'].get(asset, 0)
        except Exception as e:
            print(f"❌ Error al obtener balance: {e}")
            return None

    def get_ticker_data(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            volume = ticker.get('baseVolume', 0)

            if bid is None or ask is None:
                print(f"⚠️ Datos de ticker incompletos para {symbol}. Bid o Ask no disponibles.")
                return {
                    'spread': None,
                    'volume': volume,
                    'safe': False
                }

            spread = ask - bid
            safe = spread < 0.05  # umbral ajustable

            return {
                'spread': spread,
                'volume': volume,
                'safe': safe
            }

        except Exception as e:
            print(f"❌ Error al obtener ticker: {e}")
            return {
                'spread': None,
                'volume': None,
                'safe': False
            }


class PaperBroker:
    def __init__(self, balance=10000, leverage=1):
        self.balance = {'USDT': balance}
        self.leverage = leverage
        self.orders = []
        self.positions = {}
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        while True:
            try:
                self.exchange.load_markets()
                print("✅ Mercados de Binance cargados correctamente.")
                break
            except Exception as e:
                print(f"❌ Error al cargar mercados de Binance: {e}")
                print("🔁 Reintentando en 5 segundos...")
                time.sleep(5)
        
        
        self.exchange.load_markets()
        
        # 🔧 Histórico de balances inicializado
        self.balance_history = [self.balance['USDT']]

    def place_order(self, symbol, side, amount, price=None, order_type='market'):
        ticker = self.fetch_ticker(symbol)
        fill_price = ticker['last'] if order_type == 'market' else price

        position = self.positions.get(symbol, {'side': None, 'qty': 0, 'entry': 0})

        if side == 'buy':
            cost = amount * fill_price
            self.balance['USDT'] -= cost
        else:
            self.balance['USDT'] += amount * fill_price

        self.positions[symbol] = {
            'side': side,
            'qty': amount,
            'entry': fill_price
        }

        self.orders.append({
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': fill_price,
            'timestamp': self.exchange.milliseconds()
        })

        # 🔧 Actualizamos historial de balance después de cada operación
        self.balance_history.append(self.balance['USDT'])

        print(f"📈 [PAPER] Orden simulada: {side.upper()} {amount} {symbol} @ {fill_price}")
        return self.orders[-1]

    def get_balance(self, asset='USDT'):
        return self.balance.get(asset, 0)

    def get_drawdown(self):
        peak = max(self.balance_history)
        current = self.balance_history[-1]
        drawdown = (peak - current) / peak
        return drawdown

    def fetch_ohlcv(self, symbol, timeframe, limit):
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ticker(self, symbol):
        try:
            ohlcv = self.fetch_ohlcv(symbol, '5m', limit=1)[-1]
            close = ohlcv[4]
            simulated_bid = close * 0.9995
            simulated_ask = close * 1.0005
            return {
                'bid': simulated_bid,
                'ask': simulated_ask,
                'last': close,
                'baseVolume': ohlcv[5] if len(ohlcv) > 5 else 0
            }
        except Exception as e:
            print(f"❌ Error simulando ticker en PaperBroker: {e}")
            return {
                'bid': None,
                'ask': None,
                'last': None,
                'baseVolume': 0
            }

    def get_ticker_data(self, symbol):
        try:
            ticker = self.fetch_ticker(symbol)
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            volume = ticker.get('baseVolume', 0)

            if bid is None or ask is None:
                print(f"⚠️ [PAPER] Datos incompletos para {symbol}.")
                return {
                    'spread': None,
                    'volume': volume,
                    'safe': False
                }

            spread = ask - bid
            safe = spread < 0.05  # umbral ajustable

            return {
                'spread': spread,
                'volume': volume,
                'safe': safe
            }

        except Exception as e:
            print(f"❌ Error al obtener ticker en PaperBroker: {e}")
            return {
                'spread': None,
                'volume': None,
                'safe': False
            }

    def fetch_positions(self, symbols):
        pos = []
        for symbol in symbols:
            p = self.positions.get(symbol)
            if p:
                pos.append({
                    'symbol': symbol,
                    'contracts': p['qty'],
                    'side': p['side'],
                    'entry': p['entry']
                })
        return pos
