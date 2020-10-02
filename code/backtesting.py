#
# Event-Based Backtesting
# --Base Class (1)
#
# (c) Dr. Yves J. Hilpisch
# Artificial Intelligence in Finance
#


class BacktestingBase:
    def __init__(self, env, model, amount, ptc, ftc, verbose=False):
        self.env = env  # <1>
        self.model = model  # <2>
        self.initial_amount = amount  # <3>
        self.current_balance = amount  # <3>
        self.ptc = ptc   # <4>
        self.ftc = ftc   # <5>
        self.verbose = verbose  # <6>
        self.units = 0  # <7>
        self.trades = 0  # <8>

    def get_date_price(self, bar):
        ''' Returns date and price for a given bar.
        '''
        date = str(self.env.data.index[bar])[:10]  # <9>
        price = self.env.data[self.env.symbol].iloc[bar]  # <10>
        return date, price

    def print_balance(self, bar):
        ''' Prints the current cash balance for a given bar.
        '''
        date, price = self.get_date_price(bar)
        print(f'{date} | current balance = {self.current_balance:.2f}')  # <11>

    def calculate_net_wealth(self, price):
        return self.current_balance + self.units * price  # <12>

    def print_net_wealth(self, bar):
        ''' Prints the net wealth for a given bar
            (cash + position).
        '''
        date, price = self.get_date_price(bar)
        net_wealth = self.calculate_net_wealth(price)
        print(f'{date} | net wealth = {net_wealth:.2f}')  # <13>

    def place_buy_order(self, bar, amount=None, units=None):
        ''' Places a buy order for a given bar and for
            a given amount or number of units.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)  # <14>
            # units = amount / price  # <14>
        self.current_balance -= (1 + self.ptc) * \
            units * price + self.ftc  # <15>
        self.units += units  # <16>
        self.trades += 1  # <17>
        if self.verbose:
            print(f'{date} | buy {units} units for {price:.4f}')
            self.print_balance(bar)

    def place_sell_order(self, bar, amount=None, units=None):
        ''' Places a sell order for a given bar and for
            a given amount or number of units.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)  # <14>
            # units = amount / price  # <14>
        self.current_balance += (1 - self.ptc) * \
            units * price - self.ftc  # <15>
        self.units -= units  # <16>
        self.trades += 1  # <17>
        if self.verbose:
            print(f'{date} | sell {units} units for {price:.4f}')
            self.print_balance(bar)

    def close_out(self, bar):
        ''' Closes out any open position at a given bar.
        '''
        date, price = self.get_date_price(bar)
        print(50 * '=')
        print(f'{date} | *** CLOSING OUT ***')
        if self.units < 0:
            self.place_buy_order(bar, units=-self.units)  # <18>
        else:
            self.place_sell_order(bar, units=self.units)  # <19>
        if not self.verbose:
            print(f'{date} | current balance = {self.current_balance:.2f}')
        perf = (self.current_balance / self.initial_amount - 1) * 100  # <20>
        print(f'{date} | net performance [%] = {perf:.4f}')
        print(f'{date} | number of trades [#] = {self.trades}')
        print(50 * '=')
