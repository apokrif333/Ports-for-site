from datetime import datetime
from libs import trading_lib as tl

import pandas as pd
import numpy as np
import os

FOLDER_WITH_DATA = 'historical_data'
FOLDER_TO_SAVE = 'saved_data'
FOLDER_WITH_IMG = 'saved_data/img/'


class BasePortfolio:

    # Field list
    __slots__ = [
        'all_tickers_data',
        'strategy_data',
        'div_storage',
        'trading_days',
        'capital_not_placed',
        'max_position',
        'current_port',

        'portfolios',
        'rebalance',
        'rebalance_at',
        'forsed_rebalance',
        'div_tax',
        'commision',
        'acc_maintenance',
        'balance_start',
        'date_start',
        'date_end',
        'bench',
        'withd_depo',
        'value_withd_depo',
    ]

    def __init__(self,
                 portfolios: dict,
                 rebalance: str = 'monthly',
                 trade_rebalance_at: str = 'close',
                 forsed_rebalance: bool = False,
                 div_tax: float = 0.9,
                 commision: float = 0.0055,
                 acc_maintenance: int = 10,
                 balance_start: int = 10_000,
                 date_start: datetime = datetime(2008, 1, 1),
                 date_end: datetime = datetime.now(),
                 benchmark: str = 'QQQ',
                 withdraw_or_depo: bool = False,
                 value_withdraw_or_depo: float = -0.67):
        """

        :param portfolios: {'port_name_1': {'ticker_1': share, 'ticker_N': share}, 'port_name_N': {'ticker_N': share}}
        :param rebalance: 'monthly', 'quarterly', 'annual'
        :param trade_rebalance_at: 'open': at next open, 'close': at current close
        :param forsed_rebalance: on the last available trading day, there will be a rebalancing. 'trade_rebalance_at' automatically becomes 'close'
        :param div_tax:
        :param commision: per one share
        :param acc_maintenance: account maintenance for the period specified in the 'rebalance', in dollars
        :param balance_start:
        :param date_start:
        :param date_end:
        :param benchmark:
        :param withdraw_or_depo: have any withdrawal/deposit for the period specified in the 'rebalance'
        :param value_withdraw_or_depo: in percent. Negative value - withdrawal. Positive - deposit.
        """

        # Internals
        self.all_tickers_data = {}
        self.strategy_data = {}
        self.div_storage = []
        self.trading_days = []
        self.capital_not_placed = True
        self.max_position = 0
        self.current_port = ''

        # Properties
        self.portfolios = portfolios
        self.rebalance = rebalance
        self.rebalance_at = trade_rebalance_at if forsed_rebalance is False else 'close'
        self.forsed_rebalance = forsed_rebalance
        self.div_tax = div_tax
        self.commision = commision
        self.acc_maintenance = acc_maintenance
        self.balance_start = balance_start
        self.date_start = date_start
        self.date_end = date_end
        self.bench = benchmark
        self.withd_depo = withdraw_or_depo
        self.value_withd_depo = value_withdraw_or_depo

        assert self.rebalance in ['monthly', 'quarterly', 'annual'], f"Incorrect value for 'rebalance'"
        assert self.rebalance_at in ['open', 'close'], f"Incorrect value for 'rebalance_at'"

    # Junior functions for calculations -------------------------------------------------------------------------------
    def commision_calc(self, shares: float) -> float:
        if shares >= 100.0:
            return shares * self.commision
        else:
            return 100 * self.commision

    def yield_std_by_every_year(self, df_strategy: pd.DataFrame) -> pd.DataFrame:

        data = self.all_tickers_data
        price = 'Close' if self.rebalance_at == 'close' else 'Open'

        df_strategy['Bench'] = data[self.bench][price].add(data[self.bench]['Dividend'].cumsum())
        capital_by_years = df_strategy.set_index(df_strategy['Date']).resample('Y')['Capital', 'Bench']
        year_yield = capital_by_years.agg(lambda x: (x[-1] / x[0] - 1) * 100)
        year_std = capital_by_years.agg(lambda x: np.std(np.diff(x) / x[:-1] * 100)) * np.sqrt(252)

        worksheet = pd.concat([year_yield, year_std], axis=1)
        worksheet.columns = ['Yield_Port', 'Yield_' + self.bench, 'StDev_Port', 'StDev_' + self.bench]
        worksheet['Date'] = worksheet.index.year
        worksheet = worksheet[['Date', 'Yield_Port', 'Yield_' + self.bench, 'StDev_Port', 'StDev_' + self.bench]]

        return worksheet.round(2)

    # Logical chain of functions --------------------------------------------------------------------------------------
    def get_data(self, reload_data: bool = False) -> None:
        """
        Check all tickers for data, download data if not available, cut data by oldest and newest dates. After all, it
        call creation tickers columns in strategy dict

        :param reload_data: reload data or not
        :return: put data for all tickers to all_tickers_data
        """

        unique_tickers = {self.bench}
        for portfolio in self.portfolios.items():
            unique_tickers.update(portfolio[1].keys())

        for ticker in unique_tickers:
            if os.path.isfile(os.path.join(FOLDER_WITH_DATA, ticker + '.csv')) is False or reload_data:
                tl.download_yahoo(ticker)

        oldest_date = [self.date_start]
        newest_date = [self.date_end]
        for ticker in unique_tickers:
            self.all_tickers_data[ticker] = tl.load_csv(ticker)
            oldest_date.append(self.all_tickers_data[ticker]['Date'][0])
            newest_date.append(self.all_tickers_data[ticker]['Date'].iloc[-1])

        start_date = max(oldest_date)
        end_date = min(newest_date)
        for ticker in unique_tickers:
            self.all_tickers_data[ticker] = self.all_tickers_data[ticker].loc[
                (self.all_tickers_data[ticker]['Date'] >= start_date) &
                (self.all_tickers_data[ticker]['Date'] <= end_date)
            ]
            self.all_tickers_data[ticker] = self.all_tickers_data[ticker].reset_index(drop=True)

        self.create_columns_for_strategy_dict()

    def create_columns_for_strategy_dict(self):
        """
        Creates special columns for each ticker in the 'strategy_data', in the amount of the maximum number of
        specified tickers. After all, it call iterator of all trading days

        :return:
        """

        self.max_position = 0
        for portfolio in self.portfolios.items():
            if self.max_position < len(portfolio[1].keys()):
                self.max_position = len(portfolio[1].keys())

        first_ticker = next(iter(self.all_tickers_data))
        self.trading_days = self.all_tickers_data[first_ticker]['Date']
        number_of_rows = len(self.all_tickers_data[first_ticker])

        self.strategy_data['Date'] = self.trading_days
        for i in range(1, self.max_position + 1):
            self.strategy_data['Ticker_' + str(i)] = [0] * number_of_rows
            self.strategy_data['Shares_' + str(i)] = [0] * number_of_rows
            self.strategy_data['Dividend_' + str(i)] = [0] * number_of_rows
            self.strategy_data['Price_' + str(i)] = [0] * number_of_rows
        self.strategy_data['Cash'] = [0] * number_of_rows
        self.strategy_data['Capital'] = [0] * number_of_rows
        self.strategy_data['InOutCash'] = [0] * number_of_rows

        self.iterate_trading_days()

    def iterate_trading_days(self):
        """
        Check whether you need to do something on a given day or is it a typical day

        :return:
        """

        for day_number in range(len(self.trading_days)):

            if self.rebalance_at == 'close':

                if day_number != len(self.trading_days) - 1:

                    if self.rebalance == 'monthly' and self.trading_days[day_number].month != \
                            self.trading_days[day_number+1].month:
                        self.change_port(day_number)

                    elif self.rebalance == 'quarterly' and self.trading_days[day_number].month in (3, 6, 9, 12) and \
                            self.trading_days[day_number+1].month in (4, 7, 10, 1):
                        self.change_port(day_number)

                    elif self.rebalance == 'annual' and self.trading_days[day_number].year != \
                            self.trading_days[day_number+1].year:
                        self.change_port(day_number)

                    elif self.capital_not_placed is False:
                        self.typical_day(day_number)

                elif self.forsed_rebalance:
                    self.change_port(day_number)

                elif self.capital_not_placed is False:
                    self.typical_day(day_number)

            elif self.rebalance_at == 'open':

                if day_number != 0:

                    if self.rebalance == 'monthly' and self.trading_days[day_number-1].month != \
                            self.trading_days[day_number].month:
                        self.change_port(day_number)

                    elif self.rebalance == 'quarterly' and self.trading_days[day_number-1].month in (3, 6, 9, 12) and \
                            self.trading_days[day_number].month in (4, 7, 10, 1):
                        self.change_port(day_number)

                    elif self.rebalance == 'annual' and self.trading_days[day_number-1].year != \
                            self.trading_days[day_number].year:
                        self.change_port(day_number)

                    elif day_number == len(self.trading_days) - 1 and self.forsed_rebalance:
                        self.change_port(day_number)

                    elif self.capital_not_placed is False:
                        self.typical_day(day_number)

    def change_port(self, day_number: int):
        """
        Checking how we need to rebalance or, if the portfolio does not yet exist, what composition to take it with.
        If there are many portfolios, then there is a check which one is suitable.

        :param day_number:
        :return:
        """

        if self.capital_not_placed:
            self.dont_have_any_port(next(iter(self.portfolios)), day_number)
        else:
            self.rebalance_port(next(iter(self.portfolios)), day_number)

    def dont_have_any_port(self, port_name: str, day_number: int):
        """
        If we need withdrawal or deposit - change start balance by 'self.value_withd_depo'. Calculate capital after
        commisions, after that calculate shares and write ticker, price. Dividend is equal zero, because we can't take
        dividend in that day. Wtire cash, if our total positions weights less, than 1.

        :param port_name: the name of the portfolio in which we place capital, from 'self.portfolios'
        :param day_number:
        :return:
        """

        port_weights = self.portfolios[port_name]
        data = self.all_tickers_data
        price = 'Close' if self.rebalance_at == 'close' else 'Open'
        capital_after_trades = 0
        total_pos_weight = 0

        if self.withd_depo:
            self.strategy_data['InOutCash'][day_number] = self.balance_start * (self.value_withd_depo / 100)
            self.balance_start = self.balance_start * (1 + self.value_withd_depo / 100)

        i = 1
        for ticker in port_weights.keys():
            capital_after_comm = port_weights[ticker] * self.balance_start - \
                                 self.commision_calc(port_weights[ticker] * self.balance_start / data[ticker][price][day_number])
            self.strategy_data['Ticker_' + str(i)][day_number] = ticker
            self.strategy_data['Shares_' + str(i)][day_number] = capital_after_comm / data[ticker][price][day_number]
            self.strategy_data['Price_' + str(i)][day_number] = data[ticker][price][day_number]
            capital_after_trades += capital_after_comm
            total_pos_weight += port_weights[ticker]
            i += 1

        self.strategy_data['Cash'][day_number] = (1 - total_pos_weight) * self.balance_start
        self.strategy_data['Capital'][day_number] = capital_after_trades + self.strategy_data['Cash'][day_number]
        self.capital_not_placed = False
        self.current_port = port_name

    def rebalance_port(self, port_name: str, day_number: int):
        """
        At first, calculate total capital include dividends, withdrawal or deposit and commissions after rebalance. Then
        change values in 'self.strategy_data' to new portfolio.

        :param port_name: the name of the portfolio in which we place capital, from 'self.portfolios'
        :param day_number:
        :return:
        """

        port_weights = self.portfolios[port_name]
        data = self.all_tickers_data
        price = 'Close' if self.rebalance_at == 'close' else 'Open'

        # Capital and dividends
        capital = self.strategy_data['Cash'][day_number-1]
        for i in range(1, len(self.portfolios[self.current_port]) + 1):
            ticker = self.strategy_data['Ticker_' + str(i)][day_number-1]
            capital += self.strategy_data['Shares_' + str(i)][day_number-1] * data[ticker][price][day_number]
            self.div_storage.append(self.strategy_data['Shares_' + str(i)][day_number - 1] *
                                    data[ticker]['Dividend'][day_number] * self.div_tax)
        capital += sum(self.div_storage)

        # Withdrawal or deposit
        if self.withd_depo:
            self.strategy_data['InOutCash'][day_number] = capital * (self.value_withd_depo / 100)
            capital = capital * (1 + self.value_withd_depo / 100)

        # Commissions
        commisions = 0
        for i in range(1, self.max_position + 1):
            if self.strategy_data['Ticker_' + str(i)][day_number-1] != 0 \
                    and self.strategy_data['Ticker_' + str(i)][day_number-1] in port_weights.keys():

                ticker = self.strategy_data['Ticker_' + str(i)][day_number-1]
                required_shares = capital * port_weights[ticker] / data[ticker][price][day_number]
                share_difference = required_shares - self.strategy_data['Shares_' + str(i)][day_number-1]
                commisions += self.commision_calc(abs(share_difference))

            elif self.strategy_data['Ticker_' + str(i)][day_number-1] != 0:
                commisions += self.commision_calc(self.strategy_data['Shares_' + str(i)][day_number-1])

        new_tickers = list(port_weights.keys() - self.portfolios[self.current_port].keys())
        for ticker in new_tickers:
            new_shares = capital * port_weights[ticker] / data[ticker][price][day_number]
            commisions += self.commision_calc(new_shares)

        # Change self.strategy_data
        capital -= commisions
        total_pos_weight = 0
        for i in range(1, len(port_weights.keys()) + 1):
            ticker = list(port_weights.keys())[i-1]
            self.strategy_data['Ticker_' + str(i)][day_number] = ticker
            self.strategy_data['Shares_' + str(i)][day_number] = capital * port_weights[ticker] / data[ticker][price][day_number]
            self.strategy_data['Price_' + str(i)][day_number] = data[ticker][price][day_number]
            total_pos_weight += port_weights[ticker]

        self.strategy_data['Cash'][day_number] = capital * (1 - total_pos_weight)
        self.strategy_data['Capital'][day_number] = capital
        self.current_port = port_name
        self.div_storage = []

    def typical_day(self, day_number: int):

        data = self.all_tickers_data
        price = 'Close' if self.rebalance_at == 'close' else 'Open'
        capital = 0

        for i in range(1, len(self.portfolios[self.current_port]) + 1):
            ticker = self.strategy_data['Ticker_' + str(i)][day_number-1]
            self.strategy_data['Ticker_' + str(i)][day_number] = self.strategy_data['Ticker_' + str(i)][day_number-1]
            self.strategy_data['Shares_' + str(i)][day_number] = self.strategy_data['Shares_' + str(i)][day_number-1]
            self.strategy_data['Dividend_' + str(i)][day_number] = data[ticker]['Dividend'][day_number]
            self.strategy_data['Price_' + str(i)][day_number] = data[ticker][price][day_number]
            self.div_storage.append(self.strategy_data['Shares_' + str(i)][day_number] *
                                    data[ticker]['Dividend'][day_number] * self.div_tax)
            capital += self.strategy_data['Shares_' + str(i)][day_number] * data[ticker][price][day_number]

        self.strategy_data['Cash'][day_number] = self.strategy_data['Cash'][day_number-1]
        self.strategy_data['Capital'][day_number] = capital + self.strategy_data['Cash'][day_number]


if __name__ == '__main__':
    portfolios = {'Port_1':
                      {'SPY': .5, 'DIA': .5}
                  }
    test_port = BasePortfolio(portfolios=portfolios,
                              balance_start=100_000,
                              date_start=datetime(1999, 12, 1),
                              rebalance='quarterly',
                              trade_rebalance_at='close')
    test_port.get_data()

    # Transform data
    df_strategy = pd.DataFrame.from_dict(test_port.strategy_data)
    df_strategy = df_strategy[df_strategy.Capital != 0]
    df_strategy['Capital'] = df_strategy.Capital.astype(int)

    df_yield_by_years = test_port.yield_std_by_every_year(df_strategy)

    tl.plot_capital_plotly(FOLDER_WITH_IMG + 'Portfolio_Momentum',
                           list(df_strategy.Date),
                           list(df_strategy.Capital),
                           df_yield_by_years,
                           portfolios)

    tl.save_csv(FOLDER_TO_SAVE,
                f"InvestModel_EQ",
                df_strategy)
