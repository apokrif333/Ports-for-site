from datetime import datetime
from libs import trading_lib as tl

import pandas as pd
import numpy as np
import os


class BasePortfolio:
    FOLDER_WITH_DATA = 'historical_data'
    FOLDER_TO_SAVE = 'saved_data'
    FOLDER_WITH_IMG = 'saved_data/img'

    # Field list
    __slots__ = [
        'all_tickers_data',
        'strategy_data',
        'div_storage',
        'trading_days',
        'capital_not_placed',

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
                 trade_rebalance_at: str = 'open',
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

        # Properties
        self.portfolios = portfolios
        self.rebalance = rebalance
        self.rebalance_at = trade_rebalance_at if self.forsed_rebalance is False else 'close'
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
            if os.path.isfile(os.path.join(self.FOLDER_WITH_DATA, ticker + '.csv')) is False or reload_data:
                tl.download_yahoo(ticker)

        oldest_date = []
        newest_date = []
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

        max_position = 0
        for portfolio in self.portfolios.items():
            if max_position < len(portfolio[1].keys()):
                max_position = len(portfolio[1].keys())

        first_ticker = next(iter(self.all_tickers_data))
        self.trading_days = self.all_tickers_data[first_ticker]['Date']
        number_of_rows = len(self.all_tickers_data[first_ticker])

        self.strategy_data['Date'] = self.trading_days
        for i in range(1, max_position + 1):
            self.strategy_data['Ticker_' + str(i)] = [0] * number_of_rows
            self.strategy_data['Shares_' + str(i)] = [0] * number_of_rows
            self.strategy_data['DivTicker_' + str(i)] = [0] * number_of_rows
            self.strategy_data['Price_' + str(i)] = [0] * number_of_rows
        self.strategy_data['Cash'] = [0] * number_of_rows
        self.strategy_data['Capital'] = [0] * number_of_rows
        self.strategy_data['InOutCash'] = [0] * number_of_rows

        self.iterate_trading_days()

    def iterate_trading_days(self):

        for day_number in range(len(self.trading_days)):
            if self.rebalance_at == 'close':

                if self.rebalance == 'monthly':
                    if day_number != len(self.trading_days) + 1 \
                            and self.trading_days[day_number].month != self.trading_days[day_number+1].month:
                        self.rebalance_port(day_number)

                elif self.rebalance == 'quarterly':
                    if day_number != len(self.trading_days) + 1 and self.trading_days[day_number].month in (3, 6, 9, 12)\
                            and self.trading_days[day_number+1].month in (4, 7, 10, 1):
                        self.rebalance_port(day_number)

                elif self.rebalance_at == 'annual':
                    if day_number != len(self.trading_days) + 1 \
                            and self.trading_days[day_number].year != self.trading_days[day_number+1].year:
                        self.rebalance_port(day_number)

                elif day_number == len(self.trading_days) and self.forsed_rebalance:
                    self.rebalance_port(day_number)

            elif self.rebalance_at == 'open':

                if self.rebalance == 'monthly':
                    if day_number != 0 and self.trading_days[day_number-1].month != self.trading_days[day_number].month:
                        self.rebalance_port(day_number)

                elif self.rebalance == 'quarterly':
                    if day_number != 0 and self.trading_days[day_number-1].month in (3, 6, 9, 12)\
                            and self.trading_days[day_number].month in (4, 7, 10, 1):
                        self.rebalance_port(day_number)

                elif self.rebalance_at == 'annual':
                    if day_number != 0 and self.trading_days[day_number-1].year != self.trading_days[day_number].year:
                        self.rebalance_port(day_number)

    def rebalance_port(self, day_number: int):
        pass


if __name__ == '__main__':
    test_port = BasePortfolio({'Port_1':
                                   {'SPY': .5, 'DIA': .5}
                               })
    test_port.get_data()
