from datetime import datetime
from dateutil.relativedelta import relativedelta as rdelta

import base_ports
import numpy as np
import pandas as pd


class TargetVolatility(base_ports.BasePortfolio):
    __slots__ = [
        'vol_calc_period',
        'vol_calc_range',
        'vol_target',
        'use_margin',
        'coeff_current_vol'
    ]

    def __init__(self,
                 portfolios: dict,
                 rebalance: str = 'monthly',
                 trade_rebalance_at: str = 'close',
                 forsed_rebalance: bool = False,
                 div_tax: float = 0.9,
                 commision: float = 0.0055,
                 balance_start: int = 10_000,
                 date_start: datetime = datetime(2008, 1, 1),
                 date_end: datetime = datetime.now(),
                 benchmark: str = 'QQQ',
                 withdraw_or_depo: bool = False,
                 value_withdraw_or_depo: float = -0.67,
                 vol_calc_period: str = 'month',
                 vol_calc_range: int = 1,
                 vol_target: float = 9.0,
                 use_margin: bool = False):

        """

        :param portfolios: {'RISK_ON': {'ticker_1': share, 'ticker_N': share}, 'RISK_OFF': {'ticker_N': share}}
        :param rebalance: 'weekly', 'monthly'
        :param trade_rebalance_at: 'open': at next open, 'close': at current close
        :param forsed_rebalance: on the last available trading day, there will be a rebalancing. 'trade_rebalance_at' automatically becomes 'close'
        :param div_tax:
        :param commision: per one share
        :param balance_start:
        :param date_start:
        :param date_end:
        :param benchmark:
        :param withdraw_or_depo: have any withdrawal/deposit for the period specified in the 'rebalance'
        :param value_withdraw_or_depo: in percent. Negative value - withdrawal. Positive - deposit.
        :param vol_calc_period: 'day', 'month'. What time slice is taken for calculations. Volatility will be counted by day or month.
        :param vol_calc_range: Number of time slices. For example, if vol_calc_period = day and vol_calc_range = 20, then the volatility will be calculated based on the past 20 days. If vol_calc_period = day, than will be calculated based on the past 20 months.
        :param vol_target:
        :param use_margin: if the volatility is lower than the vol_target, the leverage is used to match the volatility
        """

        base_ports.BasePortfolio.__init__(
            self,
            portfolios=portfolios,
            rebalance=rebalance,
            trade_rebalance_at=trade_rebalance_at if forsed_rebalance is False else 'close',
            forsed_rebalance=forsed_rebalance,
            div_tax=div_tax,
            commision=commision,
            balance_start=balance_start,
            date_start=date_start,
            date_end=date_end,
            benchmark=benchmark,
            withdraw_or_depo=withdraw_or_depo,
            value_withdraw_or_depo=value_withdraw_or_depo,
        )

        # Internals
        self.coeff_current_vol= 0.0

        # Strategy parameters
        self.vol_calc_period = vol_calc_period
        self.vol_calc_range = vol_calc_range
        self.vol_target = vol_target
        self.use_margin = use_margin

        assert self.vol_calc_period in ['day', 'month'], f"Incorrect value for 'vol_calc_period'"

        self.portfolios['risk_on'] = self.portfolios.pop(next(iter(self.portfolios)))
        self.portfolios['risk_off'] = self.portfolios.pop(next(iter(self.portfolios)))

    # Junior functions for calculations -------------------------------------------------------------------------------
    def calculate_current_vol(self, day_number: int):

        # Find index for start_date
        if self.vol_calc_period == 'month':
            start_date = self.trading_days[day_number] - rdelta(months=self.vol_calc_range)
            start_date = self.trading_days[
            (self.trading_days.dt.year == start_date.year) & (self.trading_days.dt.month == start_date.month)].iloc[-1]
            start_index = list(self.trading_days).index(start_date)

        else:
            start_index = day_number - self.vol_calc_range

        # Calculate capital, if port is not placed yet
        if self.capital_not_placed:
            virtual_port = np.zeros(len(self.trading_days[start_index:day_number+1]))

            for ticker in self.portfolios['risk_on'].keys():
                ticker_data = self.all_tickers_data[ticker]
                closes = ticker_data[start_index:day_number+1][self.price]
                virtual_port += np.array(closes) * self.portfolios['risk_on'][ticker]

        # Calculate capital, if we already have port
        else:
            total_cap = 0

            for ticker in self.portfolios['risk_on'].keys():
                total_cap += self.strategy_data['On_Shares_' + ticker][day_number-1] * \
                             self.strategy_data['On_Price_' + ticker][day_number]
            for ticker in self.portfolios['risk_off'].keys():
                total_cap += self.strategy_data['Off_Shares_' + ticker][day_number-1] * \
                             self.strategy_data['Off_Price_' + ticker][day_number]

            self.strategy_data['Capital'][day_number] = total_cap + self.strategy_data['Cash'][day_number-1]
            closes = self.strategy_data[start_index:day_number+1].Capital
            virtual_port = np.array(closes)

        virtual_port_cng = np.diff(virtual_port) / virtual_port[:-1] * 100
        vol_coff = self.vol_target / (np.std(virtual_port_cng, ddof=1) * np.sqrt(252))

        self.coeff_current_vol = min(vol_coff, 1) if self.use_margin is False else vol_coff

    # Logical chain of functions --------------------------------------------------------------------------------------
    def create_columns_for_strategy_dict(self):

        first_ticker = next(iter(self.all_tickers_data))
        self.trading_days = self.all_tickers_data[first_ticker]['Date']
        number_of_rows = len(self.all_tickers_data[first_ticker])

        self.strategy_data['Date'] = self.trading_days
        for t in self.portfolios['risk_on'].keys():
            self.strategy_data['On_Ticker_' + t] = [0] * number_of_rows
            self.strategy_data['On_Price_' + t] = self.all_tickers_data[t][self.price]
            self.strategy_data['On_Shares_' + t] = [0] * number_of_rows
            self.strategy_data['On_Dividend_' + t] = self.all_tickers_data[t]['Dividend']

        for t in self.portfolios['risk_off'].keys():
            self.strategy_data['Off_Ticker_' + t] = [0] * number_of_rows
            self.strategy_data['Off_Price_' + t] = self.all_tickers_data[t][self.price]
            self.strategy_data['Off_Shares_' + t] = [0] * number_of_rows
            self.strategy_data['Off_Dividend_' + t] = self.all_tickers_data[t]['Dividend']
        self.strategy_data['Cash'] = [0] * number_of_rows
        self.strategy_data['Capital'] = [0] * number_of_rows
        self.strategy_data['InOutCash'] = [0] * number_of_rows

        print(f"self.strategy_data columns: {list(self.strategy_data.keys())}")

    def change_port(self, day_number: int):

        if self.capital_not_placed:

            if self.vol_calc_period == 'month' and \
                    (self.trading_days[day_number] - rdelta(months=self.vol_calc_range) >= self.trading_days[0]):
                self.calculate_current_vol(day_number)
                self.dont_have_any_port(day_number)

            elif self.vol_calc_period == 'day' and len(self.trading_days[:day_number+1]) >= self.vol_calc_range:
                self.calculate_current_vol(day_number)
                self.dont_have_any_port(day_number)

            else:
                return

        else:
            self.calculate_current_vol(day_number)
            self.rebalance_port()

    def dont_have_any_port(self, day_number: int):
        """
        If we need withdrawal or deposit - change start balance by 'self.value_withd_depo'. Calculate capital after
        commisions, after that calculate shares and write ticker, price. Dividend is equal zero, because we can't take
        dividend in that day. Wtire cash, if our total positions weights less, than 1.

        :param port_name: the name of the portfolio in which we place capital, from 'self.portfolios'
        :param day_number:
        :return:
        """

        data = self.all_tickers_data
        cap_after_trades = 0
        total_pos_weight = 0

        if self.withd_depo:
            self.strategy_data['InOutCash'][day_number] = self.balance_start * (self.value_withd_depo / 100)
            self.balance_start = self.balance_start * (1 + self.value_withd_depo / 100)

        # Risk on
        for ticker in self.portfolios['risk_on'].keys():
            risk_on_capital = self.portfolios['risk_on'][ticker] * self.balance_start * self.coeff_current_vol
            capital_after_comm = risk_on_capital - self.trade_commiss(risk_on_capital / data[ticker][self.price][
                day_number])
            self.strategy_data['On_Shares_' + ticker][day_number] = capital_after_comm / data[ticker][self.price][
                day_number]
            cap_after_trades += capital_after_comm
            total_pos_weight += self.portfolios['risk_on'][ticker] * self.coeff_current_vol

        # Risk off
        for ticker in self.portfolios['risk_off'].keys():
            if self.coeff_current_vol <= 1:
                risk_off_capital = self.portfolios['risk_off'][ticker] * self.balance_start * (1 - self.coeff_current_vol)
                capital_after_comm = risk_off_capital - self.trade_commiss(risk_off_capital / data[ticker][self.price][
                    day_number])
                self.strategy_data['Off_Shares_' + ticker][day_number] = capital_after_comm / data[ticker][self.price][
                    day_number]
                cap_after_trades += capital_after_comm
                total_pos_weight += self.portfolios['risk_off'][ticker] * (1 - self.coeff_current_vol)

        self.strategy_data['Cash'][day_number] = (1 - total_pos_weight) * self.balance_start
        self.strategy_data['Capital'][day_number] = cap_after_trades + self.strategy_data['Cash'][day_number]
        self.capital_not_placed = False

    def rebalance_port(self, port_name: str, day_number: int):

        data = self.all_tickers_data

        capital = self.calculate_capital_at_rebalance_day(port_name, day_number)
        capital = capital - self.rebalance_commissions(capital, port_name, day_number)

        # Change self.strategy_data
        total_pos_weight = 0

        for ticker in self.portfolios['risk_on'].keys():
            risk_on_capital = self.portfolios['risk_on'][ticker] * capital * self.coeff_current_vol
            self.strategy_data['On_Shares_' + ticker][day_number] = risk_on_capital / data[ticker][self.price][day_number]
            total_pos_weight += self.portfolios['risk_on'][ticker] * self.coeff_current_vol

        for ticker in self.portfolios['risk_off'].keys():
            if self.coeff_current_vol <= 1:
                risk_off_capital = self.portfolios['risk_off'][ticker] * capital * (1 - self.coeff_current_vol)
                self.strategy_data['Off_Shares_' + ticker][day_number] = risk_off_capital / data[ticker][self.price][day_number]
                total_pos_weight += self.portfolios['risk_off'][ticker] * (1 - self.coeff_current_vol)

        self.strategy_data['Cash'][day_number] = (1 - total_pos_weight) * capital
        self.strategy_data['Capital'][day_number] = capital
        self.div_storage = []


if __name__ == "__main__":
    portfolios = {'Risk_On':
                      {'SPY': .4, 'DIA': .6},
                  'Risk_Off':
                      {'TLT': .8, 'GLD': .2}
                  }
    test_port = TargetVolatility(portfolios=portfolios,
                                 date_start=datetime(2007, 11, 10),
                                 vol_target=9.0)

    # Preprocessing data
    test_port.download_data()
    start_date, end_date = test_port.find_oldest_newest_dates()
    test_port.cut_data_by_dates(start_date, end_date)
    test_port.create_columns_for_strategy_dict()

    # Iterate_trading_days
    for day_number in range(len(test_port.trading_days)):

        if test_port.rebalance_at == 'close':

            if day_number != len(test_port.trading_days) - 1:

                if test_port.rebalance == 'monthly' and test_port.trading_days[day_number].month != \
                        test_port.trading_days[day_number + 1].month:
                    test_port.change_port(day_number)

                elif test_port.rebalance == 'quarterly' and test_port.trading_days[day_number].month in (3, 6, 9, 12) and \
                        test_port.trading_days[day_number + 1].month in (4, 7, 10, 1):
                    test_port.change_port(day_number)

                elif test_port.rebalance == 'annual' and test_port.trading_days[day_number].year != \
                        test_port.trading_days[day_number + 1].year:
                    test_port.change_port(day_number)

                elif test_port.capital_not_placed is False:
                    test_port.typical_day(day_number)

            elif test_port.forsed_rebalance:
                test_port.change_port(day_number)

            elif test_port.capital_not_placed is False:
                test_port.typical_day(day_number)

        elif test_port.rebalance_at == 'open':

            if day_number != 0:

                if test_port.rebalance == 'monthly' and test_port.trading_days[day_number - 1].month != \
                        test_port.trading_days[day_number].month:
                    test_port.change_port(day_number)

                elif test_port.rebalance == 'quarterly' and test_port.trading_days[day_number - 1].month in (3, 6, 9, 12) and \
                        test_port.trading_days[day_number].month in (4, 7, 10, 1):
                    test_port.change_port(day_number)

                elif test_port.rebalance == 'annual' and test_port.trading_days[day_number - 1].year != \
                        test_port.trading_days[day_number].year:
                    test_port.change_port(day_number)

                elif day_number == len(test_port.trading_days) - 1 and test_port.forsed_rebalance:
                    test_port.change_port(day_number)

                elif test_port.capital_not_placed is False:
                    test_port.typical_day(day_number)
