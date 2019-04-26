from datetime import datetime
from dateutil.relativedelta import relativedelta as rdelta
from libs import trading_lib as tl

import base_ports
import numpy as np
import pandas as pd


class TargetVolatility(base_ports.BasePortfolio):
    __slots__ = [
        'vol_calc_period',
        'vol_calc_range',
        'vol_target',
        'use_margin',
        'vola_type',

        'cur_leverage',
        'on_capital',
        'off_capital'
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
                 vola_type: str = 'standard',
                 use_margin: bool = False):

        """

        :param portfolios: {'risk_on': {'ticker_1': share, 'ticker_N': share}, 'risk_off': {'ticker_N': share}}. Portfolios names: 'risk_on', 'risk_off'
        :param rebalance: 'weekly', 'monthly'
        :param trade_rebalance_at: 'open': at next open, 'close': at current close
        :param forsed_rebalance: on the last available trading day, there will be a rebalancing. 'trade_rebalance_at' automatically becomes 'close'
        :param div_tax:
        :param commision: per one share
        :param balance_start:
        :param date_start: Important what month you're pointing out. The portfolio will be started at the end of the month.
        :param date_end:
        :param benchmark:
        :param withdraw_or_depo: have any withdrawal/deposit for the period specified in the 'rebalance'
        :param value_withdraw_or_depo: in percent. Negative value - withdrawal. Positive - deposit.
        :param vol_calc_period: 'day', 'month'. What time slice is taken for calculations. Volatility will be counted by day or month.
        :param vol_calc_range: Number of time slices. For example, if vol_calc_period = day and vol_calc_range = 20, then the volatility will be calculated based on the past 20 days. If vol_calc_period = day, than will be calculated based on the past 20 months.
        :param vol_target:
        :para, vola_type: 'standard' - we calculate vol just for risk_on portfolio to capital, like PVZ. 'modify' - we calculate vol for risk_on and risk_off portfolios. That is, we consider the vol of our entire portfolio, taking into account the impact of the risk-off part.
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
        self.cur_leverage = 0.0
        self.on_capital = {}
        self.off_capital = {}

        # Strategy parameters
        self.vol_calc_period = vol_calc_period
        self.vol_calc_range = vol_calc_range
        self.vol_target = vol_target
        self.vola_type = vola_type
        self.use_margin = use_margin

        assert self.rebalance in ['weekly', 'monthly', 'quarterly', 'annual'], \
            "Incorrect value for 'rebalance'"
        assert self.vol_calc_period in ['day', 'month'], \
            "Incorrect value for 'vol_calc_period'"
        assert self.vola_type in ['standard', 'modify'], \
            "Incorrect value for 'vola_type'"
        if self.rebalance == 'weekly' and self.vol_calc_period != 'day':
            raise ValueError("For weekly rebalance, please, use day calc period")

        for port_name in portfolios.keys():
            assert port_name in ['risk_on', 'risk_off'], \
                f"Incorrect value in portfolios names - {port_name}"

    # Junior functions for calculations -------------------------------------------------------------------------------
    def data_sufficiency_check(self, start_index: int):

        for index in range(start_index, len(self.trading_days)):
            if self.vol_calc_period == 'month' and \
                    (self.trading_days[index] - rdelta(months=self.vol_calc_range) >= self.trading_days[0]):
                break
            elif self.vol_calc_period == 'day' and len(self.trading_days[:index+1]) >= self.vol_calc_range:
                break
        return index

    def calculate_current_vol(self, day_number: int):

        if self.rebalance_at == 'open':
            day_number -= 1

        data = self.all_tickers_data
        strata = self.strategy_data

        # Find index for start_date
        if self.vol_calc_period == 'month':
            start_date = self.trading_days[day_number] - rdelta(months=self.vol_calc_range)
            start_date = self.trading_days[
                (self.trading_days.dt.year == start_date.year) & (self.trading_days.dt.month == start_date.month)].iloc[-1]
            start_index = list(self.trading_days).index(start_date)
        else:
            start_index = day_number - self.vol_calc_range

        # Calculate capital, if port is not placed yet or we calc standard volatility type
        if self.capital_not_placed or self.vola_type == 'standard':
            virtual_port = np.zeros(len(self.trading_days[start_index:day_number+1]))

            for ticker in self.portfolios['risk_on'].keys():
                ticker_data = data[ticker]
                prices = ticker_data[start_index:day_number+1][self.price]
                virtual_port += np.array(prices) * self.portfolios['risk_on'][ticker]

        # Calculate modify volatility type
        else:
            total_cap = 0

            for ticker in self.portfolios['risk_on'].keys():
                total_cap += strata['On_Shares_' + ticker][day_number-1] * strata['On_Price_' + ticker][day_number]
            for ticker in self.portfolios['risk_off'].keys():
                total_cap += strata['Off_Shares_' + ticker][day_number-1] * strata['Off_Price_' + ticker][day_number]

            strata['Capital'][day_number] = total_cap + strata['Cash'][day_number-1]
            capitals = strata['Capital'][start_index:day_number+1]
            virtual_port = np.array(capitals)

        virtual_port_cng = np.diff(virtual_port) / virtual_port[:-1] * 100
        vol_coff = self.vol_target / (np.std(virtual_port_cng, ddof=1) * np.sqrt(252))

        self.cur_leverage = min(vol_coff, 1) if self.use_margin is False else vol_coff

    def calculate_capital_at_rebalance_day(self, day_number: int) -> float:

        data = self.all_tickers_data
        strata = self.strategy_data
        capital = self.strategy_data['Cash'][day_number - 1]
        self.on_capital, self.off_capital = {}, {}

        for ticker in self.portfolios['risk_on'].keys():
            self.on_capital[ticker] = strata['On_Shares_' + ticker][day_number-1] * data[ticker][self.price][day_number]
            self.div_storage.append(strata['On_Shares_' + ticker][day_number-1] * data[ticker]['Dividend'][day_number] *
                                    self.div_tax)

        for ticker in self.portfolios['risk_off'].keys():
            self.off_capital[ticker] = strata['Off_Shares_' + ticker][day_number-1] * data[ticker][self.price][day_number]
            self.div_storage.append(strata['Off_Shares_' + ticker][day_number-1] * data[ticker]['Dividend'][day_number] *
                                    self.div_tax)
        capital += sum(self.div_storage) + sum(self.on_capital.values()) + sum(self.off_capital.values())

        # Withdrawal or deposit
        if self.withd_depo:
            strata['InOutCash'][day_number] = capital * (self.value_withd_depo / 100)
            capital = capital * (1 + self.value_withd_depo / 100)

        return capital

    def rebalance_commissions(self, capital: float, day_number: int) -> float:

        data = self.all_tickers_data
        strata = self.strategy_data

        # Считаем комиссии риск-он
        commissions = 0
        for ticker in self.portfolios['risk_on'].keys():
            old_leverage = self.on_capital[ticker] / capital
            change_leverage = self.portfolios['risk_on'][ticker] * self.cur_leverage / old_leverage
            shares_for_trade = strata['On_Shares_' + ticker][day_number-1] * change_leverage - \
                               strata['On_Shares_' + ticker][day_number-1]
            commissions += self.trade_commiss(abs(shares_for_trade))

        # Считаем комиссии риск-офф
        for ticker in self.portfolios['risk_off'].keys():
            risk_off_leverage = max(0, 1 - self.cur_leverage)
            old_leverage = self.off_capital[ticker] / capital

            if old_leverage == 0:
                risk_off_capital = self.portfolios['risk_off'][ticker] * capital * risk_off_leverage
                shares_for_trade = risk_off_capital / data[ticker][self.price][day_number]
                commissions += self.trade_commiss(abs(shares_for_trade))
            else:
                change_leverage = self.portfolios['risk_off'][ticker] * risk_off_leverage / old_leverage
                shares_for_trade = strata['Off_Shares_' + ticker][day_number-1] * change_leverage - \
                                   strata['Off_Shares_' + ticker][day_number-1]
                commissions += self.trade_commiss(abs(shares_for_trade))

        return commissions

    # Logical chain of functions --------------------------------------------------------------------------------------
    def create_columns_for_strategy_dict(self):

        first_ticker = next(iter(self.all_tickers_data))
        self.trading_days = self.all_tickers_data[first_ticker]['Date']
        number_of_rows = len(self.all_tickers_data[first_ticker])

        self.strategy_data['Date'] = self.trading_days
        for t in self.portfolios['risk_on'].keys():
            self.strategy_data['On_Price_' + t] = self.all_tickers_data[t][self.price]
            self.strategy_data['On_Shares_' + t] = [0] * number_of_rows
            self.strategy_data['On_Dividend_' + t] = self.all_tickers_data[t]['Dividend']

        for t in self.portfolios['risk_off'].keys():
            self.strategy_data['Off_Price_' + t] = self.all_tickers_data[t][self.price]
            self.strategy_data['Off_Shares_' + t] = [0] * number_of_rows
            self.strategy_data['Off_Dividend_' + t] = self.all_tickers_data[t]['Dividend']
        self.strategy_data['Cash'] = [0] * number_of_rows
        self.strategy_data['Capital'] = [0] * number_of_rows
        self.strategy_data['InOutCash'] = [0] * number_of_rows

        print(f"self.strategy_data columns: {list(self.strategy_data.keys())}")

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
        strata = self.strategy_data
        cap_after_trades = 0
        total_pos_weight = 0

        if self.withd_depo:
            strata['InOutCash'][day_number] = self.balance_start * (self.value_withd_depo / 100)
            self.balance_start = self.balance_start * (1 + self.value_withd_depo / 100)

        # Risk on
        for ticker in self.portfolios['risk_on'].keys():
            risk_on_capital = self.portfolios['risk_on'][ticker] * self.balance_start * self.cur_leverage
            capital_after_comm = risk_on_capital - self.trade_commiss(risk_on_capital / data[ticker][self.price][
                day_number])
            strata['On_Shares_' + ticker][day_number] = capital_after_comm / data[ticker][self.price][day_number]
            cap_after_trades += capital_after_comm
            total_pos_weight += self.portfolios['risk_on'][ticker] * self.cur_leverage

        # Risk off
        for ticker in self.portfolios['risk_off'].keys():
            if self.cur_leverage <= 1:
                risk_off_capital = self.portfolios['risk_off'][ticker] * self.balance_start * (1 - self.cur_leverage)
                capital_after_comm = risk_off_capital - self.trade_commiss(risk_off_capital / data[ticker][self.price][
                    day_number])
                strata['Off_Shares_' + ticker][day_number] = capital_after_comm / data[ticker][self.price][day_number]
                cap_after_trades += capital_after_comm
                total_pos_weight += self.portfolios['risk_off'][ticker] * (1 - self.cur_leverage)

        strata['Cash'][day_number] = (1 - total_pos_weight) * self.balance_start
        strata['Capital'][day_number] = cap_after_trades + strata['Cash'][day_number]
        self.capital_not_placed = False

    def rebalance_port(self, capital: float, day_number: int):

        data = self.all_tickers_data
        strata = self.strategy_data

        # Change self.strategy_data
        total_pos_weight = 0

        for ticker in self.portfolios['risk_on'].keys():
            risk_on_capital = self.portfolios['risk_on'][ticker] * capital * self.cur_leverage
            strata['On_Shares_' + ticker][day_number] = risk_on_capital / data[ticker][self.price][day_number]
            total_pos_weight += self.portfolios['risk_on'][ticker] * self.cur_leverage

        for ticker in self.portfolios['risk_off'].keys():
            if self.cur_leverage <= 1:
                risk_off_capital = self.portfolios['risk_off'][ticker] * capital * (1 - self.cur_leverage)
                strata['Off_Shares_' + ticker][day_number] = risk_off_capital / data[ticker][self.price][day_number]
                total_pos_weight += self.portfolios['risk_off'][ticker] * (1 - self.cur_leverage)

        strata['Cash'][day_number] = (1 - total_pos_weight) * capital
        strata['Capital'][day_number] = capital
        self.div_storage = []

    def typical_day(self, day_number: int):

        data = self.all_tickers_data
        strata = self.strategy_data
        capital = 0

        for ticker in self.portfolios['risk_on'].keys():
            strata['On_Shares_' + ticker][day_number] = strata['On_Shares_' + ticker][day_number-1]
            capital += strata['On_Shares_' + ticker][day_number] * data[ticker][self.price][day_number]
            self.div_storage.append(
                strata['On_Shares_' + ticker][day_number] * data[ticker]['Dividend'][day_number] * self.div_tax)

        for ticker in self.portfolios['risk_off'].keys():
            strata['Off_Shares_' + ticker][day_number] = strata['Off_Shares_' + ticker][day_number-1]
            capital += strata['Off_Shares_' + ticker][day_number] * data[ticker][self.price][day_number]
            self.div_storage.append(
                strata['Off_Shares_' + ticker][day_number] * data[ticker]['Dividend'][day_number] * self.div_tax)

        strata['Cash'][day_number] = strata['Cash'][day_number - 1]
        strata['Capital'][day_number] = capital + strata['Cash'][day_number]


def working_with_capital(test_port, day_number: int):
    test_port.calculate_current_vol(day_number)
    if test_port.capital_not_placed:
        test_port.dont_have_any_port(day_number)
    else:
        capital = test_port.calculate_capital_at_rebalance_day(day_number)
        capital -= test_port.rebalance_commissions(capital, day_number)
        test_port.rebalance_port(capital, day_number)


def start(test_port) -> (pd.DataFrame, pd.DataFrame, str):
    # Preprocessing data
    test_port.download_data()
    start_date, end_date = test_port.find_oldest_newest_dates()
    test_port.cut_data_by_dates(start_date, end_date)
    test_port.create_columns_for_strategy_dict()

    # Iterate_trading_days
    start_index = test_port.start_day_index()
    start_index = test_port.data_sufficiency_check(start_index)
    for day_number in range(start_index, len(test_port.trading_days)):

        if test_port.rebalance_at == 'close':

            if day_number != len(test_port.trading_days) - 1:

                if test_port.rebalance == 'weekly' and abs(test_port.trading_days[day_number].weekday() -
                                                           test_port.trading_days[day_number + 1].weekday()) > 2:
                    working_with_capital(test_port, day_number)

                elif test_port.rebalance == 'monthly' and test_port.trading_days[day_number].month != \
                        test_port.trading_days[day_number + 1].month:
                    working_with_capital(test_port, day_number)

                elif test_port.rebalance == 'quarterly' and test_port.trading_days[day_number].month in (
                3, 6, 9, 12) and \
                        test_port.trading_days[day_number + 1].month in (4, 7, 10, 1):
                    working_with_capital(test_port, day_number)

                elif test_port.rebalance == 'annual' and test_port.trading_days[day_number].year != \
                        test_port.trading_days[day_number + 1].year:
                    working_with_capital(test_port, day_number)

                elif test_port.capital_not_placed is False:
                    test_port.typical_day(day_number)

            elif test_port.forsed_rebalance:
                working_with_capital(test_port, day_number)

            elif test_port.capital_not_placed is False:
                test_port.typical_day(day_number)

        elif test_port.rebalance_at == 'open':

            if day_number != 0:

                if test_port.rebalance == 'weekly' and abs(test_port.trading_days[day_number - 1].weekday() -
                                                           test_port.trading_days[day_number].weekday()) > 2:
                    working_with_capital(test_port, day_number)

                elif test_port.rebalance == 'monthly' and test_port.trading_days[day_number - 1].month != \
                        test_port.trading_days[day_number].month:
                    working_with_capital(test_port, day_number)

                elif test_port.rebalance == 'quarterly' and test_port.trading_days[day_number - 1].month in (
                3, 6, 9, 12) and \
                        test_port.trading_days[day_number].month in (4, 7, 10, 1):
                    working_with_capital(test_port, day_number)

                elif test_port.rebalance == 'annual' and test_port.trading_days[day_number - 1].year != \
                        test_port.trading_days[day_number].year:
                    working_with_capital(test_port, day_number)

                elif day_number == len(test_port.trading_days) - 1 and test_port.forsed_rebalance:
                    working_with_capital(test_port, day_number)

                elif test_port.capital_not_placed is False:
                    test_port.typical_day(day_number)

    # Transform data
    df_strategy = pd.DataFrame.from_dict(test_port.strategy_data)
    df_strategy = df_strategy[df_strategy.Capital != 0]
    df_strategy['Capital'] = df_strategy.Capital.astype(int)

    df_yield_by_years = test_port.df_yield_std_by_every_year(df_strategy)

    chart_name = 'TargetVolPort ' + \
                 '(' + test_port.rebalance + ') ' + \
                 '(' + 'by ' + test_port.rebalance_at + ') ' + \
                 '(' + f"VolTarget {test_port.vol_target}" + ') ' + \
                 '(' + f"Period_Range {test_port.vol_calc_period} {test_port.vol_calc_range}" + ') ' + \
                 '(' + f"Type_Leverage {test_port.vola_type} {test_port.use_margin}" + ') '

    return df_strategy, df_yield_by_years, chart_name


if __name__ == "__main__":
    portfolios = {'risk_on':
                      {'DIA': 1.0},
                  'risk_off':
                      {'TLT': 1.0}
                  }
    test_port = TargetVolatility(portfolios=portfolios,
                                 date_start=datetime(2007, 12, 20),
                                 vol_target=9.0,
                                 rebalance='monthly',
                                 vol_calc_period='day',
                                 vol_calc_range=15)

    df_strategy, df_yield_by_years, chart_name = start(test_port)

    tl.plot_capital_plotly(test_port.FOLDER_WITH_IMG + chart_name,
                           list(df_strategy.Date),
                           list(df_strategy.Capital),
                           df_yield_by_years,
                           portfolios)

    tl.save_csv(test_port.FOLDER_TO_SAVE,
                chart_name + str(test_port.ports_tickers()),
                df_strategy)
