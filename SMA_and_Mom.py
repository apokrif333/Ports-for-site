from datetime import datetime
from dateutil.relativedelta import relativedelta as rdelta
from libs import trading_lib as tl

import math
import base_ports
import pandas as pd


class SMAandMomentum(base_ports.BasePortfolio):
    __slots__ = [
        'signal_stocks',
        'signal_bonds',
        'sma_period',
        'momentum_stocks',
        'momentum_bonds'
    ]

    def __init__(self,
                 portfolios: dict,
                 rebalance: str = 'monthly',
                 trade_rebalance_at: str = 'close',
                 forsed_rebalance: bool = False,
                 div_tax: float = 0.9,
                 commision: float = 0.0055,
                 balance_start: int = 10_000,
                 date_start: datetime = datetime(2007, 12, 31),
                 date_end: datetime = datetime.now(),
                 benchmark: str = 'QQQ',
                 withdraw_or_depo: bool = False,
                 value_withdraw_or_depo: float = -0.67,
                 signal_stocks: str = 'SPY',
                 signal_bonds: str = 'TLT',
                 sma_period: int = 200,
                 momentum_stocks: int = 2,
                 momentum_bonds: int = 2):
        """

        :param portfolios: {'port_name_1': {'ticker_1': share, 'ticker_N': share}, 'port_name_N': {'ticker_N': share}}. Portfolios names: 'high_risk', 'mid_risk', 'mid_save', 'high_save'
        :param rebalance: 'monthly', 'quarterly', 'annual'
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
        :param signal_stocks: stocks which will be used to calculate simple moving average and momentum for risk_on
        :param signal_bonds: bond's ticker which will be used to calculate momentum for risk_off
        :param sma_period: period for SMA in days
        :param momentum_stocks: period for momentum, number of months
        :param momentum_bonds: period for momentum, number of months
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

        # Strategy parameters
        self.signal_stocks = signal_stocks
        self.signal_bonds = signal_bonds
        self.sma_period = sma_period
        self.momentum_stocks = momentum_stocks
        self.momentum_bonds = momentum_bonds
        self.unique_tickers.update([self.signal_stocks, self.signal_bonds])

        for port_name in portfolios.keys():
            assert port_name in ['high_risk', 'mid_risk', 'mid_save', 'high_save'], \
                f"Incorrect value in portfolios names - {port_name}"

    # Junior functions for calculations -------------------------------------------------------------------------------
    def calculate_sma(self, recalculate: bool = False):

        df = tl.load_csv(self.signal_stocks)
        if recalculate or 'SMA_' + str(self.sma_period) not in df.keys():
            print(f"Calculate SMA for {self.signal_stocks} with {self.sma_period} period")
            df['SMA_' + str(self.sma_period)] = round(df['Close'].rolling(self.sma_period).mean(), 2)
            tl.save_csv(self.FOLDER_WITH_DATA, self.signal_stocks, df)
        else:
            return

    def calculate_momentum(self, asset: str, recalculate: bool = False):

        if asset == 'stocks':
            ticker, mom = self.signal_stocks, self.momentum_stocks
        elif asset == 'bonds':
            ticker, mom = self.signal_bonds, self.momentum_bonds
        else:
            raise ValueError("Incorrect value for 'asset' in 'calculate_momentum' func")

        df = tl.load_csv(ticker)
        if recalculate or 'Momentum_' + str(mom) not in df.keys():
            print(f"Calculate momentum for {ticker} with {mom} month-period")
            mom_list = []
            for day in range(len(df.Date)):

                if day != len(df)-1 and df.Date[day].month != df.Date[day+1].month:
                    find_month = df.Date[day] - rdelta(months=mom)
                    month_cls = df[(df.Date.dt.year == find_month.year) & (df.Date.dt.month == find_month.month)].Close

                    if len(month_cls) != 0:
                        mom_list.append(df.Close[day] / month_cls.iloc[-1])
                    else:
                        mom_list.append(None)

                elif day != 0 and df.Date[day].month != df.Date[day-1].month:
                    find_month = df.Date[day] - rdelta(months=mom)
                    month_opn = df[(df.Date.dt.year == find_month.year) & (df.Date.dt.month == find_month.month)].Open

                    if len(month_opn) != 0:
                        mom_list.append(df.Open[day] / month_opn.iloc[0])
                    else:
                        mom_list.append(None)

                else:
                    mom_list.append(None)
        else:
            return

        df['Momentum_' + str(mom)] = mom_list
        tl.save_csv(self.FOLDER_WITH_DATA, ticker, df)

    def data_sufficiency_check(self, start_index: int):

        data = self.all_tickers_data
        for index in range(start_index, len(self.trading_days)):
            stocks_sma = data[self.signal_stocks]['SMA_' + str(self.sma_period)][index]
            stocks_mom = data[self.signal_stocks]['Momentum_' + str(self.momentum_stocks)][index]
            bonds_mom = data[self.signal_bonds]['Momentum_' + str(self.momentum_bonds)][index]

            if math.isnan(stocks_sma) is False and math.isnan(stocks_mom) is False and math.isnan(bonds_mom) is False:
                break

        return index

    # Logical chain of functions --------------------------------------------------------------------------------------
    def what_port_need_now(self, day_number: int):

        # Если нужно получить сигналы по close, но рассчитать цены по open
        # day_number -= 1

        data = self.all_tickers_data
        stocks_price = data[self.signal_stocks][self.price][day_number]
        stocks_sma = data[self.signal_stocks]['SMA_' + str(self.sma_period)][day_number]
        stocks_mom = data[self.signal_stocks]['Momentum_' + str(self.momentum_stocks)][day_number]
        bonds_mom = data[self.signal_bonds]['Momentum_' + str(self.momentum_bonds)][day_number]

        # High_Risk
        if stocks_price > stocks_sma and stocks_mom >= 1 > bonds_mom:
            self.new_port = 'high_risk'

        # Mid_Risk
        elif stocks_price > stocks_sma and (stocks_mom < 1 or bonds_mom >= 1):
            self.new_port = 'mid_risk'

        # High_Safe bonds_mom < 1 stocks_mom < 1
        elif stocks_price <= stocks_sma and bonds_mom < 1:
            self.new_port = 'high_save'

        # Mid_Safe
        elif stocks_price <= stocks_sma:
            self.new_port = 'mid_save'

        else:
            print(f"Stock: {stocks_price}, SMA: {stocks_sma}, Stock_mom: {stocks_mom}, Bond_mom: {bonds_mom}")
            print(f"{self.trading_days[day_number]} date does not fall under the rebalance conditions")


def working_with_capital(test_port, day_number: int):
    test_port.what_port_need_now(day_number)
    if test_port.capital_not_placed:
        test_port.dont_have_any_port(day_number)
    else:
        capital = test_port.calculate_capital_at_rebalance_day(day_number)
        capital -= test_port.rebalance_commissions(capital, day_number)
        test_port.rebalance_port(capital, day_number)


def start(test_port, recalculate_variables: bool = False) -> (pd.DataFrame, pd.DataFrame, str):
    """

    :param test_port: initialized class
    :param recalculate_variables: recalculate SMA and momentum values
    :return:
    """

    # Preprocessing data
    test_port.download_data()
    test_port.calculate_sma(recalculate=recalculate_variables)
    test_port.calculate_momentum(asset='stocks', recalculate=recalculate_variables)
    test_port.calculate_momentum(asset='bonds', recalculate=recalculate_variables)
    start_date, end_date = test_port.find_oldest_newest_dates()
    test_port.cut_data_by_dates(start_date, end_date)
    test_port.create_columns_for_strategy_dict()

    # Iterate_trading_days
    start_index = test_port.start_day_index()
    start_index = test_port.data_sufficiency_check(start_index)
    for day_number in range(start_index, len(test_port.trading_days)):

        if test_port.rebalance_at == 'close':

            if day_number != len(test_port.trading_days) - 1:

                if test_port.rebalance == 'monthly' and test_port.trading_days[day_number].month != \
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

                if test_port.rebalance == 'monthly' and test_port.trading_days[day_number - 1].month != \
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

    chart_name = 'SmaMomentumPort ' + \
                 '(' + test_port.rebalance + ') ' + \
                 '(' + 'by ' + test_port.rebalance_at + ') ' + \
                 '(' + f"Signals {test_port.signal_stocks} {test_port.signal_bonds}" + ') ' + \
                 '(' + f"Mom {test_port.momentum_stocks} {test_port.momentum_bonds}" + ') ' + \
                 '(' + f"SMA {test_port.sma_period}" + ') '

    return df_strategy, df_yield_by_years, chart_name


if __name__ == "__main__":
    # Обычный портфель обходится без плеча. Но если садить что-то круче DIA, лучше взять хотя бы 1.3 плечо на бонды
    portfolios = {
        'high_risk':
            {'FBT': .15 * .8, 'FDN': .20 * .8, 'IGV': .20 * .8, 'IHI': .15 * .8, 'ITA': .30 * .8, 'TLT': .2 * 1.3},
        'mid_risk':
            {'FBT': .15 * .8, 'FDN': .20 * .8, 'IGV': .20 * .8, 'IHI': .15 * .8, 'ITA': .30 * .8, 'TLT': .2 * 1.3},
        'mid_save':
            {'TLT': .25 * 1.3, 'GLD': .25 * 1.3, 'FBT': .15 * .5, 'FDN': .20 * .5, 'IGV': .20 * .5, 'IHI': .15 * .5,
             'ITA': .30 * .5},
        'high_save':
            {'TLT': .8 * 1.15, 'GLD': .2}
    }
    test_port = SMAandMomentum(
        balance_start=5_000,
        portfolios=portfolios,
        rebalance='monthly',
        trade_rebalance_at='close',
        date_start=datetime(2006, 9, 5),
        date_end=datetime.now(),
        signal_stocks='SPY',
        signal_bonds='TLT',
        benchmark='QQQ',
        sma_period=200
    )

    df_strategy, df_yield_by_years, chart_name = start(test_port)

    tl.plot_capital_plotly(test_port.FOLDER_WITH_IMG + chart_name,
                           list(df_strategy.Date),
                           list(df_strategy.Capital),
                           df_yield_by_years,
                           portfolios)

    tl.save_csv(test_port.FOLDER_TO_SAVE,
                chart_name + str(test_port.ports_tickers()),
                df_strategy)

""" Ports
portfolios = {
        'high_risk':
            {'DJIndex': .8, 'Treasures': .2},
        'mid_risk':
            {'DJIndex': .8, 'Treasures': .2 * 1.3},
        'mid_save':
            {'DJIndex': .5, 'Treasures': .5 * 1.3},
        'high_save':
            {'Treasures': 1.0}
    }
    
portfolios = {
    'high_risk':
        {'FBT': .15*.8, 'FDN': .20*.8, 'IGV': .20*.8, 'IHI': .15*.8, 'ITA': .30*.8, 'TLT': .2 * 1.3},
    'mid_risk':
        {'FBT': .15*.8, 'FDN': .20*.8, 'IGV': .20*.8, 'IHI': .15*.8, 'ITA': .30*.8, 'TLT': .2 * 1.3},
    'mid_save':
        {'TLT': .25 * 1.3, 'GLD': .25 * 1.3, 'FBT': .15*.5, 'FDN': .20*.5, 'IGV': .20*.5, 'IHI': .15*.5, 'ITA': .30*.5},
    'high_save':
        {'TLT': .8 * 1.15, 'GLD': .2}
    }
"""
