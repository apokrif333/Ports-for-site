from datetime import datetime
from dateutil.relativedelta import relativedelta as rdelta
from libs import trading_lib as tl

import numpy as np
import base_ports
import pandas as pd


class MultiMomentum(base_ports.BasePortfolio):
    __slots__ = [
        'absolute_mom',
        'signal_port',
        'momentums',
        'use_absolute_mom',

        'tickers_for_mom'
    ]

    def __init__(self,
                 portfolios: dict,
                 absolute_mom: str,
                 signal_port: dict,
                 momentums: dict,
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
                 use_absolute_mom: bool = True):
        """

        :param portfolios: {'port_name_1': {'ticker_1': share, 'ticker_N': share}, 'port_name_N': {'ticker_N': share}}. Portfolios names: 'high_risk', 'mid_risk', 'mid_save', 'high_save'
        :param absolute_mom:
        :param signal_port: {'port_name_1: 'ticker_1_for_signal', 'port_name_2: 'ticker_2_for_signal'...}
        :param momentums: {'int, month ago': weight for this mom in all mom weights, 'int, month ago': weight for this mom in all mom weights...}
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
        :param use_absolute_mom
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
        self.tickers_for_mom = [absolute_mom]
        self.tickers_for_mom += [value for value in signal_port.values()]

        # Strategy parameters
        self.absolute_mom = absolute_mom
        self.signal_port = signal_port
        self.momentums = momentums
        self.use_absolute_mom = use_absolute_mom

        for signal in self.signal_port.values():
            self.unique_tickers.update([signal])
        self.unique_tickers.update([self.absolute_mom])

    # Junior functions for calculations -------------------------------------------------------------------------------
    def calculate_momentum(self, recalc: bool = False):
        for ticker in set(self.tickers_for_mom):
            df = tl.load_csv(ticker)

            for mom in self.momentums.keys():

                if recalc or 'Momentum_' + str(mom) not in df.keys():
                    print(f"Calculate momentum for {ticker} with {mom} month-period")
                    mom_list = []
                    for day in range(len(df.Date)):

                        if day != len(df) - 1 and df.Date[day].month != df.Date[day + 1].month:
                            find_month = df.Date[day] - rdelta(months=mom)
                            month_cls = df[
                                (df.Date.dt.year == find_month.year) & (df.Date.dt.month == find_month.month)].Close

                            if len(month_cls) != 0:
                                mom_list.append(df.Close[day] / month_cls.iloc[-1])
                            else:
                                mom_list.append(None)

                        elif day != 0 and df.Date[day].month != df.Date[day - 1].month:
                            find_month = df.Date[day] - rdelta(months=mom)
                            month_opn = df[
                                (df.Date.dt.year == find_month.year) & (df.Date.dt.month == find_month.month)].Open

                            if len(month_opn) != 0:
                                mom_list.append(df.Open[day] / month_opn.iloc[0])
                            else:
                                mom_list.append(None)

                        else:
                            mom_list.append(None)

                else:
                    continue

                df['Momentum_' + str(mom)] = mom_list
            tl.save_csv(self.FOLDER_WITH_DATA, ticker, df)

    def data_sufficiency_check(self, start_index: int):

        data = self.all_tickers_data
        for index in range(start_index, len(self.trading_days)):
            check_list = []
            for ticker in set(self.tickers_for_mom):
                for mom in self.momentums.keys():
                    check_list.append(data[ticker]['Momentum_' + str(mom)][index])

            if True not in np.isnan(check_list):
                break

        return index

    # Logical chain of functions --------------------------------------------------------------------------------------
    def what_port_need_now(self, day_number: int):

        # Если нужно получить сигналы по close, но рассчитать цены по open
        # day_number -= 1

        data = self.all_tickers_data

        absolute = 2
        if self.use_absolute_mom:
            absolute = 0
            for mom in self.momentums.keys():
                absolute += data[self.absolute_mom]['Momentum_' + str(mom)][day_number] * self.momentums[mom]

        mom_list = [0] * len(self.signal_port)
        i = 0
        for ticker in self.signal_port.values():
            for mom in self.momentums.keys():
                mom_list[i] += data[ticker]['Momentum_' + str(mom)][day_number] * self.momentums[mom]
            i += 1

        # Risk_1
        if absolute > 1.0 and mom_list[1] < mom_list[0] > 1.0:
            self.new_port = list(self.signal_port.keys())[0]

        # Risk_2
        elif absolute > 1.0 and mom_list[0] <= mom_list[1] > 1.0:
            self.new_port = list(self.signal_port.keys())[1]

        # High_Safe
        elif absolute <= 1.0 or (mom_list[0] <= 1.0 and mom_list[1] <= 1.0 and mom_list[2] <= 1):
            self.new_port = list(self.signal_port.keys())[2]

        # Mid_Safe
        elif absolute <= 1.0 or (mom_list[0] <= 1.0 and mom_list[1] <= 1.0 and mom_list[2] > 1):
            self.new_port = list(self.signal_port.keys())[3]

        else:
            print(f"Absolute: {absolute}, 0: {mom_list[0]}, 1: { mom_list[1]}, 2: { mom_list[2]}, 3:{ mom_list[3]}")
            print(f"{self.trading_days[day_number]} date does not fall under the rebalance conditions")

        # print(self.trading_days[day_number])
        # print(self.new_port)
        # print(mom_list)


def working_with_capital(test_port, day_number: int):
    test_port.what_port_need_now(day_number)
    if test_port.capital_not_placed:
        test_port.dont_have_any_port(day_number)
    else:
        capital = test_port.calculate_capital_at_rebalance_day(day_number)
        capital -= test_port.rebalance_commissions(capital, day_number)
        test_port.rebalance_port(capital, day_number)


def start(test_port, recalculate_variables: bool = False) -> (pd.DataFrame, pd.DataFrame, str):
    # Preprocessing data
    test_port.download_data()
    test_port.calculate_momentum(recalc=recalculate_variables)
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

    chart_name = 'MultiMomentumPort ' + \
                 '(' + test_port.rebalance + ') ' + \
                 '(' + 'by ' + test_port.rebalance_at + ') '

    return df_strategy, df_yield_by_years, chart_name


if __name__ == "__main__":
    ports_names = [
        'Risk_1',
        'Risk_2',
        'High_Save',
        'Mid_Save'
    ]
    portfolios = {
        ports_names[0]:
            {'FBT': .15 * .8, 'FDN': .20 * .8, 'IGV': .20 * .8, 'IHI': .15 * .8, 'ITA': .30 * .8, 'TLT': .2 * 1.3},
        ports_names[1]:
            {'FBT': .15 * .8, 'FDN': .20 * .8, 'IGV': .20 * .8, 'IHI': .15 * .8, 'ITA': .30 * .8, 'TLT': .2 * 1.3},
        ports_names[2]:
            {'TLT': .8 * 1.3, 'GLD': .2 * 1.3},
        ports_names[3]:
            {'TLT': .3 * 1.3, 'GLD': .3 * 1.3, 'FBT': .15 * .4, 'FDN': .20 * .4, 'IGV': .20 * .4, 'IHI': .15 * .4,
             'ITA': .30 * .4},
    }
    signal_port = {
        ports_names[0]: 'SPY',
        ports_names[1]: 'SPY',
        ports_names[2]: 'TLT',
        ports_names[3]: 'TLT'
    }
    multi_mom = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
    }
    for key, _ in multi_mom.items():
        multi_mom[key] = 1 / len(list(multi_mom.keys()))

    absolute_mom = 'QQQ'

    test_port = MultiMomentum(portfolios=portfolios,
                              absolute_mom=absolute_mom,
                              signal_port=signal_port,
                              momentums=multi_mom,
                              use_absolute_mom=False,
                              date_start=datetime(2006, 8, 5),
                              benchmark='QQQ')

    df_strategy, df_yield_by_years, chart_name = start(test_port)

    tl.plot_capital_plotly(test_port.FOLDER_WITH_IMG + chart_name,
                           list(df_strategy.Date),
                           list(df_strategy.Capital),
                           df_yield_by_years,
                           portfolios)

    # tl.save_csv(test_port.FOLDER_TO_SAVE,
    #             chart_name + str(test_port.ports_tickers()),
    #             df_strategy)

""" Ports
    portfolios = {
        ports_names[0]:
            {'DJIndex': .8, 'Treasures': .2},
        ports_names[1]:
            {'DJIndex': .8, 'Treasures': .2},
        ports_names[2]:
            {'Treasures': 1.0},
        ports_names[3]:
            {'DJIndex': .4, 'Treasures': .6}
    }
    portfolios = {
        ports_names[0]:
            {'FBT': .15 * .8, 'FDN': .20 * .8, 'IGV': .20 * .8, 'IHI': .15 * .8, 'ITA': .30 * .8, 'TLT': .2 * 1.3},
        ports_names[1]:
            {'FBT': .15 * .8, 'FDN': .20 * .8, 'IGV': .20 * .8, 'IHI': .15 * .8, 'ITA': .30 * .8, 'TLT': .2 * 1.3},
        ports_names[2]:
            {'TLT': .8 * 1.3, 'GLD': .2 * 1.3},
        ports_names[3]:
            {'TLT': .3 * 1.3, 'GLD': .3 * 1.3, 'FBT': .15 * .4, 'FDN': .20 * .4, 'IGV': .20 * .4, 'IHI': .15 * .4,
             'ITA': .30 * .4},
    }
"""
