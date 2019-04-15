from datetime import datetime

import base_ports


class TargetVolatility(base_ports.BasePortfolio):
    __slots__ = [
        'vol_calc_period',
        'vol_calc_range',
        'vol_target',
        'use_margin'
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

        :param portfolios: {'port_name_1': {'ticker_1': share, 'ticker_N': share}, 'port_name_N': {'ticker_N': share}}
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

        # Strategy parameters
        self.vol_calc_period = vol_calc_period
        self.vol_calc_range = vol_calc_range
        self.vol_target = vol_target
        self.use_margin = use_margin

    def create_columns_for_strategy_dict(self, risk_on: dict, risk_off: dict):

        first_ticker = next(iter(self.all_tickers_data))
        self.trading_days = self.all_tickers_data[first_ticker]['Date']
        number_of_rows = len(self.all_tickers_data[first_ticker])

        self.strategy_data['Date'] = self.trading_days
        for t in risk_on.keys():
            self.strategy_data['On_Ticker_' + t] = [0] * number_of_rows
            self.strategy_data['On_Price_' + t] = self.all_tickers_data[t][self.price]
            self.strategy_data['On_Shares_' + t] = [0] * number_of_rows
            self.strategy_data['On_Dividend_' + t] = self.all_tickers_data[t]['Dividend']

        for t in risk_off.keys():
            self.strategy_data['Off_Ticker_' + t] = [0] * number_of_rows
            self.strategy_data['Off_Price_' + t] = self.all_tickers_data[t][self.price]
            self.strategy_data['Off_Shares_' + t] = [0] * number_of_rows
            self.strategy_data['Off_Dividend_' + t] = self.all_tickers_data[t]['Dividend']
        self.strategy_data['Cash'] = [0] * number_of_rows
        self.strategy_data['Capital'] = [0] * number_of_rows
        self.strategy_data['InOutCash'] = [0] * number_of_rows

        print(f"self.strategy_data columns: {list(self.strategy_data.keys())}")


if __name__ == "__main__":
    portfolios = {'Risk_On':
                      {'SPY': 1.0},
                  'Risk_Off':
                      {'TLT': 1.0}
                  }
    test_port = TargetVolatility(portfolios=portfolios)

    # Preprocessing data
    test_port.download_data()
    start_date, end_date = test_port.find_oldest_newest_dates()
    test_port.cut_data_by_dates(start_date, end_date)
    test_port.create_columns_for_strategy_dict(portfolios['Risk_On'], portfolios['Risk_Off'])


    """
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

                elif day_number == len(self.trading_days) - 1 and test_port.forsed_rebalance:
                    test_port.change_port(day_number)

                elif test_port.capital_not_placed is False:
                    test_port.typical_day(day_number)
    """
