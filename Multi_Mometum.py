from datetime import datetime
from dateutil.relativedelta import relativedelta as rdelta
from libs import trading_lib as tl

import base_ports
import pandas as pd


class MultiMomentum(base_ports.BasePortfolio):
    __slots__ = [
        'absolute_mom',
        'signal_port',
        'multi_momentums'
    ]

    def __init__(self,
                 portfolios: dict,
                 absolute_mom: str,
                 signal_port: dict,
                 multi_momentums: dict,
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
                 value_withdraw_or_depo: float = -0.67):
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
        self.absolute_mom = absolute_mom
        self.signal_port = signal_port
        self.multi_momentums = multi_momentums
        self.unique_tickers.update([self.signal_stocks, self.signal_bonds])

        for port_name in portfolios.keys():
            assert port_name in ['high_risk', 'mid_risk', 'mid_save', 'high_save'], \
                f"Incorrect value in portfolios names - {port_name}"
