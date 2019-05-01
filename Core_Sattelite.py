from datetime import datetime
from libs import trading_lib as tl

import base_ports, SMA_and_Mom, Target_Vol
import pandas as pd


def working_with_capital(day_number: int):
    vol_capital = 0
    mom_capital = 0

    # Calc capital for core_port
    core_port.calculate_current_vol(day_number)
    if core_port.capital_not_placed:
        core_port.dont_have_any_port(day_number)
    else:
        vol_capital = core_port.calculate_capital_at_rebalance_day(day_number)

    # Calc capital for satelt_port
    satelt_port.what_port_need_now(day_number)
    if satelt_port.capital_not_placed:
        satelt_port.dont_have_any_port(day_number)
    else:
        mom_capital = satelt_port.calculate_capital_at_rebalance_day(day_number)

    # Rebalance both ports
    if (vol_capital and mom_capital) > 0:
        capital = vol_capital + mom_capital

        vol_capital = capital * core_sattelite['core']
        vol_capital -= core_port.rebalance_commissions(vol_capital, day_number)
        core_port.rebalance_port(vol_capital, day_number)

        mom_capital = capital * core_sattelite['sattelite']
        mom_capital -= satelt_port.rebalance_commissions(mom_capital, day_number)
        satelt_port.rebalance_port(mom_capital, day_number)

    elif (vol_capital == 0 and mom_capital != 0) or (vol_capital != 0 and mom_capital == 0):
        raise ArithmeticError('The capital of one port is placed, but the other is not')


def make_charts_for_site():
    tl.capital_chart_plotly(core_port.FOLDER_WITH_IMG + 'CoreSattelite',
                            list(both_strategies_df.Date),
                            list(both_strategies_df.Capital))
    tl.drawdown_chart_plotly(core_port.FOLDER_WITH_IMG + 'CoreSattelite',
                             list(both_strategies_df.Date),
                             list(both_strategies_df.Capital))
    tl.portfolio_perform_table_plotly(core_port.FOLDER_WITH_IMG + 'CoreSattelite',
                                      list(both_strategies_df.Date),
                                      list(both_strategies_df.Capital))
    tl.portfolio_by_years_table_plotly(core_port.FOLDER_WITH_IMG + 'CoreSattelite',
                                       df_yield_by_years)


def make_json_for_site():
    for ticker in core_ports['risk_on'].keys():
        core_ports['risk_on'][ticker] = round(
            core_ports['risk_on'][ticker] * core_port.cur_leverage * core_sattelite['core'], 2)
    for ticker in core_ports['risk_off'].keys():
        leverage = (1 - core_port.cur_leverage) if core_port.cur_leverage < 1 else 0
        core_ports['risk_off'][ticker] = round(
            core_ports['risk_off'][ticker] * leverage * core_sattelite['core'], 2)
    for ticker in satelt_ports[satelt_port.old_port]:
        satelt_ports[satelt_port.old_port][ticker] = satelt_ports[satelt_port.old_port][ticker] * core_sattelite[
            'sattelite']

    allocation = pd.DataFrame({'Core_Target_Vol': core_ports,
                               'Sattelite': satelt_ports[satelt_port.old_port]
                               })
    allocation.to_json(path_or_buf=core_port.FOLDER_TO_SAVE + '/allocation.json')


if __name__ == "__main__":

    # Portfolios
    core_ports = {
        'risk_on':
            {'SPY': .4, 'DIA': .6},
        'risk_off':
            {'TLT': 1.0}
    }

    satelt_ports = {
        'high_risk':
            {'QQQ': .6, 'FBT': .4},
        'mid_risk':
            {'DIA': .6, 'XLP': .4},
        'mid_save':
            {'TLT': .6, 'SPY': .4},
        'high_save':
            {'TLT': .8, 'GLD': .2}
    }

    # Base parameters
    core_sattelite = {'core': .3, 'sattelite': .7}
    start_capital = 100_000
    date_start = datetime(2007, 12, 30)
    date_end = datetime.now()
    rebalance = 'monthly'
    trade_rebalance_at = 'close'
    benchmark = 'QQQ'
    forsed_rebalance = False

    # core_port preprocessing
    core_port = Target_Vol.TargetVolatility(
        portfolios=core_ports,
        balance_start=int(start_capital * core_sattelite['core']),
        date_start=date_start,
        date_end=date_end,
        rebalance=rebalance,
        trade_rebalance_at=trade_rebalance_at,
        benchmark=benchmark,
        forsed_rebalance=forsed_rebalance
    )
    core_port.download_data()
    vol_start_date, vol_end_date = core_port.find_oldest_newest_dates()

    # satelt_port preprocessing
    satelt_port = SMA_and_Mom.SMAandMomentum(
        portfolios=satelt_ports,
        balance_start=int(start_capital * core_sattelite['sattelite']),
        date_start=date_start,
        date_end=date_end,
        rebalance=rebalance,
        trade_rebalance_at=trade_rebalance_at,
        benchmark=benchmark,
        forsed_rebalance=forsed_rebalance
    )
    satelt_port.download_data()
    satelt_port.calculate_sma()
    satelt_port.calculate_momentum(asset='stocks')
    satelt_port.calculate_momentum(asset='bonds')
    mom_start_date, mom_end_date = satelt_port.find_oldest_newest_dates()

    # Finishing both preprocess
    start_date, end_date = max(vol_start_date, mom_start_date), min(vol_end_date, mom_end_date)
    core_port.cut_data_by_dates(start_date, end_date)
    core_port.create_columns_for_strategy_dict()
    satelt_port.cut_data_by_dates(start_date, end_date)
    satelt_port.create_columns_for_strategy_dict()

    # Iterate_trading_days
    start_index = core_port.start_day_index()
    start_index = core_port.data_sufficiency_check(start_index)
    start_index = satelt_port.data_sufficiency_check(start_index)
    for day_number in range(start_index, len(core_port.trading_days)):

        if trade_rebalance_at == 'close':

            if day_number != len(core_port.trading_days) - 1:

                if core_port.rebalance == 'monthly' and core_port.trading_days[day_number].month != \
                        core_port.trading_days[day_number + 1].month:
                    working_with_capital(day_number)

                elif core_port.rebalance == 'quarterly' and core_port.trading_days[day_number].month in (3, 6, 9, 12) and \
                        core_port.trading_days[day_number + 1].month in (4, 7, 10, 1):
                    working_with_capital(day_number)

                elif core_port.rebalance == 'annual' and core_port.trading_days[day_number].year != \
                        core_port.trading_days[day_number + 1].year:
                    working_with_capital(day_number)

                elif core_port.capital_not_placed is False:
                    core_port.typical_day(day_number)
                    satelt_port.typical_day(day_number)

            elif forsed_rebalance:
                working_with_capital(day_number)

            elif core_port.capital_not_placed is False:
                core_port.typical_day(day_number)
                satelt_port.typical_day(day_number)

        elif trade_rebalance_at == 'open':

            if day_number != 0:

                if core_port.rebalance == 'monthly' and core_port.trading_days[day_number - 1].month != \
                        core_port.trading_days[day_number].month:
                    working_with_capital(day_number)

                elif core_port.rebalance == 'quarterly' and core_port.trading_days[day_number - 1].month in (3, 6, 9, 12) and \
                        core_port.trading_days[day_number].month in (4, 7, 10, 1):
                    working_with_capital(day_number)

                elif core_port.rebalance == 'annual' and core_port.trading_days[day_number - 1].year != \
                        core_port.trading_days[day_number].year:
                    working_with_capital(day_number)
    
                elif day_number == len(core_port.trading_days) - 1 and forsed_rebalance:
                    working_with_capital(day_number)

                elif core_port.capital_not_placed is False:
                    core_port.typical_day(day_number)
                    satelt_port.typical_day(day_number)

    # Transform data
    df_strategy_vol = pd.DataFrame.from_dict(core_port.strategy_data)
    df_strategy_vol = df_strategy_vol[df_strategy_vol.Capital != 0]
    df_strategy_vol['Capital'] = df_strategy_vol.Capital.astype(int)

    df_strategy_mom = pd.DataFrame.from_dict(satelt_port.strategy_data)
    df_strategy_mom = df_strategy_mom[df_strategy_mom.Capital != 0]
    df_strategy_mom['Capital'] = df_strategy_mom.Capital.astype(int)

    both_strategies_df = df_strategy_vol.merge(df_strategy_mom, left_on='Date', right_on='Date')
    both_strategies_df['Capital'] = both_strategies_df['Capital_x'] + both_strategies_df['Capital_y']

    df_yield_by_years = core_port.df_yield_std_by_every_year(both_strategies_df)

    chart_name = 'CoreSattelitePort ' + \
                 '(' + rebalance + ') ' + \
                 '(' + 'by ' + trade_rebalance_at + ') ' + \
                 '(' + f"VolTarget {core_port.vol_target}" + ') ' + \
                 '(' + f"Period_Range {core_port.vol_calc_period} {core_port.vol_calc_range}" + ') ' + \
                 '(' + f"Mom {satelt_port.momentum_stocks} {satelt_port.momentum_bonds}" + ') ' + \
                 '(' + f"SMA {satelt_port.sma_period}" + ') '

    tl.plot_capital_plotly(core_port.FOLDER_WITH_IMG + chart_name,
                           list(both_strategies_df.Date),
                           list(both_strategies_df.Capital),
                           df_yield_by_years,
                           {**core_ports, **satelt_ports})

    tl.save_csv(core_port.FOLDER_TO_SAVE,
                chart_name,
                both_strategies_df)

    # make_charts_for_site()
    # make_json_for_site()
