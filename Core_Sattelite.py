from datetime import datetime
from dateutil.relativedelta import relativedelta as rdelta
from libs import trading_lib as tl

import base_ports, SMA_and_Mom, Target_Vol
import numpy as np
import pandas as pd


if __name__ == "__main__":

    vol_port = {
        'Risk_On':
            {'SPY': .4, 'DIA': .6},
        'Risk_Off':
            {'TLT': 1.0}
    }

    sma_mom_port = {
        'high_risk':
            {'QQQ': .6, 'FBT': .4},
        'mid_risk':
            {'DIA': .6, 'XLP': .4},
        'mid_save':
            {'TLT': .6, 'SPY': .4},
        'high_save':
            {'TLT': .8, 'GLD': .2}
    }

    start_capital = 100_000
    core_sattelite = {'core': .3, 'sattelite': .7}

    # Vol_port_preprocessing
    vol_port = Target_Vol.TargetVolatility(portfolios=vol_port,
                                           balance_start=int(start_capital * core_sattelite['core']),
                                           date_start=datetime(2007, 11, 10))
    vol_port.download_data()
    vol_start_date, vol_end_date = vol_port.find_oldest_newest_dates()

    # SMA_mom_port_preprocessing
    sma_mom_port = SMA_and_Mom.SMAandMomentum(portfolios=sma_mom_port,
                                              balance_start=int(start_capital * core_sattelite['sattelite']))
    sma_mom_port.download_data()
    sma_mom_port.calculate_sma()
    sma_mom_port.calculate_momentum(asset='stocks')
    sma_mom_port.calculate_momentum(asset='bonds')
    mom_start_date, mom_end_date = sma_mom_port.find_oldest_newest_dates()

    # Finishing both preprocess
    start_date, end_date = max(vol_start_date, mom_start_date), min(vol_end_date, mom_end_date)
    vol_port.cut_data_by_dates(start_date, end_date)
    vol_port.create_columns_for_strategy_dict()
    sma_mom_port.cut_data_by_dates(start_date, end_date)
    sma_mom_port.create_columns_for_strategy_dict()

    for day_number in range(len(vol_port.trading_days)):
        if day_number != len(vol_port.trading_days) - 1:

            if vol_port.rebalance == 'monthly' and vol_port.trading_days[day_number].month != \
                    vol_port.trading_days[day_number + 1].month and vol_port.vol_calc_period == 'month' and \
                    (vol_port.trading_days[day_number] - rdelta(months=vol_port.vol_calc_range) >=
                     vol_port.trading_days[0]):
                mom_capital = 0
                vol_capital = 0

                # SMA_Mom
                sma_mom_port.what_port_need_now(day_number)
                if sma_mom_port.capital_not_placed:
                    sma_mom_port.dont_have_any_port(day_number)
                else:
                    mom_capital = sma_mom_port.calculate_capital_at_rebalance_day(day_number)

                #Target_Vol
                vol_port.calculate_current_vol(day_number)
                if vol_port.capital_not_placed:
                    vol_port.dont_have_any_port(day_number)
                else:
                    vol_capital = vol_port.calculate_capital_at_rebalance_day(day_number)

                if (mom_capital and vol_capital) > 0:
                    capital = mom_capital + vol_capital
                    mom_capital = capital * core_sattelite['sattelite']
                    mom_capital -= sma_mom_port.rebalance_commissions(mom_capital, day_number)
                    sma_mom_port.rebalance_port(mom_capital, day_number)

                    vol_capital = capital * core_sattelite['core']
                    vol_capital -= vol_port.rebalance_commissions(vol_capital, day_number)
                    vol_port.rebalance_port(vol_capital, day_number)

            elif vol_port.capital_not_placed is False or sma_mom_port.capital_not_placed is False:

                if vol_port.capital_not_placed is False:
                    vol_port.typical_day(day_number)

                if sma_mom_port.capital_not_placed is False:
                    sma_mom_port.typical_day(day_number)

        elif vol_port.capital_not_placed is False:
            vol_port.typical_day(day_number)

        elif sma_mom_port.capital_not_placed is False:
            sma_mom_port.typical_day(day_number)

    # Transform data
    df_strategy_vol = pd.DataFrame.from_dict(vol_port.strategy_data)
    df_strategy_vol = df_strategy_vol[df_strategy_vol.Capital != 0]
    df_strategy_vol['Capital'] = df_strategy_vol.Capital.astype(int)

    df_strategy_mom = pd.DataFrame.from_dict(sma_mom_port.strategy_data)
    df_strategy_mom = df_strategy_mom[df_strategy_mom.Capital != 0]
    df_strategy_mom['Capital'] = df_strategy_mom.Capital.astype(int)

    new_df = pd.DataFrame({})
    new_df['Date'] = df_strategy_vol['Date']
    new_df['Capital'] = df_strategy_mom['Capital'] + df_strategy_vol['Capital']

    df_yield_by_years = vol_port.df_yield_std_by_every_year(new_df)
    print(df_yield_by_years )

    tl.plot_capital_plotly(vol_port.FOLDER_WITH_IMG + 'Portfolio_Momentum',
                           list(new_df.Date),
                           list(new_df.Capital),
                           df_yield_by_years,
                           {})

    tl.save_csv(vol_port.FOLDER_TO_SAVE,
                f"InvestModel_EQ",
                new_df)
