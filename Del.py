from datetime import datetime
from libs import trading_lib as tl

import pandas as pd
import SMA_and_Mom

port = {'high_risk':
            {'VFINX': .5, 'VEIEX': .5}
        }
port_class = SMA_and_Mom.SMAandMomentum(portfolios=port,
                                        date_start=datetime(1980, 1, 1),
                                        signal_stocks='VFINX',
                                        signal_bonds='VEIEX',
                                        benchmark='VFINX')
port_class.download_data()
start_date, end_date = port_class.find_oldest_newest_dates()
port_class.cut_data_by_dates(start_date, end_date)

port_class.all_tickers_data['VFINX_VEIEX'] = {}
port_class.all_tickers_data['VFINX_VEIEX']['Date'] = port_class.all_tickers_data['VFINX']['Date']
port_class.all_tickers_data['VFINX_VEIEX']['Close'] = \
    port_class.all_tickers_data['VFINX']['Close'] / port_class.all_tickers_data['VEIEX']['Close']
sma = port_class.all_tickers_data['VFINX_VEIEX']['Close']
port_class.all_tickers_data['VFINX_VEIEX']['SMA_200'] = round(sma.rolling(200).mean(), 2)

df = pd.DataFrame.from_dict(port_class.all_tickers_data['VFINX_VEIEX'])
tl.save_csv(port_class.FOLDER_WITH_DATA, 'VFINX_VEIEX', df)
