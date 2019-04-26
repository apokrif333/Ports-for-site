from libs import trading_lib as tl

import Target_Vol

port = {'risk_on':
            {'FBT': 1.0},
        'risk_off':
            {'TLT': 1.0}
        }
portfolio = Target_Vol.TargetVolatility(portfolios=port)
df, df_1, name = Target_Vol.start(portfolio)

tl.plot_capital_plotly(
    portfolio.FOLDER_WITH_IMG + name,
    list(df.Date),
    list(df.Capital),
    df_1,
    port
)
