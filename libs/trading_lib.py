# Библиотека функций для трейдов
import pandas as pd
import numpy as np
import statistics as stat
import math
import time
import cmath
import os
import re
import matplotlib.pyplot as plt

from alpha_vantage.timeseries import TimeSeries
from yahoofinancials import YahooFinancials
from datetime import datetime
from plotly.offline import plot
from plotly import graph_objs as go


# Constants
ALPHA_KEY = 'FE8STYV4I7XHRIAI'

# Variables
default_data_dir = 'historical_data'  # Директория
start_date = datetime(1985, 1, 1)  # Для yahoo, alpha выкачает всю доступную историю
end_date = datetime.now()

# Globals
alpha_count = 0


# Блок форматированная данных ------------------------------------------------------------------------------------------
# Формат даты в сторку
def dt_to_str(date: datetime) -> str:
    return "%04d-%02d-%02d" % (date.year, date.month, date.day)


# Формат строки в дату
def str_to_dt(string: str) -> datetime:
    return datetime.strptime(string, '%Y-%m-%d')


# Формат листа со строками в лист с датами из файла
def str_list_to_date(file: pd.DataFrame):
    try:
        file["Date"] = pd.to_datetime(file["Date"], dayfirst=False)
    except ValueError:
        file["Date"] = pd.to_datetime(file["Date"], format='%d-%m-%Y')


# Округление числа и конвертация его во float
def number_to_float(n) -> float:
    if empty_check(n):
        return round(float(n), 2)
    else:
        return n


# Округление числа и конвертация его в int
def number_to_int(n) -> int:
    if empty_check(n):
        return int(round(float(n), 0))
    else:
        return n


# Не пустой ли объект?
def empty_check(n) -> bool:
    return n is not None and n != 0 and not cmath.isnan(n)


# Блок скачивания цен --------------------------------------------------------------------------------------------------
# Скачиваем нужные тикеры из альфы (не сплитованные цены)
def download_alpha(ticker: str, base_dir: str = default_data_dir) -> pd.DataFrame:
    data = None
    global alpha_count

    try:
        ts = TimeSeries(key=ALPHA_KEY, retries=0)
        data, meta_data = ts.get_daily(ticker, outputsize='full')
    except Exception as err:
        if 'Invalid API call' in str(err):
            print(f'AlphaVantage: ticker data not available for {ticker}')
            return pd.DataFrame({})
        elif 'TimeoutError' in str(err):
            print(f'AlphaVantage: timeout while getting {ticker}')
        else:
            print(f'AlphaVantage: {err}')

    if data is None or len(data.values()) == 0:
        print('AlphaVantage: no data for %s' % ticker)
        return pd.DataFrame({})

    prices = {}
    for key in sorted(data.keys(), key=lambda d: datetime.strptime(d, '%Y-%m-%d')):
        secondary_dic = data[key]
        date = datetime.strptime(key, '%Y-%m-%d')
        dic_with_prices(prices, ticker, date, secondary_dic['1. open'], secondary_dic['2. high'],
                        secondary_dic['3. low'], secondary_dic['4. close'], secondary_dic['5. volume'])

    frame = pd.DataFrame.from_dict(prices, orient='index', columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    save_csv(base_dir, ticker, frame, 'alpha')
    time.sleep(15 if alpha_count != 0 else 0)
    alpha_count += 1


# Скачиваем тикеры из яху (цены и дивиденды)
def download_yahoo(ticker: str, base_dir: str = default_data_dir):
    try:
        yf = YahooFinancials(ticker)
        data = yf.get_historical_price_data(dt_to_str(start_date), dt_to_str(end_date), 'daily')
    except Exception as err:
        print(f'Unable to read data for {ticker}: {err}')
        return pd.DataFrame({})

    if data.get(ticker) is None or data[ticker].get('prices') is None or \
            data[ticker].get('timeZone') is None or len(data[ticker]['prices']) == 0:
        print(f'Yahoo: no data for {ticker}')
        return pd.DataFrame({})

    prices = {}
    for rec in sorted(data[ticker]['prices'], key=lambda r: r['date']):
        date = datetime.strptime(rec['formatted_date'], '%Y-%m-%d')
        dic_with_prices(prices, ticker, date, rec['open'], rec['high'], rec['low'], rec['close'], rec['volume'])

    if 'dividends' in data[ticker]['eventsData']:
        for date, rec in sorted(data[ticker]['eventsData']['dividends'].items(), key=lambda r: r[0]):
            date = datetime.strptime(date, '%Y-%m-%d')
            dic_with_div(prices, ticker, date, rec['amount'])

    if 'splits' in data[ticker]['eventsData']:
        for date, rec in sorted(data[ticker]['eventsData']['splits'].items(), key=lambda r: r[0]):
            date = datetime.strptime(date, '%Y-%m-%d')
            print(f"{ticker} has split {rec['splitRatio']} for {date}")

    frame = pd.DataFrame.from_dict(prices, orient='index',
                                   columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividend'])
    save_csv(base_dir, ticker, frame, 'yahoo')


# Словарь с ценами
def dic_with_prices(prices: dict, ticker: str, date: datetime, open, high, low, close, volume, dividend=0):
    if date.weekday() > 5:
        print(f'Найден выходной в {ticker} на {date}')
        return

    open = number_to_float(open)
    high = number_to_float(high)
    low = number_to_float(low)
    close = number_to_float(close)
    volume = number_to_int(volume)

    error_price = (not empty_check(open)) or (not empty_check(high)) or (not empty_check(low)) or (
        not empty_check(close))
    error_vol = not empty_check(volume)

    if error_price:
        print(f'В {ticker} на {date} имеются пустые данные')
        return
    if error_vol:
        print(f'В {ticker} на {date} нет объёма')

    prices[date] = [open, high, low, close, volume, dividend]


# Добавляем дивиденды к словарю с ценами
def dic_with_div(prices: dict, ticker: str, date: datetime, amount: float):
    if date.weekday() > 5:
        print(f'Найден выходной в {ticker} на {date}')
        return

    dividend = amount
    error_price = not empty_check(dividend)

    if error_price:
        print(f'В {ticker} на {date} имеются пустые данные в дивидендах')
        return

    prices[date][len(prices[date]) - 1] = dividend


# Блок работы с файлами ------------------------------------------------------------------------------------------------
# Сохраняем csv файл
def save_csv(base_dir: str, file_name: str, data: pd.DataFrame, source: str = 'new_file'):
    path = os.path.join(base_dir)
    if not os.path.exists(path):
        os.makedirs(path)

    if source == 'alpha':
        print(f'{file_name} отработал через альфу')
        path = os.path.join(path, file_name + ' NonSplit' + '.csv')
    elif source == 'yahoo':
        print(f'{file_name} отработал через яху')
        path = os.path.join(path, file_name + '.csv')
        path = path.replace('^', '')
    elif source == 'new_file':
        print(f'Сохраняем файл с тикером {file_name}')
        path = os.path.join(path, file_name + '.csv')
    else:
        print(f'Неопознанный источник данных для {file_name}')

    if source == 'alpha' or source == 'yahoo':
        data.to_csv(path, index_label='Date')
    else:
        data.to_csv(path, index=False)


# Загружаем csv файл
def load_csv(ticker: str, base_dir: str=default_data_dir) -> pd.DataFrame:
    path = os.path.join(base_dir, str(ticker) + '.csv')
    file = pd.read_csv(path)
    if 'Date' in file.columns:
        str_list_to_date(file)
    return file


# Обрезаем файл согласно определённым датам
def correct_file_by_dates(file: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return file.loc[(file['Date'] >= start) & (file['Date'] <= end)]


# Блок финансовых метрик -----------------------------------------------------------------------------------------------
# Считаем CAGR
def cagr(date: list, capital: list) -> float:
    years = (date[-1].year + date[-1].month / 12) - (date[0].year + date[0].month / 12)
    cagr = ((capital[-1] / capital[0]) ** (1 / years) - 1) * 100
    return round(cagr, 2)


# Считаем годовое отклонение
def st_dev(capital: list) -> float:
    day_cng = []
    for i in range(len(capital)):
        if i == 0:
            day_cng.append(0)
        else:
            day_cng.append(capital[i] / capital[i - 1] - 1)
    return stat.stdev(day_cng) * math.sqrt(252) if stat.stdev(day_cng) != 0 else 999


# Считаем предельную просадку
def draw_down(capital: list) -> float:
    high = 0
    down = []
    for i in range(len(capital)):
        if capital[i] > high:
            high = capital[i]
        down.append((capital[i] / high - 1) * 100)
    return min(down) if min(down) != 0 else 1


# Блок прочих функций --------------------------------------------------------------------------------------------------
# Ищем самую молодую дату
def newest_date_search(f_date: datetime, *args: datetime) -> datetime:
    newest_date = f_date
    for arg in args:
        if arg > newest_date:
            newest_date = arg
    return newest_date


# Ищем самую старую дату
def oldest_date_search(f_date: datetime, *args: datetime) -> datetime:
    oldest_date = f_date
    for arg in args:
        if arg < oldest_date:
            oldest_date = arg
    return oldest_date


# Рисовалка ------------------------------------------------------------------------------------------------------------
# Выводим график капитала с таблицей
def plot_capital(date: list, capital: list):
    capital = [abs(x) for x in capital]

    high = 0
    down = []
    for i in range(len(capital)):
        if capital[i] > high:
            high = capital[i]
        down.append((capital[i] / high - 1) * -100)

    names = ['Start Balance', 'End Balance', 'CAGR', 'DrawDown', 'StDev', 'Sharpe', 'MaR', 'SM']
    values = []
    values.append(capital[0])
    values.append(round(capital[-1], 0))
    values.append(cagr(date, capital))
    values.append(round(draw_down(capital), 2))
    values.append(round(st_dev(capital) * 100, 2))
    values.append(round(values[2] / values[4], 2))
    values.append(round(abs(values[2] / values[3]), 2))
    values.append(round(values[5] * values[6], 2))

    while len(names) % 4 != 0:
        names.append('')
        values.append('')

    table_metric = pd.DataFrame({'h1': names[:int(len(names) / 4)],
                                 'v1': values[:int(len(values) / 4)],
                                 'h2': names[int(len(names) / 4):int(len(names) / 2)],
                                 'v2': values[int(len(values) / 4):int(len(values) / 2)],
                                 'h3': names[int(len(names) / 2):int(len(names) * 0.75)],
                                 'v3': values[int(len(values) / 2):int(len(values) * 0.75)],
                                 'h4': names[int(len(names) * 0.75):len(names)],
                                 'v4': values[int(len(values) * 0.75):len(values)],
                                 })

    fig = plt.figure(figsize=(12.8, 8.6), dpi=80)

    ax1 = fig.add_subplot(6, 1, (1, 5))
    ax1.plot(date, down, dashes=[6, 4], color="darkgreen", alpha=0.5)
    ax1.set_ylabel('Просадки')

    ax2 = ax1.twinx()
    ax2.plot(date, capital)
    ax2.set_ylabel('Динамика капитала')

    tx1 = fig.add_subplot(6, 1, 6, frameon=False)
    tx1.axis("off")
    tx1.table(cellText=table_metric.values, loc='lower center')

    plt.show()


# Выводим тотал-график капитала с таблицами в плотли
def plot_capital_plotly(chart_name: str, date: list, capital: list, show_table: pd.DataFrame, ports: dict):
    high = 0
    down = []
    for i in range(len(capital)):
        if capital[i] > high:
            high = capital[i]
        down.append((capital[i] / high - 1) * -100)

    names1, names2 = ['Start Balance', 'End Balance', 'CAGR', 'DrawDown'], ['StDev', 'Sharpe', 'MaR', 'SM']
    values1, values2 = [], []
    values1.append(capital[0])
    values1.append(round(capital[-1], 0))
    values1.append(cagr(date, capital))
    values1.append(round(draw_down(capital), 2))
    values2.append(round(st_dev(capital) * 100, 2))
    values2.append(round(values1[2] / values2[0], 2))
    values2.append(round(abs(values1[2] / values1[3]), 2))
    values2.append(round(values2[1] * values2[2], 2))

    ports_values = []
    for i in list(ports.values()):
        ports_values.append(str(i))

    trace1 = go.Scatter(
        x=date,
        y=capital,
        mode='lines',
        line=dict(color='#1f77b4'),
        name='Port Capital'
    )
    trace2 = go.Scatter(
        x=date,
        y=down,
        mode='lines',
        line=dict(color='wheat', dash='dash'),
        name='DrawDown',
        yaxis='y2'
    )
    trace3 = go.Table(
        domain=dict(x=[0, 0.3],
                    y=[0, 0.2]),
        header=dict(
            values=[['<b>NAMES</b>'], ['<b>VALUES</b>'],
                    ['<b>NAMES</b>'], ['<b>VALUES</b>']],
            fill=dict(color='gray'),
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[names1, values1,
                    names2, values2],
            line=dict(color='#7D7F80'),
            fill=dict(color=['wheat', 'white',
                             'wheat', 'white']),
            align=['left', 'center',
                   'left', 'center']
        )
    )
    trace4 = go.Table(
        domain=dict(x=[0.35, 0.65],
                    y=[0, 0.2]),
        header=dict(
            values=show_table.columns,
            font=dict(color='white', size=12),
            fill=dict(color='gray'),
        ),
        cells=dict(
            values=[show_table[c].tolist() for c in show_table.columns],
            fill=dict(color=['palegreen', 'wheat', 'white',
                             'wheat', 'white']),
        )
    )
    trace5 = go.Table(
        domain=dict(x=[0.7, 1.0],
                    y=[0, 0.2]),
        header=dict(
            values=[['Ports names'], ['Ports contents']],
            fill=dict(color='gray'),
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[[*ports], ports_values],
            line=dict(color='#7D7F80'),
            fill=dict(color=['wheat', 'white']),
            # align=['left', 'center']
        )
    )

    plt_data = [trace1, trace2, trace3, trace4, trace5]

    plt_layout = go.Layout(
        title=chart_name.split('/')[-1],
        yaxis=dict(
            title='Port Capital',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            type='log',
            autorange=True,
            domain=[0.25, 1.0]
        ),
        yaxis2=dict(
            range=[0, 50],
            title='DrawDown',
            titlefont=dict(color='wheat'),
            tickfont=dict(color='wheat'),
            overlaying='y',
            side='right'
        ),
    )

    fig = go.Figure(
        data=plt_data,
        layout=plt_layout
    )
    plot(fig, show_link=False, filename=chart_name + '.html')


# Выводим динамику эквити в плотли
def capital_chart_plotly(chart_name: str, date: list, capital: list):
    trace1 = go.Scatter(
        x=date,
        y=capital,
        mode='lines',
        line=dict(color='#1f77b4'),
        name='Port Capital'
    )

    plt_layout = go.Layout(
        title=chart_name,
        yaxis=dict(
            title='Port Capital',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            type='log',
            autorange=True,
        ),
    )

    fig = go.Figure(
        data=[trace1],
        layout=plt_layout
    )
    plot(fig, show_link=False, filename=chart_name + '_capital' + '.html')


# Выводим динамику просадок в плотли
def drawdown_chart_plotly(chart_name: str, date: list, capital: list):
    high = 0
    down = []
    for i in range(len(capital)):
        if capital[i] > high:
            high = capital[i]
        down.append((capital[i] / high - 1) * -100)

    trace1 = go.Scatter(
        x=date,
        y=down,
        mode='lines',
        line=dict(color='#1f77b4'),
        name='Port DrawDown'
    )

    plt_layout = go.Layout(
        title=chart_name,
        yaxis=dict(
            range=[0, 50],
            title='DrawDown',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
        ),
    )

    fig = go.Figure(
        data=[trace1],
        layout=plt_layout
    )
    plot(fig, show_link=False, filename=chart_name + '_drawdown' + '.html')


# Выводим таблицу с показателями в плотли
def portfolio_perform_table_plotly(chart_name: str, date: list, capital: list):
    high = 0
    down = []
    for i in range(len(capital)):
        if capital[i] > high:
            high = capital[i]
        down.append((capital[i] / high - 1) * -100)

    names1, names2 = ['Start Balance', 'End Balance', 'CAGR', 'DrawDown'], ['StDev', 'Sharpe', 'MaR', 'SM']
    values1, values2 = [], []
    values1.append(capital[0])
    values1.append(round(capital[-1], 0))
    values1.append(cagr(date, capital))
    values1.append(round(draw_down(capital), 2))
    values2.append(round(st_dev(capital) * 100, 2))
    values2.append(round(values1[2] / values2[0], 2))
    values2.append(round(abs(values1[2] / values1[3]), 2))
    values2.append(round(values2[1] * values2[2], 2))

    trace1 = go.Table(
        header=dict(
            values=[['<b>NAMES</b>'], ['<b>VALUES</b>'],
                    ['<b>NAMES</b>'], ['<b>VALUES</b>']],
            fill=dict(color='gray'),
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[names1, values1,
                    names2, values2],
            line=dict(color='#7D7F80'),
            fill=dict(color=['wheat', 'white',
                             'wheat', 'white']),
            align=['left', 'center',
                   'left', 'center']
        )
    )

    fig = go.Figure(data=[trace1])
    plot(fig, show_link=False, filename=chart_name + '_perform_table' + '.html')


# Выводим таблицу с показателями по годам, по сравнению с бенчем в плотли
def portfolio_by_years_table_plotly(chart_name: str, show_table: pd.DataFrame,):
        trace = go.Table(
            header=dict(
                values=show_table.columns,
                font=dict(color='white', size=12),
                fill=dict(color='gray'),
            ),
            cells=dict(
                values=[show_table[c].tolist() for c in show_table.columns],
                fill=dict(color=['palegreen', 'wheat', 'white',
                                 'wheat', 'white']),
            )
        )

        fig = go.Figure(data=[trace])
        plot(fig, show_link=False, filename=chart_name + '_by_year' + '.html')
