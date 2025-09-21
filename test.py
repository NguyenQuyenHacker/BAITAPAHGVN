import numpy as np
import pandas as pd
from util import logger, dwh


def get_df_fgi():
    N_MM = 125
    N_VIX_STD = 50
    N_VIX_SMOOTH = 125
    N_SPS = 50
    N_BREADTH = 25

    df_hose = dwh.query('''
    SELECT ticker, tradingdate, closepriceadjusted AS closeprice
    FROM staging.stg_tcs_stx_mrk_hosestock
    WHERE LEN(ticker) = 3
    ORDER BY ticker, tradingdate
    ''')
    df_hose = df_hose.astype({'tradingdate': 'datetime64'})

    df_uptrend = (
        df_hose
        .assign(ticker_ma=lambda df: df.groupby('ticker')['closeprice'].rolling(window=20, closed='left').mean().values)
        .dropna()
        .assign(num_ticker_uptrend=lambda df: df['closeprice'] > df['ticker_ma'])
        .pivot_table(index='tradingdate', values='num_ticker_uptrend', aggfunc='sum').reset_index()
    )

    df_sps = (
        df_hose
        .assign(max_52w=lambda df: df.groupby('ticker')['closeprice'].rolling(window=250, closed='left').max().values)
        .assign(min_52w=lambda df: df.groupby('ticker')['closeprice'].rolling(window=250, closed='left').min().values)
        .eval("high_52w = closeprice / max_52w - 1")
        .eval("low_52w = closeprice / min_52w - 1")
        .update_where("max_52w == 0 | min_52w == 0", 'high_52w', 0)
        .update_where("max_52w == 0 | min_52w == 0", 'low_52w', 0)
        .case_when("high_52w > 0", 1, "low_52w < 0", -1, default=0, column_name='label_52w')
        .pivot_table(index='tradingdate', columns='label_52w', values='ticker', aggfunc='count').reset_index()
        .rename(columns={-1: 'num_ticker_low', 1: 'num_ticker_high', 0: 'num_ticker_sideway'})
        .fillna(0)
        .eval("net_52w = num_ticker_high - num_ticker_low")
        .assign(net_52w=lambda df: np.select(
            condlist=[df['num_ticker_low'] == 0],
            choicelist=[df['num_ticker_high']],
            default=df['net_52w']
        ))
        .eval("num_ticker = num_ticker_low + num_ticker_high + num_ticker_sideway")
        .eval("sps = net_52w.rolling(@N_SPS, closed='left').mean()")
        .select_columns('tradingdate', 'sps', 'num_ticker')
    )

    df_breadth = (
        df_hose
        .assign(price_diff=lambda df: df.groupby('ticker')['closeprice'].diff())
        .dropna()
        .case_when("price_diff > 0", 1, "price_diff < 0", -1, "price_diff == 0", 0, default=np.nan,
                   column_name='breadth')
        .pivot_table(index='tradingdate', columns='breadth', values='ticker', aggfunc='count').reset_index()
        .rename(columns={-1: 'declined', 1: 'advanced', 0: 'sideways'})
        .fillna(0)
        .eval("up_down_vol = advanced - declined")
        .eval(
            "breadth = up_down_vol.ewm(alpha=0.1, adjust=False).mean() - up_down_vol.ewm(alpha=0.05, adjust=False).mean()")
        .eval("breadth = breadth.rolling(@N_BREADTH).mean()")
        .select_columns('tradingdate', 'breadth')
    )

    df_vnindex = dwh.query('''
    SELECT tradingdate, indexvalue AS vnindex
    FROM staging.stg_tcs_stx_mrk_hoseindex
    WHERE comgroupcode = 'VNINDEX' AND tradingdate >= '2015-01-01'
    ORDER BY tradingdate
    ''')
    df_vnindex = df_vnindex.astype({'tradingdate': 'datetime64'})

    def assign_rsi(df):
        diff = df['vnindex'].diff()
        up = np.where(diff > 0, diff, 0)
        down = np.where(diff < 0, -diff, 0)
        gain = pd.Series(up).ewm(com=(13), min_periods=14).mean()
        loss = pd.Series(down).ewm(com=(13), min_periods=14).mean()
        df['rsi'] = 100 - 100 / (1 + gain / loss)
        return df

    df_final = (
        df_vnindex
        .eval("momentum = vnindex / vnindex.rolling(@N_MM).mean() - 1")
        .eval("vix = vnindex.pct_change().rolling(@N_VIX_STD).std()")
        .eval("vix = vix / vix.rolling(@N_VIX_SMOOTH).mean() - 1")
        .pipe(assign_rsi)
        .merge(df_uptrend, on='tradingdate', how='left')
        .merge(df_sps, on='tradingdate', how='left')
        .merge(df_breadth, on='tradingdate', how='left')
        .eval("ratio_ticker_uptrend = num_ticker_uptrend / num_ticker")
    )

    components = ['momentum', 'vix', 'ratio_ticker_uptrend', 'breadth', 'sps']
    for col in components:
        df_final[col] = df_final[col].rolling(window=500).rank(method='average', na_option='keep', pct=True)
        df_final[col] = df_final[col].round(decimals=2) * 100

    df_final['fear_greed_score'] = (df_final[components].mean(axis=1) / 100).round(2) * 100
    df_final['createddatetime'] = pd.Timestamp.now()
    df_final = df_final.query("tradingdate >= '2020-01-01'")
    df_final = df_final.select_columns('tradingdate', 'fear_greed_score', 'vnindex', 'rsi', 'createddatetime')
    return df_final