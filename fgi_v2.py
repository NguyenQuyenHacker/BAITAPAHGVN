import numpy as np
import pandas as pd
from util import dwh

def get_df_fgi_v2():
    """
    Fear & Greed Index - Version 2
    Schema: mart
    Logic: sử dụng VNINDEX, tính toán momentum, vix, rsi và kết hợp lại
    """

    # Tham số
    N_MM = 100
    N_VIX = 50
    N_SMOOTH = 100

    # Lấy dữ liệu từ mart schema (không dùng biến động)
    df_index = dwh.query("""
        SELECT tradingdate, indexvalue AS vnindex
        FROM mart.stg_tcs_stx_mrk_hoseindex
        WHERE comgroupcode = 'VNINDEX'
          AND tradingdate >= '2018-01-01'
        ORDER BY tradingdate
    """).astype({'tradingdate': 'datetime64'})

    # Tính momentum
    df_index['momentum'] = df_index.vnindex / df_index.vnindex.rolling(N_MM).mean() - 1

    # Tính VIX
    df_index['vix'] = df_index.vnindex.pct_change().rolling(N_VIX).std()
    df_index['vix'] = df_index['vix'] / df_index['vix'].rolling(N_SMOOTH).mean() - 1

    # Tính RSI
    diff = df_index.vnindex.diff()
    up = np.where(diff > 0, diff, 0)
    down = np.where(diff < 0, -diff, 0)
    gain = pd.Series(up).ewm(com=13, min_periods=14).mean()
    loss = pd.Series(down).ewm(com=13, min_periods=14).mean()
    df_index['rsi'] = 100 - 100 / (1 + gain / loss)

    # Chuẩn hóa theo rank phần trăm
    for col in ['momentum', 'vix']:
        df_index[col] = df_index[col].rolling(400).rank(pct=True).round(2) * 100

    # Tổng hợp thành chỉ số Fear & Greed
    df_index['fear_greed_score'] = (
        (df_index['momentum'] + df_index['vix']) / 2
    ).round(2)

    df_index['createddatetime'] = pd.Timestamp.now()

    return df_index[['tradingdate', 'fear_greed_score', 'vnindex', 'rsi', 'createddatetime']]
