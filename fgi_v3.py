import numpy as np
import pandas as pd
from util import dwh

def get_df_fgi_v4():
    """
    Fear & Greed Index - Version 4
    Kết hợp dữ liệu trung bình HOSE + VNINDEX
    """

    # Lấy giá trung bình cổ phiếu trên HOSE
    df_hose = dwh.query("""
        SELECT tradingdate, AVG(closepriceadjusted) AS avg_price
        FROM staging.stg_tcs_stx_mrk_hosestock
        WHERE LEN(ticker) = 3
        GROUP BY tradingdate
        ORDER BY tradingdate
    """).astype({'tradingdate': 'datetime64'})

    # Lấy VNINDEX
    df_index = dwh.query("""
        SELECT tradingdate, indexvalue AS vnindex
        FROM staging.stg_tcs_stx_mrk_hoseindex
        WHERE comgroupcode = 'VNINDEX'
        ORDER BY tradingdate
    """).astype({'tradingdate': 'datetime64'})

    # Merge 2 bảng
    df = df_index.merge(df_hose, on='tradingdate', how='left')

    # Momentum dựa trên VNINDEX
    df['momentum'] = df.vnindex / df.vnindex.rolling(200).mean() - 1

    # Breadth đơn giản: dựa trên avg_price thay đổi
    df['breadth'] = np.where(df.avg_price.diff() > 0, 1, -1)
    df['breadth'] = df['breadth'].rolling(10).mean()

    # Chuẩn hóa
    for col in ['momentum', 'breadth']:
        df[col] = df[col].rolling(250).rank(pct=True).round(2) * 100

    # Score
    df['fear_greed_score'] = df[['momentum', 'breadth']].mean(axis=1).round(2)

    df['createddatetime'] = pd.Timestamp.now()

    return df[['tradingdate', 'fear_greed_score', 'vnindex', 'avg_price', 'breadth', 'createddatetime']]
