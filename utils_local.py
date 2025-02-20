# utils_local.py
import os
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------- #
# 1) 資料讀取與基礎處理
# ---------------------- #

def read_kbars_local_parquet(
    start_date: str,
    end_date: str,
    data_dir: str = "./kbars_data"
) -> pd.DataFrame:
    """
    從本機資料夾 (data_dir) 讀取 TXF 當月合約之多個 Parquet 檔，
    依據 [start_date, end_date] (YYYY-MM-DD) 合併並過濾日期後回傳 DataFrame。
    回傳的 DataFrame:
      - index: datetime (ts)
      - columns: 包含 [Open, High, Low, Close, Volume, ...]
    """
    dt_start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    dt_end   = datetime.datetime.strptime(end_date,   "%Y-%m-%d")
    if dt_end < dt_start:
        raise ValueError("end_date must be >= start_date")

    # 1. 收集要讀取的檔案路徑 (以月份為單位)
    files_to_read = []
    current_dt = dt_start.replace(day=1)
    while current_dt <= dt_end:
        parquet_filename = current_dt.strftime("%Y-%m") + ".parquet"
        parquet_path = os.path.join(data_dir, parquet_filename)
        if os.path.isfile(parquet_path):
            files_to_read.append(parquet_path)
        current_dt += relativedelta(months=1)

    if not files_to_read:
        print("No Parquet files found in range. Check your data_dir or date range.")
        return pd.DataFrame()

    # 2. 讀取並合併
    df_list = []
    for f in files_to_read:
        tmp = pd.read_parquet(f)
        df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)
    if df.empty:
        return df

    # 3. 時間戳記 (ts, ms) -> datetime
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    # 4. 過濾日期範圍、排序並設定 index
    cond = (df["ts"] >= dt_start) & (df["ts"] <= dt_end)
    df = df[cond].copy()
    df.sort_values(by="ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.set_index("ts", inplace=True)
    df.index.name = "ts"
    return df


def filter_session_kbars(df: pd.DataFrame, session: str = "day") -> pd.DataFrame:
    """
    篩選盤別:
      - 'day':   8:45 ~ 13:45
      - 'night': 15:00 ~ 05:00 (簡化為 >=15:00 or <=05:00)
      - 'full':  不做篩選
    """
    if df.empty:
        return df

    if session == "day":
        st = datetime.time(8, 45)
        et = datetime.time(13, 45)
        cond = (df.index.time >= st) & (df.index.time <= et)
        return df[cond].copy()

    elif session == "night":
        st = datetime.time(15, 0)
        et = datetime.time(5, 0)
        cond = (df.index.time >= st) | (df.index.time <= et)
        return df[cond].copy()

    else:  # 'full'
        return df


def resample_kbars(df: pd.DataFrame, rule: str = "5T") -> pd.DataFrame:
    """
    將 1 分 K 壓縮成指定週期 (e.g. '5T','15T','30T','60T','D')。
    聚合: Open=first, High=max, Low=min, Close=last, Volume=sum
    """
    if df.empty:
        return df

    agg_dict = {
        "Open":  "first",
        "High":  "max",
        "Low":   "min",
        "Close": "last",
        "Volume": "sum"
    }
    rs = df.resample(rule).agg(agg_dict)
    rs.dropna(subset=["Open", "Close"], inplace=True)
    return rs


# ---------------------- #
# 2) 圖表繪製相關
# ---------------------- #

def create_interactive_candlestick_chart(df: pd.DataFrame, title: str = "TXF Candlestick"):
    """
    使用 Plotly 產生蠟燭圖 + 交易量的互動式圖表 (Figure)。
    df 需包含 [Open, High, Low, Close, Volume]，index 為 datetime。
    """
    if df.empty:
        return None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.03
    )

    # (1) 蠟燭圖
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price",
        ),
        row=1, col=1
    )
    # (2) 交易量
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume", marker_color="blue"
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        xaxis_rangeslider_visible=False
    )
    return fig


def create_daily_range_volume_chart(df: pd.DataFrame, title: str = "Daily Range vs Volume"):
    """
    將「日K」資料 (含 'range' 欄位) 繪製成 Range 與 Volume 的雙軸圖。
    假設 df.index = date, 且包含 [range, Volume] 欄位。
      - 左軸: range (柱狀)
      - 右軸: volume (折線)
    """
    if df.empty or "range" not in df.columns:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # 左軸: range (Bar) - trace 0
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["range"],
            name="Daily Range",
            marker_color="orange",
        ),
        secondary_y=False
    )
    # 右軸: volume (Line) - trace 1
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            mode="lines+markers",
            line_color="blue",
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Range", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)

    return fig


def create_daily_range_volume_chart_toggle(df: pd.DataFrame, title: str = "Daily Range vs Volume"):
    """
    與 create_daily_range_volume_chart 類似，
    但加上按鈕可切換顯示/隱藏 Volume 或 Range。
    """
    if df.empty or "range" not in df.columns:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # trace0 (Volume, Bar)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color="blue",
        ),
        secondary_y=False
    )
    # trace1 (Range, Line)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["range"],
            name="Daily Range",
            mode="lines+markers",
            line_color="orange",
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Range", secondary_y=True)

    # 加入 updatemenus (按鈕)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.15,
                showactive=True,
                buttons=list([
                    dict(
                        label="Show Both",
                        method="update",
                        args=[{"visible": [True, True]}],  # trace0, trace1
                    ),
                    dict(
                        label="Hide Volume",
                        method="update",
                        args=[{"visible": [False, True]}],
                    ),
                    dict(
                        label="Hide Range",
                        method="update",
                        args=[{"visible": [True, False]}],
                    ),
                ]),
            )
        ]
    )

    return fig


# ---------------------- #
# 3) 計算統計相關
# ---------------------- #

def calc_daily_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    適用「日K」資料 (freq="D")，計算當日震幅 range = High - Low。
    回傳 df (多一欄 'range')。
    """
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["range"] = out["High"] - out["Low"]
    return out


def calc_day_session_stats(df: pd.DataFrame):
    """
    計算「日盤 (8:45~13:45)」的統計: mean_range, mean_volume。
    回傳 (result_df, figure):
      - result_df: [date_only, mean_range, mean_volume]
      - figure: 兩個 bar subplots
    """
    if df.empty:
        return pd.DataFrame(), None

    # 只取日盤
    st_time = datetime.time(8, 45)
    et_time = datetime.time(13, 45)
    cond = (df.index.time >= st_time) & (df.index.time <= et_time)
    day_df = df[cond].copy()
    if day_df.empty:
        return pd.DataFrame(), None

    day_df["range"] = day_df["High"] - day_df["Low"]
    day_df["date_only"] = day_df.index.date

    grouped = day_df.groupby("date_only")
    result_df = grouped.agg(
        mean_range=("range", "mean"),
        mean_volume=("Volume", "mean")
    ).reset_index()

    # 繪製一個簡易雙 subplot
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["Mean Range", "Mean Volume"]
    )

    fig.add_trace(
        go.Bar(
            x=result_df["date_only"], y=result_df["mean_range"],
            name="Mean Range", marker_color="orange"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=result_df["date_only"], y=result_df["mean_volume"],
            name="Mean Volume", marker_color="blue"
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="Day Session Stats",
        hovermode="x unified",
        showlegend=False,
        annotations=[{
            "text": "商品每日一分k平均值",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": -0.3,
            "showarrow": False,
            "font": {"size": 12, "color": "gray"}
        }]
    )
    fig.update_xaxes(title_text="Date (YYYY-MM-DD)", row=2, col=1)
    fig.update_yaxes(title_text="Mean Range", row=1, col=1)
    fig.update_yaxes(title_text="Mean Volume", row=2, col=1)

    return result_df, fig
