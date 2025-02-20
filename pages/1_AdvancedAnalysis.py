import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils_local import (
    read_kbars_local_parquet,
    filter_session_kbars,
    calc_daily_range
)

def show_range_stats(daily_df: pd.DataFrame):
    """
    顯示 daily_df 的 range 統計資訊（平均值、標準差、最大值、最小值）以及其對應日期。
    """
    if daily_df.empty or "range" not in daily_df.columns:
        st.warning("daily_df 為空或沒有 'range' 欄位。")
        return

    # 1) 基礎統計
    mean_range = daily_df["range"].mean()
    std_range  = daily_df["range"].std()

    # 2) 最大震幅及日期
    max_range_val = daily_df["range"].max()
    max_range_idx = daily_df["range"].idxmax()  # 回傳該列索引
    if isinstance(max_range_idx, pd.Timestamp):
        max_range_date = max_range_idx.strftime("%Y-%m-%d")
    else:
        max_range_date = str(max_range_idx)  # 若索引不是 Timestamp，就轉字串

    # 3) 最小震幅及日期
    min_range_val = daily_df["range"].min()
    min_range_idx = daily_df["range"].idxmin()
    if isinstance(min_range_idx, pd.Timestamp):
        min_range_date = min_range_idx.strftime("%Y-%m-%d")
    else:
        min_range_date = str(min_range_idx)

    # 4) 顯示於 Streamlit
    st.write(f"平均震幅: {mean_range:.2f}")
    st.write(f"標準差: {std_range:.2f}")
    st.write(f"最大震幅: {max_range_val:.2f}，發生於 {max_range_date}")
    st.write(f"最小震幅: {min_range_val:.2f}，發生於 {min_range_date}")


def plot_correlogram(series: pd.Series, lags: int = 30):
    """
    計算並繪製指定欄位的自相關 (Correlogram)，含 95% 信賴區間的虛線。
    
    這裡使用簡單的 +/- 1.96/sqrt(N) 近似，以判斷自相關是否超出白噪音的範圍。
    """
    series = series.dropna()
    n = len(series)
    # 95% 信賴區間閾值 (白噪音假設)
    threshold = 1.96 / np.sqrt(n)

    # 計算 autocorr
    autocorr_values = [series.autocorr(lag=i) for i in range(1, lags + 1)]
    lags_range = list(range(1, lags + 1))

    # 建立圖表 (Bar for ACF)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=lags_range,
            y=autocorr_values,
            name="Autocorrelation",
            marker_color="blue"
        )
    )

    # 加入紅色虛線：+threshold, -threshold
    fig.add_shape(
        type="line", xref="paper", x0=0, x1=1, y0=threshold, y1=threshold,
        line=dict(color="red", dash="dash")
    )
    fig.add_shape(
        type="line", xref="paper", x0=0, x1=1, y0=-threshold, y1=-threshold,
        line=dict(color="red", dash="dash")
    )

    # 更新 layout
    fig.update_layout(
        title=f"Correlogram for {series.name} (up to lag {lags})",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        hovermode="x unified"
    )
    return fig


def advanced_analysis_page():
    st.title("進階分析 - Advanced Analysis")

    # 側邊欄輸入
    st.sidebar.header("進階分析 - 資料篩選")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2025-01-01"))
    end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime("2025-02-15"))
    session    = st.sidebar.selectbox("Session (盤別)", ["full", "day", "night"], index=0)
# 定義下拉選單的選項
    folder_options = ["./kbars_data", "./kbars_data_TSE", "./kbars_data_OTC", "Other"]

    # 先在側邊欄建立下拉選單，預設選項為 "./kbars_data" 可根據需求調整
    selected_option = st.sidebar.selectbox("Parquet 資料夾路徑", folder_options, index=folder_options.index("./kbars_data_OTC") if "./kbars_data_OTC" in folder_options else 0)

    # 若選擇 "Other"，則顯示文字輸入框，否則使用下拉選單的選項
    if selected_option == "Other":
        data_dir = st.sidebar.text_input("請輸入自訂資料夾路徑", "./kbars_data")
    else:
        data_dir = selected_option

    st.write("目前使用的資料夾路徑：", data_dir)
    # 新增 correlogram 分析的參數輸入
    st.sidebar.subheader("Correlogram 設定")
    lags = st.sidebar.number_input("選擇 lag 數量", min_value=1, max_value=60, value=30)

    # 讀取資料
    df = read_kbars_local_parquet(str(start_date), str(end_date), data_dir)
    if df.empty:
        st.warning("讀不到任何資料，請確認日期或檔案路徑是否正確。")
        return

    # 篩選盤別
    if session != "full":
        df = filter_session_kbars(df, session=session)

    # 先做日K (resample) 後再計算 daily range
    daily_df = df.resample("D").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Open"])   # 若某天無交易資料則可能 NaN

    daily_df = calc_daily_range(daily_df)  # 加上 'range' 欄位

    if daily_df.empty:
        st.warning("沒有可用的日K資料。")
        return

    st.subheader("日K資料")
    st.dataframe(daily_df.head(10))

    # ==============
    # 震幅統計數據
    # ==============
    mean_range = daily_df["range"].mean()
    std_range  = daily_df["range"].std()
    max_range  = daily_df["range"].max()
    min_range  = daily_df["range"].min()

    st.write(f"平均震幅: {mean_range:.2f}")
    st.write(f"標準差: {std_range:.2f}")
 
    show_range_stats(daily_df)
    # ==============
    # 震幅分布 (Histogram)
    # ==============
    st.subheader("震幅分布 (Histogram)")
    fig_hist = px.histogram(
        daily_df,
        x="range",
        nbins=20,
        title="Daily Range Distribution"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ==============
    # range vs volume
    # ==============
    st.subheader("range vs. volume 散佈圖")
    st.write("一段時間內的振幅和成交量狀況")
    fig_scatter = px.scatter(
        daily_df,
        x="range",
        y="Volume",
        title="range vs. Volume scatter"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ==============
    # Correlogram 分析 (含 95% 信賴區間)
    # ==============
    st.subheader("Correlogram 分析")
    st.write("查看商品每日震幅在不同的天數之內的序列相關狀態")

    if "range" in daily_df.columns:
        corr_fig = plot_correlogram(daily_df["range"], lags=int(lags))
        st.plotly_chart(corr_fig, use_container_width=True)
        st.markdown("""
        **如何解讀：**
        - X軸為延遲天數 (Lag)；Y軸為自相關係數。
        - 紅色虛線表示 95% 信賴區間 (±1.96/√N)。
        - 超過此區間（上或下）即表示對應的 Lag 具有統計上顯著的自相關 (在白噪音假設下)。
        """)
    else:
        st.warning("日K資料中沒有 'range' 欄位，無法進行 correlogram 分析。")

def main():
    advanced_analysis_page()

if __name__ == "__main__":
    main()
