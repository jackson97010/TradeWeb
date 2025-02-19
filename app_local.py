# app.py
import streamlit as st
import pandas as pd

from utils_local import (
    read_kbars_local_parquet,
    filter_session_kbars,
    resample_kbars,
    create_interactive_candlestick_chart,
    create_daily_range_volume_chart_toggle,
    calc_day_session_stats,
    calc_daily_range
)

def main():
    st.title("TXF K-Bar Viewer (Local Parquet) - Main Page")

    # === 側邊欄參數輸入 ===
    st.sidebar.header("資料篩選條件")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2025-01-01"))
    end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime("2025-02-15"))

    session = st.sidebar.selectbox("Session (盤別)", ["full", "day", "night"], index=0)

    freq = st.sidebar.selectbox(
        "Resample 週期",
        ["1T", "5T", "15T", "30T", "60T", "D"],  # 包含 "D" 日K
        index=1
    )

    data_dir = st.sidebar.text_input("Parquet 資料夾路徑", "./kbars_data")

    # === 讀取資料 ===
    st.write(f"讀取資料區間: {start_date} ~ {end_date}")
    df_1m = read_kbars_local_parquet(str(start_date), str(end_date), data_dir=data_dir)

    if df_1m.empty:
        st.warning("讀不到任何資料，請確認日期或檔案路徑是否正確。")
        return

    st.write(f"成功讀取 1分K 資料，共 {len(df_1m)} 筆。")
    st.dataframe(df_1m.head(10))

    # === Session 篩選 ===
    if session != "full":
        df_1m = filter_session_kbars(df_1m, session=session)
        st.write(f"篩選 {session} 盤後，共 {len(df_1m)} 筆。")

    # === Resample ===
    if freq == "1T":
        df_resampled = df_1m.copy()
    else:
        df_resampled = resample_kbars(df_1m, rule=freq)
        st.write(f"Resample 成 {freq} 後，共 {len(df_resampled)} 筆。")

    # === 繪製蠟燭圖 ===
    fig_candle = create_interactive_candlestick_chart(
        df_resampled, title=f"TXF {freq} Candlestick ({session})"
    )
    if fig_candle:
        st.plotly_chart(fig_candle, use_container_width=True)

    # === 如果是日K，顯示日震幅 + Volume 雙軸圖 (toggle 版) ===
    if freq == "D":
        daily_df = calc_daily_range(df_resampled)
        if not daily_df.empty:
            st.subheader("日K振幅 (High - Low) Data")
            st.dataframe(daily_df[["Open", "High", "Low", "Close", "Volume", "range"]])

            # 這裡示範可切換隱藏/顯示的圖表
            fig_range_vol = create_daily_range_volume_chart_toggle(daily_df, "Daily Range vs Volume (Toggle)")
            if fig_range_vol:
                st.plotly_chart(fig_range_vol, use_container_width=True)

    # === 計算日盤統計 (範例) ===
    st.subheader("日盤統計：平均振幅 & 平均成交量")
    stats_df, stats_fig = calc_day_session_stats(df_1m)
    if stats_df.empty:
        st.warning("日盤統計資料為空，可能是日期內沒有符合日盤的 K 線。")
    else:
        st.dataframe(stats_df)
        if stats_fig:
            st.plotly_chart(stats_fig, use_container_width=True)


if __name__ == "__main__":
    main()
