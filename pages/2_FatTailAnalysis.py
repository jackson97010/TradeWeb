import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import scipy.stats as stats
import time
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.lines import Line2D

# ===============================
# 1) CnyesNewsSpider (鉅亨網爬蟲)
# ===============================
import requests
import csv

class CnyesNewsSpider:
    def __init__(self):
        self.base_url = "https://api.cnyes.com/media/api/v1/newslist/category/headline"
        self.headers = {
            'Origin': 'https://news.cnyes.com/',
            'Referer': 'https://news.cnyes.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        }

    def get_newslist_info(self, page=1, limit=30, start_date=None, end_date=None):
        """
        取得指定日期或日期區間的新聞資料
        """
        today_str = datetime.today().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = today_str
        if end_date is None:
            end_date = today_str

        params = {
            'page': page,
            'limit': limit,
        }

        try:
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        except ValueError as e:
            print("日期格式錯誤:", e)
            return None

        params['startAt'] = start_ts
        params['endAt'] = end_ts

        try:
            r = requests.get(self.base_url, headers=self.headers, params=params, timeout=30)
        except Exception as e:
            print("Request error:", e)
            return None

        if r.status_code != requests.codes.ok:
            print('Request failed:', r.status_code)
            return None

        json_data = r.json().get('items', {})
        if isinstance(json_data, dict) and "data" in json_data:
            return json_data["data"]
        else:
            return json_data

# Optional: If you only need to fetch & display in Streamlit (no CSV saving), you can skip the CSV logic.
# But here we keep it for reference.
    def save_to_csv(self, news_list, file_path="news_output.csv"):
        if not news_list:
            print("沒有新聞資料可儲存。")
            return

        fieldnames = ["title", "publishAt", "url", "publisher"]
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in news_list:
                    title = item.get("title", "No Title")
                    publish_ts = item.get("publishAt", 0)
                    publish_date = (
                        datetime.fromtimestamp(publish_ts).strftime("%Y-%m-%d %H:%M:%S")
                        if publish_ts
                        else ""
                    )
                    news_id = item.get("newsId", "")
                    url = f"https://news.cnyes.com/news/id/{news_id}" if news_id else ""
                    publisher = item.get("publisher", "Unknown")
                    writer.writerow({
                        "title": title,
                        "publishAt": publish_date,
                        "url": url,
                        "publisher": publisher
                    })
            print(f"成功儲存 {len(news_list)} 筆新聞到 {file_path}")
        except Exception as e:
            print("儲存 CSV 檔案時發生錯誤:", e)

def fetch_cnyes_news_for_date(date_str, limit=10):
    """
    Helper function to fetch cnyes news for a single date (start_date=end_date=date_str).
    Returns a list of news items.
    """
    spider = CnyesNewsSpider()
    news_list = spider.get_newslist_info(
        page=1,
        limit=limit,
        start_date=date_str,
        end_date=date_str
    )
    return news_list or []


# ===============================
# 2) Fat Tail Analysis Code
# ===============================
def download_data_with_retry(ticker, start, end, max_retries=5, retry_delay=30):
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=False)
            return data
        except Exception as e:
            if "Rate limited" in str(e) or "YFRateLimitError" in str(e):
                st.warning(f"Rate limit error: {e}. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise e
    st.error("Maximum retry attempts reached. No data downloaded.")
    return pd.DataFrame()

def create_price_chart_matplotlib(df, ticker, chart_type="line"):
    fig, ax = plt.subplots(figsize=(10, 5))

    if chart_type == "line":
        ax.plot(df.index, df["Close"], color='blue', label="Close Price")
        ax.set_title(f"{ticker} Daily Trend (Line Chart)")
        ax.legend(loc="upper right")
    else:
        # Candlestick
        ohlc_data = df[['Open', 'High', 'Low', 'Close']].copy().reset_index()
        ohlc_data['Date'] = ohlc_data['Date'].map(mdates.date2num)
        ohlc_values = ohlc_data[['Date', 'Open', 'High', 'Low', 'Close']].values
        candlestick_ohlc(ax, ohlc_values, width=0.6, colorup='g', colordown='r', alpha=0.8)
        proxy_line = Line2D([0], [0], color='black', lw=2)
        ax.legend([proxy_line], ["Candlestick Chart"], loc="upper right")
        ax.set_title(f"{ticker} Daily Trend (Candlestick)")

    ax.set_ylabel("Price")
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    return fig

def create_returns_chart_plotly(df, mean_return, upper_threshold, lower_threshold):
    returns = df["Return"]
    positive_fat_tails = df[df["Return"] > upper_threshold]
    negative_fat_tails = df[df["Return"] < lower_threshold]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns,
            mode="lines+markers",
            name="Daily Returns",
            line=dict(color="blue"),
            marker=dict(color="blue", size=5, opacity=0.7)
        )
    )

    fig.add_hline(
        y=mean_return,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_return:.4f}",
        annotation_position="top left"
    )
    fig.add_hline(
        y=upper_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"+2 Std Dev: {upper_threshold:.4f}",
        annotation_position="top left"
    )
    fig.add_hline(
        y=lower_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"-2 Std Dev: {lower_threshold:.4f}",
        annotation_position="bottom left"
    )

    if not positive_fat_tails.empty:
        fig.add_trace(
            go.Scatter(
                x=positive_fat_tails.index,
                y=positive_fat_tails["Return"],
                mode="markers",
                name="Positive Fat Tail",
                marker=dict(color="lime", size=8),
            )
        )

    if not negative_fat_tails.empty:
        fig.add_trace(
            go.Scatter(
                x=negative_fat_tails.index,
                y=negative_fat_tails["Return"],
                mode="markers",
                name="Negative Fat Tail",
                marker=dict(color="red", size=8),
            )
        )

    fig.update_layout(
        title="Daily Returns with Fat Tail Events",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        hovermode="x unified"
    )
    return fig

def create_distribution_chart_matplotlib(returns, mean_return, std_return, lower_threshold, upper_threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    n, bins, patches = ax.hist(returns, bins=50, density=True, alpha=0.6, color='blue', label="Historical Returns")

    x_pdf = np.linspace(returns.min(), returns.max(), 1000)
    pdf = stats.norm.pdf(x_pdf, loc=mean_return, scale=std_return)
    ax.plot(x_pdf, pdf, 'r', lw=2, label="Fitted Normal PDF")

    ax.axvline(lower_threshold, color='green', linestyle='--', linewidth=2, label=f"-2 Std Dev: {lower_threshold:.4f}")
    ax.axvline(upper_threshold, color='green', linestyle='--', linewidth=2, label=f"+2 Std Dev: {upper_threshold:.4f}")

    ax.set_title("Distribution of Daily Returns")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig

def create_qq_plot_plotly(returns, mean_return, std_return):
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    prob = (np.arange(n) + 0.5) / n
    theoretical_quantiles = stats.norm.ppf(prob, loc=mean_return, scale=std_return)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_returns,
            mode="markers",
            name="Data Points",
            marker=dict(color="blue", size=6)
        )
    )

    min_val = min(theoretical_quantiles.min(), sorted_returns.min())
    max_val = max(theoretical_quantiles.max(), sorted_returns.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="45-degree line",
            line=dict(color="red", dash="dash")
        )
    )

    fig.update_layout(
        title="Q-Q Plot of Daily Returns (Normal)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sorted Returns",
        hovermode="x unified"
    )
    return fig


def fat_tail_analysis_page():
    st.title("Fat Tail Analysis with Cnyes News")

    # -------------------------
    # Sidebar inputs
    # -------------------------
    st.sidebar.header("Fat Tail Analysis Settings")
    ticker = st.sidebar.text_input("Ticker symbol", value="^N225")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2025, 1, 24))
    trend_format = st.sidebar.selectbox("Chart Type", ["candlestick", "line"], index=0)

    if st.sidebar.button("Download & Analyze"):
        with st.spinner("Downloading data..."):
            df = download_data_with_retry(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                max_retries=5,
                retry_delay=30
            )

        if df.empty:
            st.warning("No data downloaded. Please check your ticker or date range.")
            return

        # Calculate returns
        df["Return"] = df["Close"].pct_change()
        returns = df["Return"].dropna()

        if returns.empty:
            st.warning("No valid returns data available. Check date range.")
            return

        mean_return = returns.mean()
        std_return = returns.std()
        upper_threshold = mean_return + 2 * std_return
        lower_threshold = mean_return - 2 * std_return

        # 1) Static Price Chart
        st.subheader("Price Chart (Static)")
        fig_price = create_price_chart_matplotlib(df, ticker, chart_type=trend_format)
        st.pyplot(fig_price)

        # 2) Interactive Returns Chart
        st.subheader("Daily Returns & Fat Tail Events (Interactive)")
        returns_chart = create_returns_chart_plotly(df, mean_return, upper_threshold, lower_threshold)
        st.plotly_chart(returns_chart, use_container_width=True)

        # 3) Static Distribution Analysis
        st.subheader("Distribution Analysis (Static)")
        fig_dist = create_distribution_chart_matplotlib(returns, mean_return, std_return, lower_threshold, upper_threshold)
        st.pyplot(fig_dist)

        # 4) Q-Q Plot (Interactive)
        st.subheader("Q-Q Plot (Interactive)")
        qq_chart = create_qq_plot_plotly(returns, mean_return, std_return)
        st.plotly_chart(qq_chart, use_container_width=True)

        # 5) Probabilities under normal distribution
        p_lower = stats.norm.cdf(lower_threshold, loc=mean_return, scale=std_return)
        p_upper = 1 - stats.norm.cdf(upper_threshold, loc=mean_return, scale=std_return)
        st.subheader("Extreme Event Probabilities (Normal Dist. Assumption)")
        st.write(f"在常態分佈下，return < {lower_threshold:.4f} 的機率：{p_lower:.4f}")
        st.write(f"在常態分佈下，return > {upper_threshold:.4f} 的機率：{p_upper:.4f}")

        # 6) Dates of extreme return events
        st.subheader("Fat Tail Event Dates")

        # Negative fat tails
        negative_fat_tails = df[df["Return"] < lower_threshold]
        negative_dates = negative_fat_tails.index.strftime("%Y-%m-%d").tolist()

        # Positive fat tails
        positive_fat_tails = df[df["Return"] > upper_threshold]
        positive_dates = positive_fat_tails.index.strftime("%Y-%m-%d").tolist()

        st.write(f"報酬率低於 -2 標準差（{lower_threshold:.4f}）的日期：")
        if negative_dates:
            for date_str in negative_dates:
                st.markdown(f"- **{date_str}**")
                # Add an expander to show that day's news
                with st.expander("Show Cnyes News"):
                    news_list = fetch_cnyes_news_for_date(date_str, limit=10)
                    if not news_list:
                        st.write("No news found or error.")
                    else:
                        for item in news_list:
                            title = item.get("title", "No Title")
                            news_id = item.get("newsId", "")
                            url = f"https://news.cnyes.com/news/id/{news_id}" if news_id else ""
                            publish_ts = item.get("publishAt", 0)
                            if publish_ts:
                                publish_date = datetime.fromtimestamp(publish_ts).strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                publish_date = "Unknown"
                            st.markdown(f"- **{publish_date}**: [{title}]({url})")
        else:
            st.write("None")

        st.write("---")
        st.write(f"報酬率高於 +2 標準差（{upper_threshold:.4f}）的日期：")
        if positive_dates:
            for date_str in positive_dates:
                st.markdown(f"- **{date_str}**")
                # Add an expander to show that day's news
                with st.expander("Show Cnyes News"):
                    news_list = fetch_cnyes_news_for_date(date_str, limit=10)
                    if not news_list:
                        st.write("No news found or error.")
                    else:
                        for item in news_list:
                            title = item.get("title", "No Title")
                            news_id = item.get("newsId", "")
                            url = f"https://news.cnyes.com/news/id/{news_id}" if news_id else ""
                            publish_ts = item.get("publishAt", 0)
                            if publish_ts:
                                publish_date = datetime.fromtimestamp(publish_ts).strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                publish_date = "Unknown"
                            st.markdown(f"- **{publish_date}**: [{title}]({url})")
        else:
            st.write("None")

    else:
        st.info("請在左側選擇 Ticker 與日期，然後點擊 **Download & Analyze** 以進行 Fat Tail 分析。")


def main():
    fat_tail_analysis_page()

if __name__ == "__main__":
    main()
