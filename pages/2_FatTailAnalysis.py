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

def download_data_with_retry(ticker, start, end, max_retries=5, retry_delay=30):
    """
    Download data from Yahoo Finance with a retry mechanism to handle rate limits.
    """
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=False)
            return data
        except Exception as e:
            # Check for known rate-limit messages
            if "Rate limited" in str(e) or "YFRateLimitError" in str(e):
                st.warning(f"Rate limit error encountered: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise e
    st.error("Maximum retry attempts reached. No data downloaded.")
    return pd.DataFrame()

def create_price_chart_matplotlib(df, ticker, chart_type="line"):
    """
    Create a static Matplotlib chart of the price (Close) data.
    If chart_type = "line", we plot a simple line chart.
    If chart_type = "candlestick", we plot a candlestick chart.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if chart_type == "line":
        ax.plot(df.index, df["Close"], color='blue', label="Close Price")
        ax.set_title(f"{ticker} Daily Trend (Line Chart)")
        ax.legend(loc="upper right")
    else:
        # Prepare data for candlestick
        ohlc_data = df[['Open', 'High', 'Low', 'Close']].copy().reset_index()
        ohlc_data['Date'] = ohlc_data['Date'].map(mdates.date2num)
        ohlc_values = ohlc_data[['Date', 'Open', 'High', 'Low', 'Close']].values

        candlestick_ohlc(ax, ohlc_values, width=0.6, colorup='g', colordown='r', alpha=0.8)
        proxy_line = Line2D([0], [0], color='black', lw=2)
        ax.legend([proxy_line], ["Candlestick Chart"], loc="upper right")
        ax.set_title(f"{ticker} Daily Trend (Candlestick)")

    ax.set_ylabel("Price")
    ax.grid(True)
    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    return fig

def create_returns_chart_plotly(df, mean_return, upper_threshold, lower_threshold):
    """
    Create an interactive Plotly chart for daily returns, with lines for mean and ±2 std dev.
    Mark 'fat tail' points in different colors.
    """
    returns = df["Return"]
    positive_fat_tails = df[df["Return"] > upper_threshold]
    negative_fat_tails = df[df["Return"] < lower_threshold]

    fig = go.Figure()

    # Daily returns scatter/line
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

    # Mean line
    fig.add_hline(
        y=mean_return,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_return:.4f}",
        annotation_position="top left"
    )

    # Upper threshold
    fig.add_hline(
        y=upper_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"+2 Std Dev: {upper_threshold:.4f}",
        annotation_position="top left"
    )

    # Lower threshold
    fig.add_hline(
        y=lower_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"-2 Std Dev: {lower_threshold:.4f}",
        annotation_position="bottom left"
    )

    # Positive fat tails
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

    # Negative fat tails
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
    """
    Create a static Matplotlib histogram of returns + fitted normal PDF overlay,
    and vertical lines for ±2 std dev.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # 1) Histogram
    # Using 'density=True' to get a normalized histogram
    n, bins, patches = ax.hist(returns, bins=50, density=True, alpha=0.6, color='blue', label="Historical Returns")

    # 2) Normal PDF overlay
    x_pdf = np.linspace(returns.min(), returns.max(), 1000)
    pdf = stats.norm.pdf(x_pdf, loc=mean_return, scale=std_return)
    ax.plot(x_pdf, pdf, 'r', lw=2, label="Fitted Normal PDF")

    # 3) ±2 std dev lines
    ax.axvline(lower_threshold, color='green', linestyle='--', linewidth=2, label=f"-2 Std Dev: {lower_threshold:.4f}")
    ax.axvline(upper_threshold, color='green', linestyle='--', linewidth=2, label=f"+2 Std Dev: {upper_threshold:.4f}")

    ax.set_title("Distribution of Daily Returns")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    return fig

def create_qq_plot_plotly(returns, mean_return, std_return):
    """
    Create an interactive Q-Q plot against a normal distribution using Plotly.
    We'll compute theoretical quantiles for each data point, then plot them vs. sorted returns.
    We'll also add a 45-degree reference line.
    """
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    # Probability points
    prob = (np.arange(n) + 0.5) / n
    # Theoretical quantiles under normal
    theoretical_quantiles = stats.norm.ppf(prob, loc=mean_return, scale=std_return)

    fig = go.Figure()

    # Scatter for the Q-Q points
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_returns,
            mode="markers",
            name="Data Points",
            marker=dict(color="blue", size=6)
        )
    )

    # 45-degree reference line (from min to max)
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
    st.title("Fat Tail Analysis (Hybrid)")

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
            st.warning("No valid returns data available after calculation. Check the date range.")
            return

        mean_return = returns.mean()
        std_return = returns.std()
        upper_threshold = mean_return + 2 * std_return
        lower_threshold = mean_return - 2 * std_return

        # -------------------------
        # 1) Static Price Chart (Matplotlib)
        # -------------------------
        st.subheader("Price Chart (Static)")
        fig_price = create_price_chart_matplotlib(df, ticker, chart_type=trend_format)
        st.pyplot(fig_price)  # Displays a non-interactive chart

        # -------------------------
        # 2) Interactive Returns Chart
        # -------------------------
        st.subheader("Daily Returns & Fat Tail Events (Interactive)")
        returns_chart = create_returns_chart_plotly(df, mean_return, upper_threshold, lower_threshold)
        st.plotly_chart(returns_chart, use_container_width=True)

        # -------------------------
        # 3) Static Distribution Analysis (Matplotlib)
        # -------------------------
        st.subheader("Distribution Analysis (Static)")
        fig_dist = create_distribution_chart_matplotlib(returns, mean_return, std_return, lower_threshold, upper_threshold)
        st.pyplot(fig_dist)

        # -------------------------
        # 4) Q-Q Plot (Interactive)
        # -------------------------
        st.subheader("Q-Q Plot (Interactive)")
        qq_chart = create_qq_plot_plotly(returns, mean_return, std_return)
        st.plotly_chart(qq_chart, use_container_width=True)

        # -------------------------
        # 5) Theoretical probabilities under normal distribution
        # -------------------------
        p_lower = stats.norm.cdf(lower_threshold, loc=mean_return, scale=std_return)
        p_upper = 1 - stats.norm.cdf(upper_threshold, loc=mean_return, scale=std_return)

        st.subheader("Extreme Event Probabilities (Normal Dist. Assumption)")
        st.write(f"在常態分佈下，return < {lower_threshold:.4f} 的機率：{p_lower:.4f}")
        st.write(f"在常態分佈下，return > {upper_threshold:.4f} 的機率：{p_upper:.4f}")

        # -------------------------
        # 6) Dates of extreme return events
        # -------------------------
        st.subheader("Fat Tail Event Dates")
        negative_fat_tails = df[df["Return"] < lower_threshold]
        positive_fat_tails = df[df["Return"] > upper_threshold]

        negative_dates = negative_fat_tails.index.strftime("%Y-%m-%d").tolist()
        positive_dates = positive_fat_tails.index.strftime("%Y-%m-%d").tolist()

        st.write(f"報酬率低於 -2 標準差（{lower_threshold:.4f}）的日期：")
        st.write(negative_dates)
        st.write("---")
        st.write(f"報酬率高於 +2 標準差（{upper_threshold:.4f}）的日期：")
        st.write(positive_dates)

    else:
        st.info("請在左側選擇 Ticker 與日期，然後點擊 **Download & Analyze** 以進行 Fat Tail 分析。")


def main():
    fat_tail_analysis_page()

if __name__ == "__main__":
    main()
