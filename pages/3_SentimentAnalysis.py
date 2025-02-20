import streamlit as st
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- FinBERT 載入 ----------------
# 使用 FinBERT 模型 (資源來源: https://huggingface.co/ProsusAI/finbert)
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# --------------------------------------------------

# 若已有主頁面設定，則此處可註解掉
st.set_page_config(
    page_title="Bohmian's Stock News Sentiment Analyzer",
    layout="wide"
)

def get_news(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}&ty=c&ta=1&p=d"
    req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html.parser')
    news_table = html.find(id='news-table')
    return news_table

def convert_to_iso_date(date_str):
    try:
        dt = datetime.datetime.strptime(date_str, "%b-%d-%y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return datetime.datetime.today().strftime('%Y-%m-%d')

def parse_news(news_table):
    parsed_news = []
    last_date_str = None
    today_str = datetime.datetime.today().strftime('%Y-%m-%d')
    
    rows = news_table.find_all('tr')
    for row in rows:
        try:
            headline_tag = row.a
            headline_text = headline_tag.get_text(strip=True)
            headline_link = headline_tag['href'] if headline_tag.has_attr('href') else ""
            date_scrape = row.td.text.strip().split()
            
            if len(date_scrape) == 2:
                maybe_date, maybe_time = date_scrape
                if maybe_date.lower() == "today":
                    date_str = today_str
                else:
                    date_str = convert_to_iso_date(maybe_date)
                time_str = maybe_time
                last_date_str = date_str
            elif len(date_scrape) == 1:
                time_str = date_scrape[0]
                date_str = last_date_str if last_date_str else today_str
            else:
                continue
            
            parsed_news.append([date_str, time_str, headline_text, headline_link])
        except Exception as e:
            continue

    df = pd.DataFrame(parsed_news, columns=['date', 'time', 'headline', 'link'])
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    return df

def finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    pos = float(probs[0])
    neg = float(probs[1])
    neu = float(probs[2])
    compound = pos - neg
    
    return {
        "neg": neg,
        "neu": neu,
        "pos": pos,
        "sentiment_score": compound
    }

def score_news(df):
    results = df['headline'].apply(finbert_sentiment)
    scores_df = pd.DataFrame(results.tolist())
    parsed_and_scored_news = df.join(scores_df)
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    return parsed_and_scored_news

def stock_news_sentiment_page():
    # 頁面標題與說明，包含來源資訊
    st.header("US Stock News Sentiment Analyzer")
    st.markdown(
        "**Resources:** News data provided by [Finviz](https://finviz.com) | "
        "Sentiment analysis powered by [FinBERT](https://huggingface.co/ProsusAI/finbert)"
    )

    ticker = st.text_input('Enter Stock Ticker', '').upper()

    if ticker:
        try:
            news_table = get_news(ticker)
            parsed_news_df = parse_news(news_table)
            parsed_and_scored_news = score_news(parsed_news_df)
            
            # 依據日期排序（以 datetime index 排序後，再以 date 分組）
            parsed_and_scored_news = parsed_and_scored_news.sort_index(ascending=False)
            
            st.markdown(f"### {ticker} News")
            
            # 以日期作為分組，使用 expander 展開各日期新聞內容
            grouped = parsed_and_scored_news.groupby('date')
            for date, group in grouped:
                with st.expander(f"{date} ({len(group)} 篇新聞)"):
                    for idx, row in group.iterrows():
                        md_text = f"**{row['time']}** - [{row['headline']}]({row['link']})\n\n"
                        md_text += (
                            f"Sentiment: **pos:** {row['pos']:.2f}, **neg:** {row['neg']:.2f}, "
                            f"**neu:** {row['neu']:.2f}, **compound:** {row['sentiment_score']:.2f}\n\n---"
                        )
                        st.markdown(md_text)
        except Exception as e:
            st.write("請輸入有效的股票代號，例如 'TSLA' 後按 Enter。")
            st.write(str(e))
    else:
        st.write("請先輸入股票代號。")

    # 隱藏 Streamlit 內建選單與版權宣告
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    stock_news_sentiment_page()

if __name__ == "__main__":
    main()
