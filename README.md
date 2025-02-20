# TradeWeb

TradeWeb is a multi-page web application built with Streamlit that offers tools for stock market analysis. It includes features such as advanced technical analysis, fat tail analysis, sentiment analysis (powered by FinBERT), and PDF to Markdown conversion.

## Features

- **Advanced Analysis:** Explore technical indicators and charts.
- **Fat Tail Analysis:** Evaluate market risks and anomalies.
- **Sentiment Analysis:** Analyze stock news sentiment using FinBERT.
- **PDF to Markdown:** Convert PDF documents to Markdown format.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jackson97010/TradeWeb.git
   cd TradeWeb
2. **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
## Running the Application
Launch the app by running:

    streamlit run app_local.py
## Project Structure
    TradeWeb/
    ├── app.py              # Main entry point for the app
    ├── app_local.py        # Local testing version
    ├── pages/              # Multi-page application folder
    │   ├── 1_AdvancedAnalysis.py
    │   ├── 2_FatTailAnalysis.py
    │   ├── 3_SentimentAnalysis.py
    │   └── 4_PdfToMarkdown.py
    ├── utils_local.py      # Utility functions
    ├── requirements.txt    # Required packages
    └── README.md           # This file
## Liscense
MIT License

Copyright (c) 2025 jackson97010

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

