# Market Hub

Local Streamlit app with three tools:

1. **Backtester** — pull OHLC for any stock/crypto and measure indicator accuracy (MA crossovers, RSI, MACD)
2. **Fund Holdings** — track 13F filings for famous funds (Buffett, Burry, Ackman, Druckenmiller, etc.) directly from SEC EDGAR
3. **Economic Indicators** — latest US macro data (CPI, NFP, Fed Funds, yield curve, VIX, etc.) from FRED

All data is free. No API keys required.

## Setup

```bash
cd ~/market-hub
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

The app opens at http://localhost:8501

## Notes & limitations

- **13F filings** are quarterly with a 45-day lag — that's the legal reality, not a tool limitation. Crypto generally doesn't appear in 13Fs (only spot Bitcoin/Ether ETFs like IBIT, FBTC, ETHA).
- **Pre-2022 filings** report `value` in thousands of dollars. **Post-2022** report actual dollars. The app shows raw filing values.
- **Intraday data** (1h, 4h) on Yahoo Finance is limited to ~730 days lookback.
- **Forward accuracy** = % of signal events where the next N-bar return matched signal direction. **Win rate** = % of round-trip trades that were profitable. They measure different things — both are shown.
- **Dark pool data**: not included. Real dark pool prints require a paid feed (Cheddar Flow, BlackBoxStocks, Unusual Whales). FINRA publishes weekly ATS volume free but with significant delay; can be added on request.

## Adding a fund to the famous list

Edit `holdings.py` → `FAMOUS_FUNDS`. Find a CIK by searching at https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany.
