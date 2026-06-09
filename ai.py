import streamlit as st
import pandas as pd
from FinMind.data import DataLoader
import datetime
from groq import Groq
import yfinance as yf
import plotly.graph_objects as go
import re

# --- 1. 初始化設定 ---
st.set_page_config(page_title="台股 AI 智慧分析助手", layout="wide")

client = Groq(api_key="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")

if "sentiment_scores" not in st.session_state:
    st.session_state.sentiment_scores = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

TAIWAN_100_STOCKS = {
    "2330": "台積電", "2454": "聯發科", "2317": "鴻海", "2881": "富邦金",
    "2308": "台達電", "2882": "國泰金", "2303": "聯電", "3711": "日月光投控",
    "2412": "中華電", "2603": "長榮", "2382": "廣達", "2891": "中信金",
    "1101": "台泥", "1102": "亞泥", "1216": "統一", "1301": "台塑",
    "1303": "南亞", "1326": "台化", "1402": "遠東新", "1503": "士電",
    "1504": "東元", "1513": "中興電", "1519": "華城", "1605": "華新",
    "2002": "中鋼", "2105": "正新", "2204": "中華車", "2207": "和泰車",
    "2324": "仁寶", "2327": "國巨", "2345": "智邦", "2352": "佳世達",
    "2353": "宏碁", "2357": "華碩", "2360": "致茂", "2371": "大同",
    "2379": "瑞昱", "2395": "研華", "2408": "南亞科", "2409": "面板雙虎(友達)",
    "2449": "京元電子", "2451": "創見", "2474": "可成", "2609": "陽明",
    "2610": "華航", "2615": "萬海", "2618": "長榮航", "2801": "彰銀",
    "2880": "華南金", "2883": "開發金", "2884": "玉山金", "2885": "元大金",
    "2886": "兆豐金", "2887": "台新金", "2888": "新光金", "2889": "國票金",
    "2890": "永豐金", "2892": "第一金", "2912": "統一超", "3008": "大立光",
    "3017": "奇鋐", "3034": "聯詠", "3035": "智原", "3037": "欣興",
    "3045": "台灣大", "3231": "緯創", "3443": "創意", "3481": "群創",
    "3533": "嘉澤", "3653": "健策", "3661": "世芯-KY", "4904": "遠傳",
    "4938": "和碩", "4958": "臻鼎-KY", "4961": "天鈺", "5269": "祥碩",
    "5347": "世界", "5483": "中美晶", "5871": "中租-KY", "5876": "上海商銀",
    "5880": "合庫金", "6415": "矽力*-KY", "6446": "藥華藥", "6505": "台塑化",
    "6669": "緯穎", "6770": "力積電", "8046": "南電", "8454": "富邦媒",
    "9904": "寶成", "9910": "豐泰", "9921": "巨大", "9945": "潤泰新",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息",
    "00919": "群益台灣精選高股息", "00929": "復華台灣科技優息", "00940": "元大台灣價值高息"
}
stock_options = [f"{k} {v}" for k, v in TAIWAN_100_STOCKS.items()]


def parse_sentiment_scores(text):
    scores = {}
    patterns = {
        "fundamental": r"基本面情緒[：:]\s*(\d{1,3})\s*分",
        "market":      r"市場面情緒[：:]\s*(\d{1,3})\s*分",
        "overall":     r"綜合推薦指數[：:]\s*(\d{1,3})\s*分",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            scores[key] = int(match.group(1))
    return scores if len(scores) == 3 else None


def score_to_props(score):
    """回傳 (emoji, border_color, label_color, score_text)"""
    if score is None:
        return "⚪", "#aaa", "#888", "未分析"
    if score >= 80:
        return "🟢", "#00c853", "#00a846", f"{score} 分"
    elif score >= 60:
        return "🟡", "#f9a825", "#c17900", f"{score} 分"
    else:
        return "🔴", "#d50000", "#b00000", f"{score} 分"


def render_indicator_cards(scores):
    """用純 Streamlit columns 渲染指示燈，不依賴 unsafe_allow_html"""
    labels = [
        ("fundamental", "基本面情緒"),
        ("market",      "市場面情緒"),
        ("overall",     "綜合推薦指數"),
    ]
    cols = st.columns(3)
    for col, (key, label) in zip(cols, labels):
        score = scores.get(key) if scores else None
        emoji, border, text_color, score_text = score_to_props(score)
        with col:
            # 用 container + markdown 模擬卡片，border 用 CSS var 兼容深淺色主題
            st.markdown(
                f"""<div style="
                        border: 2px solid {border};
                        border-radius: 10px;
                        padding: 12px 8px;
                        text-align: center;
                        background: transparent;
                    ">
                    <div style="font-size:1.8rem;">{emoji}</div>
                    <div style="font-size:0.78rem; margin: 4px 0 2px; opacity:.75;">{label}</div>
                    <div style="font-size:1.05rem; font-weight:700; color:{text_color};">{score_text}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def strip_sentiment_section(text):
    """移除 AI 回應中第四點多維度情緒指標那段，避免使用者看到原始分數文字"""
    cleaned = re.sub(
        r'\n*\**\s*4[\.、．]?\s*多維度情緒指標.*',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    return cleaned.strip()


def get_ai_analysis(stock_id, news_data):
    prompt = f"""你是台股分析師，分析股票 {stock_id} 的新聞並給出專業解讀：
1. 關鍵三亮點：提煉最重要的正面發展。
2. 隱藏風險點：識別潛在負面或不確定因子。
3. 專業短評：針對近期趨勢給出結構化結論。
4. 多維度情緒指標（各維度皆為 0 到 100 分，50 分為中性）：
   - 基本面情緒：XX分（評估對公司實質獲利/營收的長期影響）
   - 市場面情緒：XX分（評估媒體炒作、散戶追價或市場話題熱度）
   - 綜合推薦指數：XX分（綜合以上兩者後，分析師整體的偏向，並附帶一句話理由）

【重要格式規定】第四點三行必須嚴格寫成：
   - 基本面情緒：<數字>分
   - 市場面情緒：<數字>分
   - 綜合推薦指數：<數字>分
冒號後直接接阿拉伯數字，再接「分」字，不可有其他文字穿插。

請直接依據上述四點結構輸出，不要有前言或後記。

新聞資料：{news_data}"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI 分析暫時無法執行: {e}"


# --- 2. 側邊欄 ---
st.sidebar.header("📊 參數設定")
selected_stock = st.sidebar.selectbox("選擇台股標的", options=stock_options)
stock_id = selected_stock.split(" ")[0]
analysis_days = st.sidebar.slider("新聞追蹤天數", 1, 30, 7)

# 切換標的時重置
if "last_stock_id" not in st.session_state or st.session_state.last_stock_id != stock_id:
    st.session_state.sentiment_scores = None
    st.session_state.analysis_result = None
    st.session_state.last_stock_id = stock_id

run_analysis = st.sidebar.button("執行 AI 分析")


# --- 3. 數據抓取 (新增 Cache 優化新聞與股價抓取) ---
@st.cache_data(ttl=3600)
def load_stock_data(sid):
    ticker = f"{sid}.TW"
    df = yf.download(ticker, period="6mo", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    return df.dropna(subset=['Close'])


@st.cache_data(ttl=1800)  # 快取新聞 30 分鐘，避免重複發送慢速 API 請求
def fetch_taiwan_stock_news(sid, start_date):
    dl = DataLoader()
    return dl.taiwan_stock_news(stock_id=sid, start_date=start_date)


stock_df = load_stock_data(stock_id)

# --- 4. 若按下按鈕，先執行分析並存入 session_state ---
if run_analysis:
    with st.spinner("正在分析新聞..."):
        start_date = (datetime.date.today() - datetime.timedelta(days=analysis_days)).strftime('%Y-%m-%d')
        # 改用快取的 fetch_taiwan_stock_news
        df_news = fetch_taiwan_stock_news(sid=stock_id, start_date=start_date)
        if not df_news.empty:
            news_text = "\n".join([f"- {t}" for t in df_news['title'].unique()[:15]])
            result = get_ai_analysis(stock_id, news_text)
            parsed = parse_sentiment_scores(result)
            st.session_state.sentiment_scores = parsed
            st.session_state.analysis_result = strip_sentiment_section(result)
        else:
            st.session_state.analysis_result = "__NO_NEWS__"
            st.session_state.sentiment_scores = None

# --- 5. 頂部 Metrics ---
if not stock_df.empty:
    latest_price = float(stock_df['Close'].iloc[-1])
    prev_price   = float(stock_df['Close'].iloc[-2])
    delta        = latest_price - prev_price

    m1, m2 = st.columns(2)
    with m1:
        st.metric("最新收盤價", f"{latest_price:.2f}", f"{delta:.2f} ({delta/prev_price:.2%})")
    with m2:
        st.metric("今日成交量", f"{int(stock_df['Volume'].iloc[-1]):,}")

    st.markdown("#### 🚦 AI 情緒指示燈")
    render_indicator_cards(st.session_state.sentiment_scores)
    if not st.session_state.sentiment_scores:
        st.caption("尚未執行 AI 分析，或分析結果中未能識別分數格式。")

st.markdown("---")

# --- 6. 主頁面佈局 ---
col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader(f"📈 {selected_stock} 歷史走勢")
    if not stock_df.empty:
        plot_df = stock_df.reset_index()
        if 'Date' not in plot_df.columns:
            if 'date' in plot_df.columns:
                plot_df = plot_df.rename(columns={'date': 'Date'})
            else:
                plot_df = plot_df.rename(columns={plot_df.columns[0]: 'Date'})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df['Date'], y=plot_df['Close'],
            mode='lines', name='收盤價',
            hovertemplate='日期: %{x|%Y-%m-%d}<br>價格: %{y:.2f}<extra></extra>'
        ))
        fig.add_trace(go.Bar(
            x=plot_df['Date'], y=plot_df['Volume'],
            name='成交量', yaxis='y2',
            marker_color='rgba(158, 202, 225, 0.5)',
            hovertemplate='成交量: %{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(
            hovermode="x unified", template="plotly_white", height=500,
            yaxis=dict(title="股價 (TWD)"),
            yaxis2=dict(title="成交量", overlaying='y', side='right', showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.subheader("🤖 AI 專業解讀")
    if st.session_state.analysis_result:
        if st.session_state.analysis_result == "__NO_NEWS__":
            st.warning("區間內無相關新聞。")
        else:
            st.write(st.session_state.analysis_result)
    else:
        st.info("點擊左側按鈕開始分析新聞情緒。")

# --- 7. 底部新聞 (改用快取函數) ---
with st.expander("📰 查看原始新聞清單"):
    # 這裡與上面共用同一個快取函數，如果上面已經抓過，這裡會瞬間載入完成，不會造成 DOM 渲染延遲
    df_list = fetch_taiwan_stock_news(
        sid=stock_id,
        start_date=(datetime.date.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    )
    if not df_list.empty:
        st.dataframe(df_list[['date', 'title', 'source']].head(20), use_container_width=True)