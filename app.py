import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# -------------------- Cấu hình trang --------------------
st.set_page_config(
    page_title="AI Stock Forecast | LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Màu sắc và giao diện gần với ứng dụng mẫu: đơn giản, tập trung vào dự báo.
st.markdown("""
<style>
    .stApp { background: #f7f9fc; }
    [data-testid="stHeader"] { background: rgba(247,249,252,0.9); }
    .hero {
        background: linear-gradient(120deg, #092447 0%, #164e83 100%);
        padding: 1.35rem 1.6rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 1.85rem; }
    .hero p { color: #e2edf8; margin: .45rem 0 0 0; }
    .metric-card {
        background: white; padding: 1rem 1.1rem; border-radius: 12px;
        border: 1px solid #e4eaf2; min-height: 104px;
    }
    .metric-label { color: #5b6b7f; font-size: .88rem; }
    .metric-value { color: #102d4d; font-size: 1.45rem; font-weight: 750; margin-top: .25rem; }
    .small-note { color: #617185; font-size: .88rem; }
    div.stButton > button[kind="primary"] {
        background: #145da0; border: 0; border-radius: 9px; font-weight: 650;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📈 ỨNG DỤNG DỰ BÁO GIÁ CỔ PHIẾU BẰNG AI (LSTM)</h1>
  <p>Tải dữ liệu lịch sử • Huấn luyện mô hình LSTM • Dự báo các phiên tiếp theo • Đánh giá sai số</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Hàm tiện ích --------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_stock_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Tải dữ liệu lịch sử từ Yahoo Finance. end_date được chuyển thành ngày kế tiếp vì yfinance loại trừ ngày kết thúc."""
    raw = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance có thể trả về MultiIndex tùy phiên bản/mã cổ phiếu.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
        else:
            return pd.DataFrame()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        close = raw["Close"]

    result = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")})
    result.index = pd.to_datetime(result.index)
    result = result.dropna()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def make_sequences(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i, 0])
        y.append(values[i, 0])
    return np.asarray(X, dtype=np.float32).reshape(-1, window, 1), np.asarray(y, dtype=np.float32)


def train_and_forecast(
    close_prices: np.ndarray,
    window: int,
    epochs: int,
    batch_size: int,
    future_days: int,
    train_ratio: float,
):
    """Chia train/test theo thời gian; scaler chỉ fit trên tập train để hạn chế rò rỉ dữ liệu."""
    n = len(close_prices)
    split = int(n * train_ratio)
    if split <= window or n - split < 5:
        raise ValueError("Dữ liệu chưa đủ dài cho cấu hình hiện tại. Hãy chọn khoảng thời gian dài hơn.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_prices[:split])
    scaled_all = scaler.transform(close_prices)

    X_all, y_all = make_sequences(scaled_all, window)
    # y_all index corresponds to original observation window..n-1.
    target_indices = np.arange(window, n)
    train_mask = target_indices < split
    test_mask = target_indices >= split

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Không đủ mẫu train/test. Hãy chọn khoảng thời gian dài hơn.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.1),
        LSTM(50, return_sequences=False),
        Dropout(0.1),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    y_test_price = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    pred_price = scaler.inverse_transform(pred_scaled).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_test_price, pred_price)))
    mae = float(mean_absolute_error(y_test_price, pred_price))
    r2 = float(r2_score(y_test_price, pred_price)) if len(y_test_price) >= 2 else float("nan")

    # Dự báo đệ quy: mỗi dự báo tiếp theo dùng kết quả vừa dự báo.
    recent = scaled_all[-window:].reshape(1, window, 1).copy()
    future_scaled = []
    for _ in range(future_days):
        next_value = float(model.predict(recent, verbose=0)[0, 0])
        future_scaled.append(next_value)
        next_step = np.array(next_value, dtype=np.float32).reshape(1, 1, 1)
        recent = np.concatenate([recent[:, 1:, :], next_step], axis=1)
    future_price = scaler.inverse_transform(np.asarray(future_scaled).reshape(-1, 1)).ravel()

    return {
        "model": model,
        "history": history,
        "test_actual": y_test_price,
        "test_pred": pred_price,
        "test_dates": None,  # được gắn theo chỉ số ở phía gọi
        "future_price": future_price,
        "metrics": {"RMSE": rmse, "MAE": mae, "R²": r2},
        "split": split,
    }


def money(value):
    return f"{value:,.2f}"


# -------------------- Thanh bên --------------------
with st.sidebar:
    st.header("⚙️ Thiết lập dự báo")
    ticker = st.text_input("Mã cổ phiếu", value="ACB.VN", help="Ví dụ: ACB.VN, VNM.VN, FPT.VN, HPG.VN hoặc AAPL.")
    today = date.today()
    default_start = date(max(2000, today.year - 7), 1, 1)
    start_date = st.date_input("Từ ngày", value=default_start, max_value=today - timedelta(days=1))
    end_date = st.date_input("Đến ngày", value=today, min_value=start_date, max_value=today)
    st.divider()
    window = st.slider("Số phiên lịch sử đầu vào (lookback)", min_value=20, max_value=120, value=60, step=10)
    epochs = st.slider("Số epoch tối đa", min_value=5, max_value=30, value=10, step=5)
    batch_size = st.selectbox("Batch size", options=[16, 32, 64], index=1)
    future_days = st.slider("Số phiên cần dự báo", min_value=1, max_value=20, value=10)
    train_ratio = st.slider("Tỷ lệ dữ liệu huấn luyện", min_value=0.70, max_value=0.90, value=0.80, step=0.05)
    run_button = st.button("🚀 Tải dữ liệu & Chạy dự báo", type="primary", use_container_width=True)
    st.caption("Nguồn dữ liệu: Yahoo Finance (thông qua yfinance). Không cần API key.")

# -------------------- Nội dung chính --------------------
st.subheader("Tổng quan")
st.write("Nhập mã cổ phiếu ở thanh bên trái, chọn thời gian và nhấn **Tải dữ liệu & Chạy dự báo**.")
st.markdown(
    '<p class="small-note">Mô hình học từ chuỗi giá đóng cửa. Kết quả là ước lượng tham khảo, không phải khuyến nghị mua/bán.</p>',
    unsafe_allow_html=True
)

if run_button:
    cleaned_ticker = ticker.strip().upper()
    if not cleaned_ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()
    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    with st.spinner(f"Đang tải dữ liệu cho {cleaned_ticker}..."):
        try:
            data = load_stock_data(cleaned_ticker, start_date, end_date)
        except Exception as exc:
            st.error(f"Không tải được dữ liệu: {exc}")
            st.stop()

    if data.empty:
        st.error("Không tìm thấy dữ liệu. Kiểm tra mã cổ phiếu và khoảng thời gian. Với cổ phiếu Việt Nam, thử định dạng như ACB.VN hoặc FPT.VN.")
        st.stop()

    if len(data) < max(100, window + 20):
        st.error(f"Chỉ có {len(data)} phiên dữ liệu. Hãy chọn khoảng thời gian dài hơn hoặc giảm lookback.")
        st.stop()

    prices = data["Close"].to_numpy(dtype=np.float32).reshape(-1, 1)
    with st.spinner("Đang huấn luyện mô hình LSTM và tính toán dự báo. Lần đầu có thể mất vài phút..."):
        try:
            result = train_and_forecast(prices, window, epochs, batch_size, future_days, train_ratio)
        except Exception as exc:
            st.error(f"Không thể huấn luyện mô hình: {exc}")
            st.stop()

    test_start = result["split"]
    test_dates = data.index[test_start:]
    # Defensive alignment in case sequence count differs due to windowing.
    actual = result["test_actual"]
    pred = result["test_pred"]
    test_dates = test_dates[-len(actual):]

    st.success(f"Đã hoàn tất dự báo cho {cleaned_ticker}. Dữ liệu gồm {len(data):,} phiên giao dịch.")
    latest_close = float(data["Close"].iloc[-1])
    predicted_last = float(result["future_price"][-1])
    pct_change = (predicted_last / latest_close - 1) * 100 if latest_close else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Giá đóng cửa gần nhất</div><div class="metric-value">{money(latest_close)}</div><div class="small-note">{data.index[-1].strftime("%d/%m/%Y")}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Dự báo cuối kỳ</div><div class="metric-value">{money(predicted_last)}</div><div class="small-note">Sau {future_days} phiên dự báo</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Thay đổi ước tính</div><div class="metric-value">{pct_change:+.2f}%</div><div class="small-note">So với giá đóng cửa gần nhất</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">RMSE kiểm tra</div><div class="metric-value">{money(result["metrics"]["RMSE"])}</div><div class="small-note">Càng thấp thường càng tốt</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📉 Giá & dự báo", "🧪 Đánh giá mô hình", "📋 Bảng dự báo", "🗂️ Dữ liệu lịch sử"])

    with tab1:
        st.markdown("#### Giá đóng cửa lịch sử và dự báo tương lai")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data.index, data["Close"], label="Giá đóng cửa lịch sử", linewidth=1.6)
        future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
        ax.plot(future_dates, result["future_price"], label="Giá dự báo LSTM", marker="o", linewidth=2)
        ax.axvline(data.index[-1], linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá đóng cửa")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Các ngày tương lai trên biểu đồ là ngày làm việc ước tính; lịch nghỉ lễ và ngày giao dịch thực tế có thể khác.")

        st.markdown("#### Giá thực tế so với giá mô hình dự báo trên tập kiểm tra")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(test_dates, actual, label="Giá thực tế", linewidth=1.5)
        ax2.plot(test_dates, pred, label="Giá dự báo", linewidth=1.5)
        ax2.set_xlabel("Ngày")
        ax2.set_ylabel("Giá đóng cửa")
        ax2.grid(True, alpha=0.25)
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with tab2:
        st.markdown("#### Chỉ số đánh giá trên tập kiểm tra")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", money(result["metrics"]["RMSE"]))
        m2.metric("MAE", money(result["metrics"]["MAE"]))
        m3.metric("R²", f'{result["metrics"]["R²"]:.4f}')
        st.write("""
        - **RMSE**: căn bậc hai của sai số bình phương trung bình; phạt sai số lớn mạnh hơn.
        - **MAE**: sai số tuyệt đối trung bình, tính theo đơn vị giá của mã cổ phiếu.
        - **R²**: mức độ phù hợp của dự báo so với biến động thực tế; có thể âm nếu mô hình dự báo kém hơn mốc trung bình.
        """)
        st.markdown("#### Loss trong quá trình huấn luyện")
        hist = result["history"].history
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(hist.get("loss", []), label="Training loss")
        if hist.get("val_loss"):
            ax3.plot(hist["val_loss"], label="Validation loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Mean squared error")
        ax3.grid(True, alpha=0.25)
        ax3.legend()
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        st.info("Đây là mô hình minh họa học từ giá đóng cửa, chưa sử dụng tin tức, báo cáo tài chính, thanh khoản hay các cú sốc thị trường.")

    with tab3:
        forecast_df = pd.DataFrame({
            "Phiên dự báo": np.arange(1, future_days + 1),
            "Ngày dự kiến": future_dates.strftime("%d/%m/%Y"),
            "Giá dự báo": result["future_price"],
        })
        forecast_df["Thay đổi so với giá gần nhất (%)"] = (forecast_df["Giá dự báo"] / latest_close - 1) * 100
        st.dataframe(
            forecast_df.style.format({
                "Giá dự báo": "{:,.2f}",
                "Thay đổi so với giá gần nhất (%)": "{:+.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = forecast_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Tải bảng dự báo CSV", data=csv_bytes, file_name=f"du_bao_{cleaned_ticker.replace('.', '_')}.csv", mime="text/csv")

    with tab4:
        st.dataframe(data.tail(100).rename_axis("Ngày"), use_container_width=True)
        csv_hist = data.to_csv().encode("utf-8-sig")
        st.download_button("⬇️ Tải dữ liệu lịch sử CSV", data=csv_hist, file_name=f"lich_su_{cleaned_ticker.replace('.', '_')}.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("**Cách diễn giải nhanh:** Đọc RMSE/MAE cùng biểu đồ dự báo trên tập kiểm tra; không kết luận mô hình tốt chỉ dựa vào một chỉ số. Dự báo nhiều phiên liên tiếp có thể tích lũy sai số.")
else:
    st.markdown("### Quy trình hoạt động")
    cols = st.columns(4)
    steps = [
        ("01", "Tải dữ liệu", "Lấy giá đóng cửa lịch sử từ Yahoo Finance."),
        ("02", "Chuẩn hóa", "Đưa dữ liệu về thang 0–1 và tạo chuỗi lookback."),
        ("03", "Huấn luyện LSTM", "Dùng dữ liệu quá khứ để học mẫu biến động."),
        ("04", "Dự báo & đánh giá", "Vẽ biểu đồ, tính RMSE/MAE/R² và xuất bảng."),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">BƯỚC {num}</div>
              <div style="font-size:1.05rem;font-weight:700;color:#102d4d;margin:.35rem 0">{title}</div>
              <div class="small-note">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Mã cổ phiếu tham khảo")
    st.write("**Việt Nam:** `ACB.VN`, `VNM.VN`, `FPT.VN`, `HPG.VN`  ·  **Quốc tế:** `AAPL`, `MSFT`, `NVDA`")
    st.warning("Kết quả chỉ phục vụ học tập và minh họa kỹ thuật dự báo, không phải lời khuyên đầu tư. Dữ liệu và khả năng tải mã cổ phiếu phụ thuộc Yahoo Finance.")
