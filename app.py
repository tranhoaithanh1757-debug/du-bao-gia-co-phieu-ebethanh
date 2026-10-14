import warnings
warnings.filterwarnings("ignore")

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Dự báo giá cổ phiếu bằng LSTM",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# CSS tối ưu hiển thị điện thoại
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: max(1rem, 4%);
        padding-right: max(1rem, 4%);
    }
    h1 {font-size: clamp(1.8rem, 5.5vw, 3rem) !important; line-height: 1.15 !important;}
    h2 {font-size: clamp(1.35rem, 4.5vw, 2rem) !important;}
    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        border-radius: 10px;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {font-size: clamp(1.6rem, 5vw, 2.3rem);}
    [data-testid="stDataFrame"] {width: 100%;}
    @media (max-width: 600px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Mô hình Dự báo Giá Cổ phiếu bằng LSTM")
st.caption(
    "Dự báo giá đóng cửa (Close) cho 10 phiên giao dịch tiếp theo. "
    "Kết quả mang tính tham khảo, không phải khuyến nghị đầu tư."
)

# =========================
# HÀM XỬ LÝ DỮ LIỆU / MÔ HÌNH
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def download_close(ticker: str, start_date: date, end_date: date):
    # yfinance coi end là ngày không bao gồm, nên cộng thêm 1 ngày
    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Xử lý cả định dạng cột đơn và MultiIndex của yfinance
    if isinstance(df.columns, pd.MultiIndex):
        close_candidates = [c for c in df.columns if str(c[0]).lower() == "close"]
        if not close_candidates:
            return pd.DataFrame()
        close = df[close_candidates[0]]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df["Close"]

    result = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")})
    result = result.dropna()
    result.index = pd.to_datetime(result.index).tz_localize(None)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def make_sequences(values, window=60):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i, 0])
        y.append(values[i, 0])
    return np.asarray(X, dtype=np.float32).reshape(-1, window, 1), np.asarray(y, dtype=np.float32)


def train_and_forecast(df: pd.DataFrame, forecast_days: int = 10):
    values = df[["Close"]].to_numpy(dtype=np.float32)
    if len(values) < 120:
        raise ValueError(
            "Dữ liệu có ít hơn 120 phiên. Hãy chọn khoảng thời gian dài hơn để mô hình có đủ dữ liệu."
        )

    window = 60
    split = int(len(values) * 0.8)
    split = max(split, window + 20)
    if len(values) - split < 10:
        split = len(values) - 10

    # Chỉ fit scaler trên tập train để hạn chế rò rỉ dữ liệu
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(values[:split])
    all_scaled = scaler.transform(values)

    X_train, y_train = make_sequences(train_scaled, window)
    if len(X_train) < 20:
        raise ValueError("Không đủ dữ liệu huấn luyện sau khi tạo chuỗi.")

    # 10% cuối của tập train dùng làm validation theo đúng thứ tự thời gian
    val_size = max(1, int(len(X_train) * 0.1))
    X_fit, y_fit = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]

    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0,
    )

    # Tạo chuỗi test bằng 60 phiên ngay trước điểm bắt đầu test + test thực tế
    test_context = all_scaled[split - window:]
    X_test, y_test_scaled = make_sequences(test_context, window)
    if len(X_test) == 0:
        raise ValueError("Không đủ dữ liệu để đánh giá tập kiểm tra.")

    test_pred_scaled = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred_scaled).reshape(-1)
    y_test_real = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_test_real, test_pred)))
    mae = float(mean_absolute_error(y_test_real, test_pred))
    r2 = float(r2_score(y_test_real, test_pred)) if len(y_test_real) > 1 else float("nan")

    # Dự báo lặp 10 phiên tiếp theo từ 60 phiên gần nhất
    last_window = all_scaled[-window:].reshape(1, window, 1)
    future_scaled = []
    for _ in range(forecast_days):
        next_scaled = model.predict(last_window, verbose=0)[0, 0]
        future_scaled.append(next_scaled)
        next_step = np.array(next_scaled, dtype=np.float32).reshape(1, 1, 1)
        last_window = np.concatenate([last_window[:, 1:, :], next_step], axis=1)

    future_prices = scaler.inverse_transform(
        np.asarray(future_scaled).reshape(-1, 1)
    ).reshape(-1)

    return {
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "test_dates": df.index[split:],
        "test_actual": y_test_real,
        "test_pred": test_pred,
        "future_prices": future_prices,
        "latest_close": float(values[-1, 0]),
        "latest_date": df.index[-1],
    }


# =========================
# NHẬP THÔNG TIN
# =========================
with st.container(border=True):
    ticker_input = st.text_input(
        "Mã cổ phiếu (theo yfinance, ví dụ: VNM.VN, VIC.VN, AAPL)",
        value="VNM.VN",
        help="Với nhiều mã cổ phiếu Việt Nam, hãy thêm .VN, ví dụ VNM.VN hoặc FPT.VN.",
    )
    today = date.today()
    default_start = date(max(2000, today.year - 8), 1, 1)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Ngày bắt đầu", value=default_start, max_value=today)
    with col2:
        end_date = st.date_input("Ngày kết thúc", value=today, min_value=start_date, max_value=today)

    forecast_days = st.selectbox("Số phiên muốn dự báo", [5, 10, 15, 20], index=1)
    run = st.button("🚀 Chạy dự báo", type="primary", use_container_width=True)

if run:
    ticker = ticker_input.strip().upper()
    if not ticker:
        st.error("Bà hãy nhập mã cổ phiếu trước nha.")
        st.stop()
    if start_date >= end_date:
        st.error("Ngày kết thúc phải sau ngày bắt đầu.")
        st.stop()

    try:
        with st.spinner(f"Đang tải dữ liệu và huấn luyện mô hình LSTM cho {ticker}... Quá trình có thể mất vài phút."):
            df = download_close(ticker, start_date, end_date)
            if df.empty:
                raise ValueError(
                    "Không tìm thấy dữ liệu. Hãy kiểm tra mã cổ phiếu và khoảng ngày. "
                    "Ví dụ cổ phiếu Việt Nam: VNM.VN, FPT.VN, VIC.VN."
                )
            result = train_and_forecast(df, forecast_days)

        st.success(f"Đã hoàn tất dự báo cho {ticker}.")

        st.header("📌 Giá Close gần nhất")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Giá đóng cửa", f"{result['latest_close']:,.2f}")
        with metric_col2:
            st.metric("Ngày dữ liệu gần nhất", result["latest_date"].strftime("%d/%m/%Y"))

        st.header("📊 Đánh giá mô hình")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{result['rmse']:,.2f}")
        m2.metric("MAE", f"{result['mae']:,.2f}")
        m3.metric("R² Score", f"{result['r2']:.4f}")

        st.header(f"📈 Biểu đồ Giá thực tế vs Dự báo {forecast_days} phiên tới")
        # Chỉ vẽ phần lịch sử gần đây để biểu đồ dễ đọc trên điện thoại
        hist = df.tail(min(180, len(df)))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist.index, hist["Close"], label="Giá Close thật", linewidth=1.5)

        test_dates = pd.to_datetime(result["test_dates"])
        test_mask = test_dates >= hist.index[0]
        if np.any(test_mask):
            ax.plot(
                test_dates[test_mask],
                result["test_pred"][test_mask],
                label="Dự báo test",
                linewidth=1.5,
            )

        # Dùng ngày làm việc làm mốc hiển thị; không khẳng định đây là lịch giao dịch chính thức
        future_dates = pd.bdate_range(
            start=pd.Timestamp(result["latest_date"]) + pd.offsets.BDay(1),
            periods=forecast_days,
        )
        forecast_x = [pd.Timestamp(result["latest_date"])] + list(future_dates)
        forecast_y = [result["latest_close"]] + list(result["future_prices"])
        ax.plot(
            forecast_x,
            forecast_y,
            color="red",
            marker="o",
            markersize=4,
            linewidth=2,
            label=f"Dự báo {forecast_days} phiên tới",
        )
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá Close")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.header(f"📋 Bảng giá dự báo {forecast_days} phiên tới")
        forecast_table = pd.DataFrame({
            "Phiên": [f"Ngày +{i}" for i in range(1, forecast_days + 1)],
            "Ngày dự kiến": [d.strftime("%d/%m/%Y") for d in future_dates],
            "Giá Close dự báo": [round(float(x), 2) for x in result["future_prices"]],
        })
        st.dataframe(
            forecast_table,
            use_container_width=True,
            hide_index=True,
        )

        csv = forecast_table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải bảng dự báo CSV",
            data=csv,
            file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Xem biểu đồ Loss khi huấn luyện"):
            st.caption(
                "Để giữ giao diện gọn, biểu đồ Loss có thể được bổ sung trong phiên bản tiếp theo. "
                "RMSE, MAE và R² ở trên được tính trên tập kiểm tra theo thứ tự thời gian."
            )

        st.info(
            "Lưu ý: Ngày dự kiến trong bảng dùng ngày làm việc để minh họa; lịch nghỉ và ngày giao dịch "
            "thực tế có thể khác nhau. Dự báo LSTM có thể sai lệch đáng kể, đặc biệt khi thị trường biến động."
        )

    except Exception as e:
        st.error(f"Không thể chạy dự báo: {e}")
        st.caption("Nếu lỗi liên quan đến dữ liệu, hãy thử mã VNM.VN hoặc FPT.VN và kiểm tra ngày bắt đầu/kết thúc.")

else:
    st.info("Nhập mã cổ phiếu và khoảng thời gian, sau đó bấm **Chạy dự báo** để xem kết quả.")
