# Website dự báo giá cổ phiếu bằng AI (LSTM)

Ứng dụng Streamlit dành cho bài tập môn Tài chính tiền tệ. Ứng dụng tải giá đóng cửa lịch sử qua `yfinance`, huấn luyện mô hình LSTM, dự báo các phiên tiếp theo và báo cáo RMSE, MAE, R².

## Chạy trên máy tính

Yêu cầu: Python 3.11 (khuyến nghị) và kết nối Internet.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Cài thư viện và chạy:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Trình duyệt thường mở tại `http://localhost:8501`.

## Triển khai thành website online miễn phí

Bạn cần có tài khoản GitHub và Streamlit Community Cloud. Mình không thể tự đăng nhập tài khoản hoặc xuất bản website thay bạn; hãy thực hiện các bước dưới đây để nhận URL công khai.

1. Giải nén thư mục này.
2. Đăng nhập GitHub và tạo repository mới, ví dụ `ai-stock-forecast-lstm`.
3. Tải lên `app.py`, `requirements.txt`, `README.md` và thư mục `.streamlit` (có thể bỏ README nếu muốn).
4. Mở https://share.streamlit.io/ và kết nối tài khoản GitHub.
5. Chọn **Create app** → repository vừa tạo → branch `main` → Main file path `app.py`.
6. Trong **Advanced settings**, chọn Python **3.11** nếu có.
7. Nhấn **Deploy** và chờ cài đặt thư viện/khởi động ứng dụng. Lần đầu có thể mất vài phút do TensorFlow.
8. Khi ứng dụng chạy thành công, Streamlit sẽ cấp URL dạng `https://ten-ung-dung.streamlit.app`. Mở thử URL, nhập `ACB.VN`, chọn thời gian và bấm nút chạy.
9. Nếu lỗi khởi động, mở trang quản lý ứng dụng và xem log. Nếu lỗi liên quan bộ nhớ hoặc TensorFlow, thử khoảng thời gian ngắn hơn hoặc cân nhắc môi trường triển khai khác.

Tài liệu chính thức: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Cách sử dụng

- Mã Việt Nam thường có dạng `ACB.VN`, `FPT.VN`, `VNM.VN`, `HPG.VN`.
- Chọn khoảng thời gian đủ dài; mô hình cần tối thiểu khoảng 100 phiên dữ liệu.
- Chọn lookback (mặc định 60 phiên), số epoch và số phiên dự báo.
- Nhấn **Tải dữ liệu & Chạy dự báo**.
- Xem tab giá/dự báo, đánh giá mô hình, bảng dự báo và dữ liệu lịch sử.
- Có thể tải bảng dự báo hoặc dữ liệu lịch sử dạng CSV.

## Mô hình và giới hạn

- Đầu vào: chuỗi giá đóng cửa.
- Cấu trúc: LSTM(50) → Dropout → LSTM(50) → Dropout → Dense → Dense(1).
- Chia dữ liệu theo thứ tự thời gian; scaler được fit trên phần train.
- Dự báo nhiều bước theo kiểu đệ quy, nên sai số có thể tích lũy.
- RMSE và MAE được tính trên tập kiểm tra. R² có thể âm.
- Mô hình không dùng tin tức, chỉ báo kỹ thuật, báo cáo tài chính, thanh khoản hoặc yếu tố vĩ mô. Kết quả không phải lời khuyên mua/bán.
- `yfinance` là thư viện bên thứ ba và dữ liệu có thể thiếu hoặc mã cổ phiếu có thể không được hỗ trợ tại một số thời điểm.

## Cấu trúc

```text
website_du_bao_co_phieu_lstm/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```
