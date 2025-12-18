import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.express as px
import io
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ==========================================
# CẤU HÌNH KẾT NỐI (ĐÃ CẬP NHẬT)
# ==========================================
TELEGRAM_TOKEN = "789123456:AAFlK..." # Anh hãy dán mã Token dài từ BotFather vào đây
TELEGRAM_CHAT_ID = "6786726849"          # Đã cập nhật Chat ID của anh
PIXEL_TO_MM = 0.1  # Tỷ lệ mặc định (10 pixel = 1mm)

st.set_page_config(page_title="AI Civil Inspection - Mobile Scanner", layout="wide")

# Hàm tải mô hình AI
@st.cache_resource
def load_yolo_model():
    return YOLO("crack_detector_model.pt")

# Hàm gửi báo cáo qua Telegram
def send_telegram(image_rgb, message):
    try:
        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        io_buf = io.BytesIO(buffer)
        io_buf.name = 'crack_report.jpg'
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': io_buf}
        data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message, 'parse_mode': 'Markdown'}
        requests.post(url, files=files, data=data)
        return True
    except Exception as e:
        st.error(f"Lỗi gửi Telegram: {e}")
        return False

# Logic đánh giá theo TCVN 9381:2012
def diagnose_tcvn(w):
    if w <= 0.2:
        return "Cấp A (An toàn)", "Không", "Ổn định. Nứt co ngót bề mặt.", "🟢"
    elif 0.2 < w <= 0.5:
        return "Cấp B (Theo dõi)", "Nguy cơ cao", "Có dấu hiệu thấm. Theo dõi độ võng.", "🟡"
    elif 0.5 < w <= 1.5:
        return "Cấp C (Nguy hiểm)", "Rất cao", "Rủi ro PHÁ HOẠI. Khả năng đã võng sàn.", "🟠"
    else:
        return "Cấp D (Khẩn cấp)", "Cực kỳ cao", "NGUY CƠ SẬP ĐỔ. Cần gia cường ngay!", "🔴"

# Bộ xử lý Video Live
class CrackProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)
        annotated_frame = results[0].plot()
        
        for box in results[0].boxes:
            w_mm = round(box.xywh[0][2].item() * PIXEL_TO_MM, 2)
            _, level, _, _ = diagnose_tcvn(w_mm)
            cv2.putText(annotated_frame, f"{w_mm}mm-{level}", 
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Giao diện chính
model = load_yolo_model()
st.title("🏗️ AI Civil Inspection - Mobile Scanner")

with st.sidebar:
    st.header("📋 Thông tin hiện trường")
    eng_name = st.text_input("Kỹ sư", "Kỹ sư công trường")
    proj_name = st.text_input("Dự án", "Dự án kiểm tra")
    ele_id = st.text_input("Mã cấu kiện", "Dầm/Sàn")
    mode = st.radio("Chế độ", ["📱 Quét trực tiếp (Live)", "📸 Chụp ảnh gửi báo cáo"])

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Thời gian", "Cấu kiện", "Rộng (mm)", "Cấp độ"])

if mode == "📱 Quét trực tiếp (Live)":
    webrtc_streamer(
        key="crack-live",
        video_processor_factory=CrackProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False}
    )

else:
    img_file = st.camera_input("Chụp ảnh vết nứt")
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        results = model(img)
        annotated_img = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        st.image(annotated_rgb, caption="Ảnh chẩn đoán", use_container_width=True)
        
        crack_list = []
        for box in results[0].boxes:
            w_mm = round(box.xywh[0][2].item() * PIXEL_TO_MM, 2)
            level, seepage, warning, icon = diagnose_tcvn(w_mm)
            crack_list.append([icon, w_mm, level, seepage, warning])
        
        if crack_list:
            df = pd.DataFrame(crack_list, columns=[" ","Rộng(mm)", "Cấp độ", "Thấm", "Cảnh báo"])
            st.table(df)
            max_w = df["Rộng(mm)"].max()
            best_diag = df.loc[df["Rộng(mm)"].idxmax()]
            
            if st.button("📤 GỬI BÁO CÁO VỀ TELEGRAM"):
                msg = f"🏗️ *Dự án:* {proj_name}\n🔧 *Cấu kiện:* {ele_id}\n📏 *Rộng max:* {max_w}mm\n📊 *Trạng thái:* {best_diag['Cấp độ']}\n⚠️ *Cảnh báo:* {best_diag['Cảnh báo']}\n👷 *Kỹ sư:* {eng_name}"
                if send_telegram(annotated_rgb, msg):
                    st.success("✅ Đã gửi báo cáo về điện thoại!")
                
                new_data = pd.DataFrame([[datetime.now(), ele_id, max_w, best_diag['Cấp độ']]], columns=st.session_state.history.columns)
                st.session_state.history = pd.concat([st.session_state.history, new_data])