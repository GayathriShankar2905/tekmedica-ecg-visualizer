import streamlit as st
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, detrend, find_peaks, resample
import matplotlib.pyplot as plt

# ======================================
# PAGE CONFIGURATION
# ======================================
st.set_page_config(
    page_title="TekMedica ECG Virtual Kit",
    layout="wide",
)

# ======================================
# CUSTOM CSS (Virtual Lab Theme)
# ======================================
st.markdown("""
    <style>
    /* Background Gradient */
    body {
        background: linear-gradient(to right, #f0f9ff, #cbebff, #e0f7fa);
        font-family: 'Poppins', sans-serif;
    }

    /* Header Styling */
    .main-header {
        text-align: center;
        color: #003366;
        font-weight: bold;
        line-height: 1.4;
        font-size: 20px;
        background: #e6f2ff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }

    .sub-header {
        text-align: center;
        color: #0077b6;
        font-size: 28px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 20px;
        letter-spacing: 1px;
    }

    /* Section Card Style */
    .card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

    /* Buttons */
    .stDownloadButton > button {
        background-color: #0077b6 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stDownloadButton > button:hover {
        background-color: #005f8d !important;
    }

    /* Chart borders */
    .stPlotlyChart, .stAltairChart, .stVegaLiteChart, .stLineChart {
        border: 1px solid #cce7ff;
        border-radius: 10px;
        background-color: #f8fcff;
        padding: 10px;
    }

    </style>
""", unsafe_allow_html=True)

# ======================================
# HEADER
# ======================================
st.markdown("""
<div class="main-header">
    <h3>SRM Institute of Science and Technology</h3>
    <h4>College of Engineering and Technology | School of Bioengineering</h4>
    <h4>Department of Biomedical Engineering</h4>
</div>
<h2 class="sub-header">🩺 TekMedica Club – ECG Virtual Analysis Kit</h2>
""", unsafe_allow_html=True)

# ======================================
# HELPER FUNCTIONS
# ======================================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=45, fs=250):
    b, a = butter_bandpass(lowcut, highcut, fs, order=4)
    return filtfilt(b, a, data)

def normalize_signal(data):
    return (data - np.mean(data)) / np.std(data)

def detect_r_peaks(signal, fs=250):
    peaks, _ = find_peaks(signal, distance=0.25*fs, prominence=np.std(signal)*0.8)
    return peaks

def compute_hrv(peaks, fs=250):
    rr_intervals = np.diff(peaks) / fs
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    heart_rate = 60.0 / mean_rr if mean_rr > 0 else 0
    return rr_intervals, {"Mean RR (s)": mean_rr, "SDNN (s)": sdnn, "RMSSD (s)": rmssd, "Heart Rate (bpm)": heart_rate}

def st_segment_analysis(signal, r_peaks, fs=250):
    st_values = []
    offset = int(0.08 * fs)
    window = int(0.12 * fs)
    for r in r_peaks:
        start = r + offset
        end = r + offset + window
        if end < len(signal):
            st_values.append(np.mean(signal[start:end]))
    return np.mean(st_values) if len(st_values) > 0 else 0

# ======================================
# FILE UPLOAD SECTION
# ======================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Upload ECG File")
st.write("Upload your `.mat` ECG data file to begin analysis:")

uploaded_file = st.file_uploader("Upload ECG `.mat` file", type=["mat"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    mat_data = sio.loadmat(uploaded_file)
    keys = [k for k in mat_data.keys() if not k.startswith("__")]
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔑 Available Keys")
    st.write(keys)
    selected_key = st.selectbox("Select ECG signal key:", keys)
    st.markdown('</div>', unsafe_allow_html=True)

    if selected_key:
        signal = np.squeeze(mat_data[selected_key])
        if signal.ndim > 1:
            lead_idx = st.selectbox("Select Lead:", range(signal.shape[0]))
            signal = signal[lead_idx, :]
            st.info(f"Analyzing Lead {lead_idx + 1}")

        fs = st.number_input("Sampling Frequency (Hz)", value=250, step=50)

        # Step 1: Raw ECG
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1️⃣ Raw ECG Signal")
        st.line_chart(signal)
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 2: Resample
        new_fs = st.number_input("Resample to (Hz)", value=250, step=50)
        if new_fs != fs:
            signal = resample(signal, int(len(signal) * new_fs / fs))
            fs = new_fs

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2️⃣ Detrending")
        detrended = detrend(signal)
        st.line_chart(detrended)
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 3: Filtering
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("3️⃣ Bandpass Filtering (0.5–45 Hz)")
        filtered = bandpass_filter(detrended, 0.5, 45, fs)
        st.line_chart(filtered)
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 4: Normalization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("4️⃣ Normalized ECG Signal")
        normalized = normalize_signal(filtered)
        st.line_chart(normalized)
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 5: R-Peak Detection
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("5️⃣ R-Peak Detection")
        r_peaks = detect_r_peaks(normalized, fs)
        fig, ax = plt.subplots()
        ax.plot(normalized, label="ECG")
        ax.plot(r_peaks, normalized[r_peaks], "ro", label="R-peaks")
        ax.legend()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 6: ST-Segment
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("6️⃣ ST-Segment Analysis")
        st_val = st_segment_analysis(normalized, r_peaks, fs)
        st.metric(label="Average ST-Segment Elevation", value=f"{st_val:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 7: HRV
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("7️⃣ Heart Rate Variability (HRV) Analysis")
        rr_intervals, hrv_metrics = compute_hrv(r_peaks, fs)
        st.write(hrv_metrics)
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 8: Poincaré Plot
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("8️⃣ Poincaré Plot (RRn vs RRn+1)")
        fig2, ax2 = plt.subplots()
        ax2.scatter(rr_intervals[:-1], rr_intervals[1:], alpha=0.7, color="#0077b6")
        ax2.set_xlabel("RRn (s)")
        ax2.set_ylabel("RRn+1 (s)")
        ax2.set_title("Poincaré Plot")
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 9: Final Vector
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✅ Final Vector for ML Input")
        st.write("Shape:", normalized.shape)
        st.write(normalized)

        csv_data = pd.DataFrame(normalized, columns=["ECG_Normalized"])
        csv = csv_data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Preprocessed ECG CSV", csv, "ecg_processed.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Upload a .mat file to begin.")
