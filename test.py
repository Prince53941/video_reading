import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import subprocess
import imageio
import imageio_ffmpeg as ffmpeg

# ---------- Page config ----------
st.set_page_config(
    page_title="Practical Video Lab",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Video utilities ----------
def save_uploaded_video(uploaded_file):
    """Save uploaded file to a temporary location and return the path."""
    suffix = "." + uploaded_file.name.split(".")[-1] if "." in uploaded_file.name else ".mp4"
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    tfile.flush()
    return tfile.name


def load_video_reader(video_path):
    """Create an imageio video reader (FFmpeg-based)."""
    return imageio.get_reader(video_path, "ffmpeg")


def check_has_audio(video_path):
    """Check if video file has an audio stream using ffmpeg."""
    ffmpeg_binary = ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_binary, "-i", video_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # FFmpeg prints stream info to stderr; search for 'Audio:'
    return "Audio:" in proc.stderr


def _safe_int_frames(nframes):
    """Convert nframes from metadata to a safe int or None."""
    if nframes is None:
        return None
    try:
        # Sometimes nframes is float('inf') or a huge float
        if isinstance(nframes, (float, np.floating)):
            if not np.isfinite(nframes):
                return None
            if nframes > 1e9:   # absurdly large, treat as unknown
                return None
        value = int(nframes)
        if value < 0:
            return None
        return value
    except Exception:
        return None


def get_video_properties(video_reader, video_path):
    meta = video_reader.get_meta_data()

    fps = meta.get("fps", None)
    duration = meta.get("duration", None)
    nframes_raw = meta.get("nframes", None)
    nframes = _safe_int_frames(nframes_raw)
    size = meta.get("size", None)

    # Fallback for duration if not provided and data looks sane
    if duration is None and fps and nframes is not None and fps > 0:
        duration = nframes / fps

    width, height = (size if size is not None else (None, None))
    has_audio = check_has_audio(video_path)

    return {
        "Width": width,
        "Height": height,
        "Duration (s)": round(duration, 2) if duration is not None and np.isfinite(duration) else None,
        "FPS": round(fps, 2) if fps is not None and np.isfinite(fps) else None,
        "Frames": nframes,
        "Has audio": has_audio,
        "Backend": meta.get("plugin", "ffmpeg")
    }


def get_frame_image(video_reader, time_sec):
    """Get a frame (as PIL image) at a given time in seconds."""
    meta = video_reader.get_meta_data()
    fps = meta.get("fps", 1.0) or 1.0
    nframes_raw = meta.get("nframes", None)
    nframes = _safe_int_frames(nframes_raw)

    # Convert time -> frame index
    index = int(time_sec * fps)
    if nframes is not None:
        index = max(0, min(index, nframes - 1))
    else:
        index = max(0, index)

    frame = video_reader.get_data(index)  # numpy array (H,W,3) in RGB
    img = Image.fromarray(frame.astype(np.uint8))
    return img


def extract_audio_bytes(video_path, format_ext="mp3"):
    """
    Extract audio track from video and return (bytes, mime).
    Returns (None, None) if extraction fails.
    """
    ffmpeg_binary = ffmpeg.get_ffmpeg_exe()
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_ext}")
    tmp_audio.close()

    cmd = [
        ffmpeg_binary,
        "-y",
        "-i", video_path,
        "-vn",              # no video
        "-acodec", "mp3",   # MP3 audio codec
        tmp_audio.name
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        return None, None

    try:
        with open(tmp_audio.name, "rb") as f:
            data = f.read()
    except Exception:
        return None, None

    if not data:
        return None, None

    mime = "audio/mpeg"
    return data, mime

# ---------- Styling ----------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
      :root {
        --page-bg: #071029;
        --card-bg: #0d1624;
        --muted: #9fb0c8;
        --text: #eaf4ff;
        --accent: #38bdf8;
        --accent-2: #18a77a;
        --radius: 12px;
      }
      html, body, [class*="css"] {
        background-color: var(--page-bg) !important;
        color: var(--text) !important;
        font-family: "Poppins", sans-serif;
      }
      .title-main { font-size: 36px; font-weight:700; margin:0; }
      .subtitle { color: var(--muted); font-size:16px; margin-top:6px; }
      .tagline { color: var(--muted); font-size:13px; margin-bottom:18px; }
      .card { background: var(--card-bg); border-radius: var(--radius); padding: 18px; box-shadow: 0 10px 30px rgba(2,6,23,0.6); }
      .kpi { display:inline-block; margin-right:12px; padding:10px 14px; border-radius:10px; background: rgba(255,255,255,0.03); }
      .small-muted { color:var(--muted); font-size:13px; }
      .footer { text-align:center; color:var(--muted); margin-top:20px; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        "<div class='card'><h4 style='margin:0;'>Upload Video</h4>"
        "<div class='small-muted' style='margin-top:6px'>Supported: MP4, AVI, MOV, MKV</div></div>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("", type=["mp4", "avi", "mov", "mkv"])

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='card'><div class='small-muted'>About</div>"
        "<div style='margin-top:6px'>Practical Video Lab — Deep Learning (Basic Video Processing)</div>"
        "</div>",
        unsafe_allow_html=True
    )

# ---------- Header ----------
st.markdown("<div class='title-main'>Practical Video Lab</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deep Learning — Basic Video Processing</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Practical • Hands-on video tasks for coursework</div>", unsafe_allow_html=True)

# ---------- Main content ----------
if uploaded_file is None:
    st.markdown(
        "<div class='card' style='padding:46px; text-align:center;'>"
        "<h3 style='margin:0'>🎬 Upload a video to begin</h3>"
        "<p class='small-muted' style='margin-top:8px'>Start the practical by selecting a file from the sidebar.</p>"
        "</div>",
        unsafe_allow_html=True
    )
else:
    # Save file and open reader
    video_path = save_uploaded_video(uploaded_file)
    video_reader = load_video_reader(video_path)
    props = get_video_properties(video_reader, video_path)

    # KPI row
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        res_text = f"{props['Width']} × {props['Height']}" if props["Width"] and props["Height"] else "Unknown"
        st.markdown(
            f"<div class='kpi card'><b>{res_text}</b>"
            "<div class='small-muted'>Resolution</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        dur_text = f"{props['Duration (s)']} s" if props["Duration (s)"] is not None else "Unknown"
        st.markdown(
            f"<div class='kpi card'><b>{dur_text}</b>"
            "<div class='small-muted'>Duration</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        fps_text = f"{props['FPS']}" if props["FPS"] is not None else "Unknown"
        st.markdown(
            f"<div class='kpi card'><b>{fps_text}</b>"
            "<div class='small-muted'>FPS</div></div>",
            unsafe_allow_html=True
        )
    with c4:
        has_audio = "Yes" if props["Has audio"] else "No"
        st.markdown(
            f"<div class='kpi card'><b>{has_audio}</b>"
            "<div class='small-muted'>Audio Track</div></div>",
            unsafe_allow_html=True
        )

    # Tabs
    tabs = st.tabs(["Preview", "Frames", "Audio", "Properties"])

    # Preview
    with tabs[0]:
        st.markdown("<div class='card'><b>Video Preview</b></div>", unsafe_allow_html=True)
        st.video(uploaded_file)

    # Frames
    with tabs[1]:
        st.markdown("<div class='card'><b>Frame Explorer (Image part)</b></div>", unsafe_allow_html=True)
        duration = props["Duration (s)"]
        if duration is None or duration <= 0:
            st.warning("Could not determine video duration for frame selection.")
        else:
            default_time = float(duration / 2.0)
            time_sec = st.slider(
                "Select timestamp (seconds)",
                min_value=0.0,
                max_value=float(max(duration, 0.1)),
                value=float(default_time),
                step=0.5,
            )
            frame_img = get_frame_image(video_reader, time_sec)
            st.image(frame_img, caption=f"Frame at {time_sec:.2f} s", use_column_width=True)

    # Audio
    with tabs[2]:
        st.markdown("<div class='card'><b>Audio Extraction (Audio part)</b></div>", unsafe_allow_html=True)

        if not props["Has audio"]:
            st.warning("This video appears to have no audio track.")
        else:
            st.markdown(
                "<p class='small-muted'>Extract and play/download the audio from this video.</p>",
                unsafe_allow_html=True
            )

            if st.button("Extract audio as MP3"):
                with st.spinner("Extracting audio..."):
                    audio_bytes, mime = extract_audio_bytes(video_path, format_ext="mp3")

                if audio_bytes is not None:
                    st.audio(audio_bytes, format=mime)
                    st.download_button(
                        "Download audio",
                        data=audio_bytes,
                        file_name="extracted_audio.mp3",
                        mime=mime,
                    )
                else:
                    st.error("Failed to extract audio. The file may not contain an audio track or ffmpeg failed.")

    # Properties
    with tabs[3]:
        st.markdown("<div class='card'><b>Video Properties</b></div>", unsafe_allow_html=True)
        for k, v in props.items():
            st.write(f"**{k}:** {v}")

    # Close reader
    try:
        video_reader.close()
    except Exception:
        pass

# Footer
st.markdown("<div class='footer'>Built for Practical • Practical Video Lab</div>", unsafe_allow_html=True)
