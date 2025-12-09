import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import io
from moviepy.editor import VideoFileClip

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

def get_video_clip(video_path):
    """Create a MoviePy VideoFileClip."""
    return VideoFileClip(video_path)

def get_video_properties(clip):
    fps = getattr(clip, "fps", None) or getattr(clip.reader, "fps", None)
    duration = clip.duration if clip.duration is not None else 0
    frame_count = int(fps * duration) if fps and duration else None

    return {
        "Width": clip.w,
        "Height": clip.h,
        "Duration (s)": round(duration, 2),
        "FPS": round(fps, 2) if fps else None,
        "Frames": frame_count,
        "Has audio": clip.audio is not None
    }

def get_frame_image(clip, time_sec):
    """Get a frame (as PIL image) at a given time in seconds."""
    time_sec = max(0, min(time_sec, max(clip.duration - 1e-3, 0)))
    frame = clip.get_frame(time_sec)  # RGB numpy array
    img = Image.fromarray(frame)
    return img

def extract_audio_bytes(clip, format_ext="mp3"):
    """
    Extract audio from clip and return bytes.
    Requires ffmpeg to be available in the environment.
    """
    if clip.audio is None:
        return None, None

    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_ext}")
    tmp_audio.close()

    # Suppress verbose moviepy logger
    clip.audio.write_audiofile(tmp_audio.name, logger=None)

    with open(tmp_audio.name, "rb") as f:
        data = f.read()

    mime = "audio/mpeg" if format_ext == "mp3" else "audio/wav"
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
    # Save and open video
    video_path = save_uploaded_video(uploaded_file)
    clip = get_video_clip(video_path)
    props = get_video_properties(clip)

    # KPI row
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        st.markdown(
            f"<div class='kpi card'><b>{props['Width']} × {props['Height']}</b>"
            "<div class='small-muted'>Resolution</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='kpi card'><b>{props['Duration (s)']} s</b>"
            "<div class='small-muted'>Duration</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='kpi card'><b>{props['FPS']}</b>"
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
        # Use original uploaded file object for preview
        st.video(uploaded_file)

    # Frames
    with tabs[1]:
        st.markdown("<div class='card'><b>Frame Explorer (Image part)</b></div>", unsafe_allow_html=True)

        duration = props["Duration (s)"] or 0.0
        if duration <= 0:
            st.warning("Could not determine video duration.")
        else:
            default_time = float(duration / 2.0)
            time_sec = st.slider(
                "Select timestamp (seconds)",
                min_value=0.0,
                max_value=float(max(duration, 0.1)),
                value=float(default_time),
                step=0.5,
            )
            frame_img = get_frame_image(clip, time_sec)
            st.image(frame_img, caption=f"Frame at {time_sec:.2f} s", use_column_width=True)

    # Audio
    with tabs[2]:
        st.markdown("<div class='card'><b>Audio Extraction (Audio part)</b></div>", unsafe_allow_html=True)

        if not props["Has audio"]:
            st.warning("This video has no audio track.")
        else:
            st.markdown(
                "<p class='small-muted'>Extract and play/download the audio from this video.</p>",
                unsafe_allow_html=True
            )

            if st.button("Extract audio as MP3"):
                with st.spinner("Extracting audio..."):
                    audio_bytes, mime = extract_audio_bytes(clip, format_ext="mp3")

                if audio_bytes is not None:
                    st.audio(audio_bytes, format=mime)
                    st.download_button(
                        "Download audio",
                        data=audio_bytes,
                        file_name="extracted_audio.mp3",
                        mime=mime,
                    )
                else:
                    st.error("Failed to extract audio.")

    # Properties
    with tabs[3]:
        st.markdown("<div class='card'><b>Video Properties</b></div>", unsafe_allow_html=True)
        for k, v in props.items():
            st.write(f"**{k}:** {v}")

# Footer
st.markdown("<div class='footer'>Built for Practical • Practical Video Lab</div>", unsafe_allow_html=True)
