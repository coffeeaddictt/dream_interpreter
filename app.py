import base64, streamlit as st

# ────────────────────────── CLOUD BACKGROUND ──────────────────────────
def add_cloud_background(img: str, mode: str = "cover"):
    with open(img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    css_size   = mode if mode in ("cover", "contain") else "auto"
    css_repeat = "repeat" if mode == "repeat" else "no-repeat"
    st.markdown(f'''
    <style>
      body, .stApp {{
        background: url("data:image/png;base64,{b64}") center fixed;
        background-size: {css_size};
        background-repeat: {css_repeat};
      }}
    </style>
    ''', unsafe_allow_html=True)

add_cloud_background("dreamy_background.jpg", "cover")

# ───────────────────────── IMPORTS / SET-UP ──────────────────────────
import os, sqlite3, numpy as np, pandas as pd, soundfile as sf, json
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from openai import OpenAI

# ───────────────────────── OPENAI SETUP ─────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 Please set OPENAI_API_KEY.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

def chatgpt_sentiment(text: str):
    prompt = (
        "Classify the sentiment of the following text as Positive, Negative, or Neutral. "
        "Return a JSON object `{label:…,confidence:…}`.\n\n"
        f"Text: \"{text}\""
    )
    resp = client.chat.completions.create(
        model="gpt-4", messages=[{"role":"user","content":prompt}],
        temperature=0.0, timeout=60
    )
    return json.loads(resp.choices[0].message.content)

def map_emotion(label, conf, thresh=0.6):
    if label == "Positive" and conf >= thresh: return "happy"
    if label == "Negative" and conf >= thresh: return "stressful"
    return "neutral"

def init_db():
    conn = sqlite3.connect("dreams.db")
    conn.execute("""
      CREATE TABLE IF NOT EXISTS dreams(
        id INTEGER PRIMARY KEY,
        text TEXT,
        interpretation TEXT,
        image_url TEXT,
        emotion TEXT,
        score REAL,
        timestamp TEXT,
        language TEXT DEFAULT 'en'
      )
    """)
    return conn
conn = init_db()

def transcribe(path, lang):
    return client.audio.transcriptions.create(
        file=open(path,"rb"), model="whisper-1",
        response_format="text", language=lang,
        temperature=0.0, timeout=60
    )

def interpret(txt):
    r = client.chat.completions.create(
        model="gpt-4", temperature=0.7,
        messages=[
            {"role":"system","content":"You are an expert dream interpreter."},
            {"role":"user","content":txt}
        ],
        timeout=60
    )
    return r.choices[0].message.content.strip()

def generate_img(prompt):
    try:
        r = client.images.generate(prompt=prompt, n=1, size="1024x1024", timeout=60)
        return r.data[0].url
    except Exception:
        return None

def save_entry(txt, intp, url, emo, score, lang):
    conn.execute("""
      INSERT INTO dreams
        (text, interpretation, image_url, emotion, score, timestamp, language)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (txt, intp, url, emo, score, datetime.utcnow().isoformat(), lang))
    conn.commit()

def show_history():
    df = pd.read_sql("SELECT * FROM dreams ORDER BY timestamp DESC", conn)
    if df.empty:
        st.info("No dreams saved.")
        return
    for r in df.itertuples():
        head = f"{r.timestamp[:19]} [{r.language}] — {r.emotion} ({r.score:.2f})"
        with st.expander(head):
            st.write("**Transcript:**", r.text)
            st.write("**Interpretation:**", r.interpretation)
            if r.image_url:
                st.image(r.image_url, use_container_width=True)

class Recorder(AudioProcessorBase):
    def __init__(self):
        self.buffer = b""
        self.sample_rate = None
    def recv(self, frame):
        self.buffer += frame.to_ndarray().tobytes()
        self.sample_rate = frame.sample_rate
        return frame

# ────────────────────────── STREAMLIT UI ──────────────────────────
st.set_page_config(page_title="Dream Synthesizer", layout="centered")
st.title("Dream Synthesizer 🌙")

mode = st.sidebar.radio("Mode", ["Record Dream", "Upload Dream", "History"])
lang = st.sidebar.selectbox("Language", ["en","fr"],
    format_func=lambda x: "English" if x=="en" else "French"
)

def process_audio(path):
    txt  = transcribe(path, lang)
    intp = interpret(txt)
    res  = chatgpt_sentiment(txt)
    emo  = map_emotion(res["label"], res["confidence"])
    img  = generate_img(f"Dream-like illustration: {txt[:150]}")

    st.markdown("### ▶️ Playback");      st.audio(path)
    st.markdown("### ✍️ Transcript");    st.write(txt)
    st.markdown("### 💭 Interpretation");st.write(intp)
    st.markdown("### 📈 Mood");          st.metric("Mood", emo, f"{res['confidence']:.2f}")
    if img:
        st.markdown("### 🖼️ Generated Image"); st.image(img, use_container_width=True)

    save_entry(txt, intp, img or "", emo, res["confidence"], lang)

if mode == "Record Dream":
    st.header("Record Your Dream")
    st.markdown("<h1 style='text-align:center;font-size:4rem;margin-bottom:-1rem;'>🎤</h1>", unsafe_allow_html=True)

    # ←— SWITCHED TO SENDONLY: only audio controls, no video placeholder
    ctx = webrtc_streamer(
        key="recorder",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio":True,"video":False},
        audio_processor_factory=Recorder
    )

    if st.button("Analyze Recording"):
        rec = ctx.audio_processor
        buf = rec.buffer if rec else b""
        if not buf:
            st.error("No recording detected.")
        else:
            tmp = "rec.wav"
            sf.write(tmp, np.frombuffer(buf, dtype=np.int16), rec.sample_rate or 48000)
            process_audio(tmp)

elif mode == "Upload Dream":
    st.header("Upload Your Dream Audio")
    upl = st.file_uploader("Choose .wav or .mp3", type=["wav","mp3"])
    if upl:
        tmp = f"upl.{upl.name.rsplit('.',1)[-1]}"
        with open(tmp,"wb") as f: f.write(upl.read())
        process_audio(tmp)
    else:
        st.info("Please upload an audio file.")
else:
    show_history()
