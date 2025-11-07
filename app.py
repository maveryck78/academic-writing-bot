# app.py
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
import streamlit as st
from duckduckgo_search import DDGS
import google.generativeai as genai
from openai import OpenAI

# ---------- CONFIG ----------
st.set_page_config(page_title="Academic Writing Bot Pro", layout="centered")
st.title("Academic Writing Bot Pro")
st.caption("Grammar | Tone | Humanize | Plagiarism | AI-Check – powered by Gemini + Grok")

# API keys (use Streamlit secrets)
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
GROK_KEY   = st.secrets.get("GROK_API_KEY")

if not GEMINI_KEY or not GROK_KEY:
    st.error("API keys missing – add `GEMINI_API_KEY` and `GROK_API_KEY` in **Secrets** (sidebar → Settings → Secrets).")
    st.stop()

genai.configure(api_key=GEMINI_KEY)
gemini = genai.GenerativeModel('gemini-1.5-flash')
grok   = OpenAI(api_key=GROK_KEY, base_url="https://api.x.ai/v1")

# ---------- NLTK ----------
try:
    sent_tokenize("test")
except LookupError:
    nltk.download('punkt', quiet=True)

# ---------- HELPERS ----------
def correct_grammar(text):
    prompt = f"""You are a precise academic proofreader. Return ONLY:
1. Corrected text
2. Bullet list of changes
Text: {text}"""
    return gemini.generate_content(prompt).text

def detect_ai(text):
    prompt = f"""Score 0-100% how likely this is AI-generated. Return:
- Score: X%
- Explanation: …
Text: {text}"""
    return gemini.generate_content(prompt).text

def check_plagiarism(text):
    sentences = sent_tokenize(text)[:5]
    suspicious = []
    with DDGS() as ddgs:
        for s in sentences:
            if len(s.split()) > 12:
                try:
                    res = list(ddgs.text(f'"{s}"', max_results=3))
                    links = [r['href'] for r in res if 'wikipedia' not in r['href'].lower()]
                    if links:
                        suspicious.append((s, links[:2]))
                except:
                    pass
    if suspicious:
        out = "Potential plagiarism:\n"
        for phrase, links in suspicious:
            out += f"\n**\"{phrase}\"**\n→ {', '.join(links)}\n"
        out += "\n*Paraphrase & cite!*"
        return out
    return "No exact matches found."

def adjust_tone(text, style):
    styles = {
        "academic": "formal academic (precise, objective, no contractions)",
        "formal":   "professional business tone",
        "professional": "clear expert tone"
    }
    prompt = styles.get(style.lower(), styles["academic"])
    resp = grok.chat.completions.create(
        model="grok-3-mini",
        messages=[
            {"role":"system","content":"You are an expert academic stylist."},
            {"role":"user","content":f"Rewrite in {prompt}. Keep meaning 100% intact. Sound natural:\n{text}"}
        ],
        temperature=0.7,
        max_tokens=2048
    )
    return resp.choices[0].message.content

def humanize(text):
    resp = grok.chat.completions.create(
        model="grok-3-mini",
        messages=[
            {"role":"system","content":"Make text sound human-written: vary sentence length, natural flow."},
            {"role":"user","content":f"Humanize:\n{text}"}
        ],
        temperature=1.0
    )
    return resp.choices[0].message.content

def basic_improve(text):
    rep = {r'\bI think\b':'It is argued that', r'\ba lot\b':'numerous',
           r'\bvery\b':'highly', r'\bbig\b':'significant'}
    for p, r in rep.items():
        text = re.sub(p, r, text, flags=re.I)
    return text

# ---------- UI ----------
col1, col2 = st.columns([3,1])
with col1:
    user_text = st.text_area("Your text / command", height=150, placeholder="e.g. grammar: This are a test...")
with col2:
    style = st.selectbox("Tone", ["Academic","Formal","Professional"], index=0)

if st.button("Run", type="primary"):
    if not user_text.strip():
        st.warning("Enter some text!")
    else:
        cmd = user_text.lower().split(":",1)[0].strip() if ":" in user_text else ""
        txt = user_text[len(cmd)+1:].strip() if cmd else user_text.strip()

        with st.spinner("Working…"):
            if cmd == "grammar":
                st.subheader("Grammar Correction")
                st.code(correct_grammar(txt))
            elif cmd.startswith("tone"):
                st.subheader(f"{style} Tone")
                st.write(adjust_tone(txt, style.lower()))
            elif cmd == "humanize":
                st.subheader("Humanized")
                st.write(humanize(txt))
            elif cmd == "plagcheck":
                st.subheader("Plagiarism Check")
                st.markdown(check_plagiarism(txt))
            elif cmd == "aicheck":
                st.subheader("AI Detection")
                st.code(detect_ai(txt))
            elif cmd == "improve":
                st.subheader("Basic Tone")
                st.write(basic_improve(txt))
            else:
                # Quick actions via buttons
                st.subheader("Quick Actions")
                colA, colB, colC = st.columns(3)
                with colA:
                    if st.button("Grammar"):
                        st.code(correct_grammar(txt))
                with colB:
                    if st.button("Tone"):
                        st.write(adjust_tone(txt, style.lower()))
                with colC:
                    if st.button("Humanize"):
                        st.write(humanize(txt))
                st.markdown("---")
                st.markdown("**Tips**  \n" + "• `grammar:`  \n• `tone:`  \n• `plagcheck:`  \n• `aicheck:`")
