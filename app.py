# app.py - AI Shorts Assistant - Clip Finder Version
# This version finds potential clips and lets the user play them directly from YouTube.

import os
import re
import tempfile
import streamlit as st
import traceback

# Simplified dependencies: moviepy, yt-dlp, and gdown are no longer needed.
from openai import OpenAI, BadRequestError as OpenAIBadRequestError
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as GoogleAPIErrors
import docx  # Requires python-docx

# ---
# 1. SYSTEM PROMPT (Remains the same - core logic is sound)
# ---
SYSTEM_PROMPT = """
You are an expert YouTube Shorts strategist and video editor.

Your job is to analyze the full transcript of a long-form interview or podcast and extract powerful 30–60 second Shorts using two formats:
1. Direct Clips — continuous timestamp segments that tell a complete story.
2. Franken-Clips — stitched from non-contiguous timestamps, using a hook from one part and payoff from another.

---

🛑 STRICT RULE: DO NOT REWRITE OR SUMMARIZE ANY DIALOGUE.

You must:
- Use the transcript lines exactly as they appear in the provided SRT/transcript.
- Do not shorten, reword, paraphrase, or compress the speaker's sentences.
- Keep all original punctuation, phrasing, and spelling.
- Only include full dialogue blocks — no cherry-picking fragments from within a block.
- ALWAYS provide EXACT timestamps in HH:MM:SS,mmm format (e.g., 00:01:23,450)

The output should allow a video editor to directly cut the clip using the given timestamps and script.

---

📌 ANALYSIS GOALS:
- Deeply read and understand the entire transcript before selecting Shorts.
- Prioritize clips with emotional, insightful, or surprising moments.
- Each Short must follow a story arc (hook → context → insight → takeaway).
- Do not suggest clips unless they feel self-contained and high-retention.

---

🎯 THEMES TO PRIORITIZE:
- Money, fame, or behind-the-scenes industry truths
- Firsts and breakthroughs (first paycheck, big break, first failure)
- Vulnerability: burnout, fear, comparison, loneliness
- Transformation: then vs now
- Hacks or hard-earned lessons
- Breaking stereotypes or taboos

---

🛠 HOW TO BUILD FRANKEN-CLIPS:
- Start with a strong hook from any timestamp.
- Skip filler or weak replies.
- Jump to the later timestamp where the real answer, story, or insight is delivered.
- Stitch together in timestamp order.
- Ensure the whole story makes sense even though the timestamps are non-contiguous.

---

📦 OUTPUT FORMAT (repeat for each Short):

**Short Title:** [Catchy title with emoji]
**Estimated Duration:** [e.g., 42 seconds]
**Type:** [Direct Clip / Franken-Clip]

**Timestamps:**
START: 00:01:23,450 --> END: 00:01:35,200
[For Franken-clips, list multiple timestamp ranges]

**Script:**
[Exact dialogue from transcript - no modifications]

**Rationale:**
[Brief explanation why this will go viral]

---

🛑 CRITICAL REMINDERS:
- Provide EXACT timestamps that match the SRT format
- Do not modify any dialogue
- Ensure timestamps are accurate and complete
- Each clip should be 30-60 seconds total

Generate the requested number of shorts following this exact format.
"""

# ---
# 2. HELPER FUNCTIONS
# ---

def get_openai_api_key() -> str:
    return st.secrets.get("openai", {}).get("api_key", "")

def get_google_api_key() -> str:
    return st.secrets.get("google_ai", {}).get("api_key", "")

@st.cache_data(show_spinner=False)
def fetch_openai_models(api_key: str):
    default_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    if not api_key: return default_models
    try:
        client = OpenAI(api_key=api_key)
        models = [m.id for m in client.models.list().data if m.id.startswith("gpt-")]
        return sorted(list(set(models + default_models)))
    except Exception:
        st.error("Could not fetch OpenAI models. Using default list.")
        return default_models

@st.cache_data(show_spinner=False)
def fetch_gemini_models(api_key: str):
    default_models = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro"]
    if not api_key: return default_models
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name]
        model_ids = [m.split('/')[-1] for m in models]
        return sorted(list(set(model_ids + default_models))) if model_ids else default_models
    except Exception:
        st.error("Could not fetch Gemini models. Using default list.")
        return default_models

def read_transcript_file(uploaded_file) -> str:
    try:
        if uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def parse_srt_timestamp(timestamp_str: str) -> int:
    """Convert SRT timestamp format to total seconds."""
    timestamp_str = timestamp_str.strip().replace(',', '.')
    try:
        time_parts = timestamp_str.split(':')
        if len(time_parts) == 3:
            h, m, s = map(float, time_parts)
            return int(h * 3600 + m * 60 + s)
        elif len(time_parts) == 2:
            m, s = map(float, time_parts)
            return int(m * 60 + s)
        return int(float(time_parts[0]))
    except Exception:
        return 0

def analyze_transcript_with_llm(transcript: str, count: int, model_name: str, provider_name: str):
    user_content = f"{transcript}\n\nPlease generate {count} unique potential shorts following the exact format specified."
    
    if provider_name == "OpenAI":
        api_key = get_openai_api_key()
        if not api_key:
            st.error("OpenAI API key not set.")
            return None
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
                temperature=0.7, max_tokens=4000
            )
            return resp.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
            return None

    elif provider_name == "Google":
        api_key = get_google_api_key()
        if not api_key:
            st.error("Google AI API key not set.")
            return None
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            safety_settings = {category: HarmBlockThreshold.BLOCK_NONE for category in HarmCategory}
            full_prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"
            resp = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=4000),
                safety_settings=safety_settings
            )
            return resp.text
        except Exception as e:
            st.error(f"Google AI API error: {e}")
            return None
    return None

def parse_ai_output(text: str) -> list:
    """
    Robustly parses the AI's markdown output into a list of structured clip data.
    This version checks for the existence of each match object before accessing its groups
    to prevent 'NoneType' errors.
    """
    clips = []
    # Split by the main header to isolate each clip's data
    sections = re.split(r'\*\*Short Title:\*\*', text)
    
    for i, section in enumerate(sections[1:], 1): # Start from 1 to skip the part before the first title
        try:
            # --- Safely extract each piece of information ---
            
            # Title
            title_match = re.search(r'^(.*?)(?:\n|\*\*)', section, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else f"Untitled Clip {i}"
            
            # Type
            type_match = re.search(r'\*\*Type:\*\*\s*(.*?)(?:\n|\*\*)', section)
            clip_type = type_match.group(1).strip() if type_match else "Unknown"

            # Rationale
            rationale_match = re.search(r'\*\*Rationale:\*\*(.*?)(?:\n\*\*|$)', section, re.DOTALL)
            rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

            # Script
            script_match = re.search(r'\*\*Script:\*\*(.*?)\*\*Rationale:\*\*', section, re.DOTALL)
            script = script_match.group(1).strip() if script_match else "Script not found."

            # Timestamps
            timestamp_text_match = re.search(r'\*\*Timestamps:\*\*(.*?)\*\*Script:\*\*', section, re.DOTALL)
            timestamps = []
            if timestamp_text_match:
                timestamp_text = timestamp_text_match.group(1)
                timestamp_matches = re.findall(r'START:\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*END:\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', timestamp_text)
                for start_str, end_str in timestamp_matches:
                    start_sec = parse_srt_timestamp(start_str)
                    timestamps.append({"start_str": start_str, "end_str": end_str, "start_sec": start_sec})

            # Only add the clip if we successfully found timestamps
            if timestamps:
                clips.append({
                    "title": title,
                    "type": clip_type,
                    "rationale": rationale,
                    "script": script,
                    "timestamps": timestamps
                })
            else:
                st.warning(f"Could not find valid timestamps for clip section {i}. Skipping.")

        except Exception as e:
            # This broad exception is a fallback, but the specific checks should prevent most crashes.
            st.warning(f"An unexpected error occurred while parsing clip section {i}: {e}")
            
    return clips

# ---
# 3. STREAMLIT APP
# ---

def main():
    st.set_page_config(page_title="AI Shorts Finder", layout="wide", page_icon="🎬")
    
    st.title("🎬 AI Shorts Finder")
    st.markdown("**Find viral YouTube Shorts from long-form content. The AI suggests clips, you review them instantly.**")

    # Initialize session state for the video player
    if 'video_url_to_play' not in st.session_state:
        st.session_state.video_url_to_play = None
    if 'video_start_time' not in st.session_state:
        st.session_state.video_start_time = 0

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("1. YouTube Video")
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="Paste your YouTube URL here...",
            help="The video you want to find clips from."
        )

        st.subheader("2. Transcript")
        uploaded_transcript = st.file_uploader(
            "Upload Transcript File", type=["srt", "txt", "docx"],
            help="A transcript with timestamps is required for analysis."
        )
        
        st.markdown("---")
        
        st.subheader("3. AI Settings")
        provider = st.selectbox("AI Provider:", ["Google", "OpenAI"])
        
        if provider == "OpenAI":
            model = st.selectbox("Model:", fetch_openai_models(get_openai_api_key()), index=0)
        else:
            model = st.selectbox("Model:", fetch_gemini_models(get_google_api_key()), index=0)

        clips_count = st.slider("Number of Shorts to Find:", 1, 10, 5)

    # Main Content Area
    if st.button("🚀 Find Potential Shorts", type="primary", use_container_width=True):
        if not video_url:
            st.error("❌ Please provide a YouTube URL.")
            return
        if not uploaded_transcript:
            st.error("❌ Please upload a transcript file.")
            return
        
        with st.spinner("📖 Reading transcript..."):
            transcript_content = read_transcript_file(uploaded_transcript)
            if not transcript_content: return
        st.success("✅ Transcript loaded.")

        with st.spinner(f"🧠 Analyzing transcript with {provider}..."):
            ai_response = analyze_transcript_with_llm(transcript_content, clips_count, model, provider)
            if not ai_response: return
        st.success("✅ AI analysis complete.")

        with st.spinner("📝 Parsing AI recommendations..."):
            clips_data = parse_ai_output(ai_response)
            if not clips_data:
                st.error("❌ Could not parse any valid clips from the AI response.")
                return
        st.success(f"✅ Found {len(clips_data)} potential clips!")
        
        # Store results in session state to persist them
        st.session_state.clips_data = clips_data
        st.session_state.video_url_to_play = video_url # Set the video to play
        st.session_state.video_start_time = 0 # Reset to start

    # --- Display Results ---
    if 'clips_data' in st.session_state and st.session_state.clips_data:
        st.markdown("---")
        st.header("🌟 Your Potential Shorts")

        # Video Player Area
        if st.session_state.video_url_to_play:
            st.video(st.session_state.video_url_to_play, start_time=st.session_state.video_start_time)

        for i, clip in enumerate(st.session_state.clips_data):
            with st.container():
                st.subheader(f"📱 {clip['title']}")
                
                col_info, col_script = st.columns(2)
                
                with col_info:
                    st.markdown(f"**Type:** `{clip['type']}`")
                    with st.expander("💡 Viral Rationale"):
                        st.info(clip['rationale'])
                    
                    # Timestamp Play Buttons
                    st.markdown("**Timestamps (click to play):**")
                    for j, ts in enumerate(clip['timestamps']):
                        button_label = f"▶️ Play Segment {j+1} ({ts['start_str']})"
                        if st.button(button_label, key=f"play_{i}_{j}"):
                            st.session_state.video_start_time = ts['start_sec']
                            # No rerun needed, Streamlit re-renders on widget interaction

                with col_script:
                    st.text_area("📜 Script", clip['script'], height=200, key=f"script_{i}")
                
                st.markdown("---")

if __name__ == "__main__":
    main()
