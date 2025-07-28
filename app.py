# app.py - AI Shorts Assistant - Combined Finder & Generator
# This version intelligently switches between "Clip Finder" (for YouTube)
# and "Clip Generator" (for Google Drive).

import os
import re
import tempfile
import streamlit as st
import traceback

# All necessary libraries for both modes
from moviepy.editor import VideoFileClip, concatenate_videoclips
import gdown
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

def parse_srt_timestamp(timestamp_str: str) -> float:
    """Convert SRT timestamp format to total seconds."""
    timestamp_str = timestamp_str.strip().replace(',', '.')
    try:
        time_parts = timestamp_str.split(':')
        if len(time_parts) == 3:
            h, m, s_ms = time_parts
            return int(h) * 3600 + int(m) * 60 + float(s_ms)
        elif len(time_parts) == 2:
            m, s_ms = time_parts
            return int(m) * 60 + float(s_ms)
        return float(time_parts[0])
    except Exception:
        return 0.0

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
    """Robustly parses the AI's markdown output into a list of structured clip data."""
    clips = []
    sections = re.split(r'\*\*Short Title:\*\*', text)
    
    for i, section in enumerate(sections[1:], 1):
        try:
            title_match = re.search(r'^(.*?)(?:\n|\*\*)', section, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else f"Untitled Clip {i}"
            
            type_match = re.search(r'\*\*Type:\*\*\s*(.*?)(?:\n|\*\*)', section)
            clip_type = type_match.group(1).strip() if type_match else "Unknown"

            rationale_match = re.search(r'\*\*Rationale:\*\*(.*?)(?:\n\*\*|$)', section, re.DOTALL)
            rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

            script_match = re.search(r'\*\*Script:\*\*(.*?)(?=\*\*Rationale:\*\*)', section, re.DOTALL)
            script = script_match.group(1).strip() if script_match else "Script not found."

            timestamp_text_match = re.search(r'\*\*Timestamps:\*\*(.*?)(?=\*\*Script:\*\*)', section, re.DOTALL)
            timestamps = []
            if timestamp_text_match:
                timestamp_text = timestamp_text_match.group(1)
                timestamp_matches = re.findall(r'START:\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*END:\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', timestamp_text)
                for start_str, end_str in timestamp_matches:
                    start_sec = parse_srt_timestamp(start_str)
                    end_sec = parse_srt_timestamp(end_str)
                    timestamps.append({"start_str": start_str, "end_str": end_str, "start_sec": start_sec, "end_sec": end_sec})

            if timestamps:
                clips.append({
                    "title": title, "type": clip_type, "rationale": rationale,
                    "script": script, "timestamps": timestamps
                })
        except Exception as e:
            st.warning(f"Could not parse clip section {i}: {e}")
    return clips

def download_drive_file(drive_url: str, download_path: str) -> str:
    """Downloads a Google Drive file and verifies its integrity."""
    try:
        output_path = os.path.join(download_path, 'downloaded_video.mp4')
        gdown.download(drive_url, output_path, quiet=False, fuzzy=True)

        # --- Verification Step ---
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            raise Exception("Downloaded file is missing or empty.")

        # Try to read the duration immediately to verify the 'moov atom'
        try:
            with VideoFileClip(output_path) as clip:
                duration = clip.duration
            if duration is None or duration <= 0:
                raise Exception("Video file is corrupted (duration is zero or None).")
            st.info(f"Verified downloaded file. Duration: {duration:.2f} seconds.")
            return output_path
        except Exception as e:
            raise Exception(f"Downloaded file appears to be corrupted and cannot be read by MoviePy. Error: {e}. This often happens with incomplete downloads. Please check the Google Drive sharing settings and try again.")

    except Exception as e:
        raise Exception(f"Google Drive download failed: {e}. Ensure the link is public and correct.")

def generate_clips(video_path: str, clips_data: list, output_dir: str) -> list:
    """Cuts and stitches video clips based on parsed AI data."""
    generated_clips = []
    source_video = VideoFileClip(video_path)
    video_duration = source_video.duration
    
    for i, clip_data in enumerate(clips_data):
        st.info(f"Processing Clip {i+1}/{len(clips_data)}: '{clip_data['title']}'")
        try:
            subclips = []
            for ts in clip_data["timestamps"]:
                start_time, end_time = ts['start_sec'], ts['end_sec']
                if start_time < video_duration and end_time <= video_duration:
                    subclips.append(source_video.subclip(start_time, end_time))
                else:
                    st.warning(f"Segment {ts['start_str']} -> {ts['end_str']} is out of video bounds. Skipping.")
            
            if not subclips:
                st.error(f"No valid segments for clip '{clip_data['title']}'. Skipping.")
                continue

            final_clip = concatenate_videoclips(subclips) if len(subclips) > 1 else subclips[0]
            
            safe_title = re.sub(r'[^\w\s-]', '', clip_data['title']).strip().replace(' ', '_')
            output_filepath = os.path.join(output_dir, f"clip_{i+1}_{safe_title[:20]}.mp4")
            
            final_clip.write_videofile(output_filepath, codec="libx264", audio_codec="aac", temp_audiofile=f'temp-audio_{i}.m4a', remove_temp=True, logger=None)
            
            generated_clips.append({
                "path": output_filepath, "title": clip_data['title'], "type": clip_data['type'], "rationale": clip_data['rationale']
            })
            st.success(f"✅ Generated clip: {clip_data['title']}")

        except Exception as e:
            st.error(f"Failed to generate clip '{clip_data['title']}': {e}")
        finally:
            if 'final_clip' in locals(): final_clip.close()
            if 'subclips' in locals():
                for sc in subclips: sc.close()

    source_video.close()
    return generated_clips

# ---
# 3. STREAMLIT APP
# ---

def main():
    st.set_page_config(page_title="AI Shorts Assistant", layout="wide", page_icon="🎬")
    
    st.title("🎬 AI Shorts Assistant")
    st.markdown("**Intelligently find or generate viral shorts from your long-form content.**")

    # Initialize session state
    if 'video_url_to_play' not in st.session_state:
        st.session_state.video_url_to_play = None
    if 'video_start_time' not in st.session_state:
        st.session_state.video_start_time = 0
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("1. Video Source")
        video_source = st.radio("Choose your video source:", ["YouTube", "Google Drive"])

        video_url = st.text_input(f"{video_source} URL", placeholder=f"Paste your {video_source} URL here...")
        
        st.subheader("2. Transcript")
        uploaded_transcript = st.file_uploader("Upload Transcript File", type=["srt", "txt", "docx"])
        
        st.markdown("---")
        
        st.subheader("3. AI Settings")
        provider = st.selectbox("AI Provider:", ["Google", "OpenAI"])
        
        if provider == "OpenAI":
            model = st.selectbox("Model:", fetch_openai_models(get_openai_api_key()), index=0)
        else:
            model = st.selectbox("Model:", fetch_gemini_models(get_google_api_key()), index=0)

        clips_count = st.slider("Number of Clips to Find/Generate:", 1, 10, 5)
    
    # Main action button
    button_text = "🚀 Find Potential Shorts" if video_source == "YouTube" else "🚀 Generate Video Clips"
    if st.button(button_text, type="primary", use_container_width=True):
        if not video_url or not uploaded_transcript:
            st.error("❌ Please provide both a video URL and a transcript file.")
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
        st.success(f"✅ Parsed {len(clips_data)} recommendations.")
        
        # --- Conditional Workflow ---
        if video_source == "YouTube":
            st.session_state.results = {"type": "finder", "data": clips_data}
            st.session_state.video_url_to_play = video_url
            st.session_state.video_start_time = 0
            st.rerun() # Rerun to display results immediately
        
        elif video_source == "Google Drive":
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    with st.spinner("⬇️ Downloading video from Google Drive..."):
                        video_path = download_drive_file(video_url, temp_dir)
                    st.success("✅ Video downloaded and verified.")

                    with st.spinner("🎬 Generating and stitching video clips... This may take a while."):
                        generated_clips = generate_clips(video_path, clips_data, temp_dir)
                    
                    # To display clips, we need to move them out of the temp dir
                    final_clip_paths = []
                    if generated_clips:
                        persistent_dir = "generated_clips"
                        if not os.path.exists(persistent_dir):
                            os.makedirs(persistent_dir)
                        # Clear old clips from persistent dir
                        for f in os.listdir(persistent_dir):
                            os.remove(os.path.join(persistent_dir, f))
                        
                        for clip in generated_clips:
                            new_path = os.path.join(persistent_dir, os.path.basename(clip['path']))
                            os.rename(clip['path'], new_path)
                            clip['path'] = new_path
                            final_clip_paths.append(clip)
                    
                    st.session_state.results = {"type": "generator", "data": final_clip_paths}
                    st.rerun() # Rerun to display results immediately

                except Exception as e:
                    st.error(f"An error occurred during the clip generation process: {e}")
                    st.code(traceback.format_exc())

    # --- Display Results ---
    if st.session_state.results:
        results = st.session_state.results
        st.markdown("---")
        
        # Display for YouTube "Clip Finder"
        if results["type"] == "finder":
            st.header("🌟 Your Potential Shorts (Finder Mode)")
            if st.session_state.video_url_to_play:
                st.video(st.session_state.video_url_to_play, start_time=st.session_state.video_start_time)
            
            for i, clip in enumerate(results["data"]):
                st.subheader(f"📱 {clip['title']}")
                col_info, col_script = st.columns(2)
                with col_info:
                    st.markdown(f"**Type:** `{clip['type']}`")
                    st.markdown("**Timestamps (click to play):**")
                    for j, ts in enumerate(clip['timestamps']):
                        if st.button(f"▶️ Play Segment {j+1} ({ts['start_str']})", key=f"play_{i}_{j}"):
                            st.session_state.video_start_time = int(ts['start_sec'])
                with col_script:
                    st.text_area("📜 Script", clip['script'], height=150, key=f"script_{i}")
                st.markdown("---")
        
        # Display for Google Drive "Clip Generator"
        elif results["type"] == "generator":
            st.header("✅ Your Generated Clips (Generator Mode)")
            if not results["data"]:
                st.warning("No clips were successfully generated.")
            for clip in results["data"]:
                st.subheader(f"🎬 {clip['title']}")
                col_video, col_info = st.columns(2)
                with col_video:
                    if os.path.exists(clip['path']):
                        st.video(clip['path'])
                        with open(clip['path'], "rb") as file:
                            st.download_button("⬇️ Download Clip", file, file_name=os.path.basename(clip['path']))
                    else:
                        st.error("Clip file not found.")
                with col_info:
                     st.markdown(f"**Type:** `{clip['type']}`")
                     st.info(clip['rationale'])
                st.markdown("---")

if __name__ == "__main__":
    main()
