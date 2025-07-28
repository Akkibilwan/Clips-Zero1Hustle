# app.py - AI Shorts Assistant
# Combines AI-driven editing plans with automated video clipping.

import os
import re
import tempfile
import streamlit as st
import shutil

# All necessary libraries are imported here.
# The environment setup via requirements.txt and packages.txt is critical for these to work.
from moviepy.editor import VideoFileClip, concatenate_videoclips
import yt_dlp
import gdown
from openai import OpenAI, BadRequestError as OpenAIBadRequestError
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as GoogleAPIErrors
import docx # Requires python-docx

# ---
# 1. SYSTEM PROMPT (from Code 2 - The Core AI Logic)
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
- Do not shorten, reword, paraphrase, or compress the speaker’s sentences.
- Keep all original punctuation, phrasing, and spelling.
- Only include full dialogue blocks — no cherry-picking fragments from within a block.

The output should allow a video editor to directly cut the clip using the given timestamps and script, without needing to interpret or reconstruct phrasing.

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

You must include an equal number of Franken-Clips and Direct Clips if possible.

---

📦 OUTPUT FORMAT (repeat for each Short):

Potential Short Title: [Catchy title with emoji if helpful]
Estimated Duration: [e.g., 42 seconds]
Type: [Direct Clip / Franken-Clip]

Transcript for Editor:
| Timestamp | Speaker | Dialogue |
|----------|---------|----------|
| 00:00:00,000 --> 00:00:04,000 | Speaker Name | Verbatim transcript line here |
| ... | ... | ... |

Rationale for Virality:
[Brief explanation — why this short works. Don’t skip this.]

---

🛑 CRITICAL REMINDERS:
- Do not summarize or "clean up" the speaker’s words.
- Do not shorten lines for brevity.
- Only use lines that appear exactly in the transcript.
- If a timestamp contains multiple lines, include the full lines verbatim.

Now read the full transcript carefully and return high-quality Direct and Franken-Clips that follow this format.
"""

# ---
# 2. HELPER FUNCTIONS
# ---

# --- API Key & Model Fetching ---
def get_openai_api_key() -> str:
    """Retrieve the OpenAI API key from Streamlit secrets."""
    return st.secrets.get("openai", {}).get("api_key", "")

def get_google_api_key() -> str:
    """Retrieve the Google AI API key from Streamlit secrets."""
    return st.secrets.get("google_ai", {}).get("api_key", "")

@st.cache_data(show_spinner=False)
def fetch_openai_models(api_key: str):
    """Fetch available GPT models, fallback to defaults on error."""
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
    """Fetch available Gemini models, fallback to defaults on error."""
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

# --- File & URL Handlers ---
def read_transcript_file(uploaded_file) -> str:
    """Reads the content of an uploaded .srt, .txt, or .doc file."""
    if uploaded_file.name.endswith('.docx'):
        try:
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error reading .docx file: {e}")
            return ""
    else:
        return uploaded_file.read().decode("utf-8")

def download_youtube_video(url: str, download_path: str) -> str:
    """Downloads a YouTube video to a specified path with robust options."""
    output_template = os.path.join(download_path, 'downloaded_video.%(ext)s')
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.5',
        },
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info_dict)
            if os.path.exists(downloaded_file):
                return downloaded_file
            else:
                for file in os.listdir(download_path):
                    if file.startswith("downloaded_video"):
                        return os.path.join(download_path, file)
                raise FileNotFoundError("Downloaded video file not found.")
    except yt_dlp.utils.DownloadError as e:
        if "HTTP Error 403" in str(e):
            raise Exception("YouTube download failed (403 Forbidden). This video may be private, region-locked, or YouTube is blocking server requests. Please use the direct file upload method instead.")
        else:
            raise e

def download_drive_file(drive_url: str, download_path: str) -> str:
    """Downloads a Google Drive file."""
    output_path = os.path.join(download_path, 'downloaded_video.mp4')
    gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
    return output_path

def time_to_seconds(time_str: str) -> float:
    """Converts HH:MM:SS,ms or HH:MM:SS.ms format to seconds."""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])

# --- AI & Parsing ---
def analyze_transcript_with_llm(transcript: str, count: int, model_name: str, provider_name: str):
    """Generates shorts ideas using the selected provider's API."""
    user_content = transcript + f"\n\nPlease generate {count} unique potential shorts in the specified format."
    
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
                temperature=0.7,
                max_tokens=4000
            )
            return resp.choices[0].message.content
        except Exception as e:
            st.error(f"An unexpected OpenAI API error occurred: {e}")
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
            st.error(f"An unexpected Google AI API error occurred: {e}")
            return None
    return None

def parse_ai_output(text: str) -> list:
    """Parses the AI's markdown output into a list of structured clip data."""
    clips = []
    short_sections = re.split(r'Potential Short Title:', text)
    
    for i, section in enumerate(short_sections):
        if not section.strip():
            continue

        try:
            title_match = re.search(r'^(.*)', section, re.MULTILINE)
            type_match = re.search(r'Type:\s*(.*)', section, re.MULTILINE)
            rationale_match = re.search(r'Rationale for Virality:\s*([\s\S]*)', section, re.MULTILINE)
            
            title = title_match.group(1).strip() if title_match else f"Untitled Clip {i}"
            clip_type = type_match.group(1).strip() if type_match else "Direct Clip"
            rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

            table_rows = re.findall(r'\|\s*(.*?-->.*?)\s*\|', section, re.MULTILINE)
            timestamps = []
            for row in table_rows:
                start_str, end_str = [t.strip() for t in row.split('-->')]
                timestamps.append((time_to_seconds(start_str), time_to_seconds(end_str)))

            if timestamps:
                clips.append({
                    "title": title, "type": clip_type, "rationale": rationale, "timestamps": timestamps
                })
        except Exception as e:
            st.warning(f"Could not parse a clip section: {e}")
            
    return clips

# --- Video Processing ---
def generate_clips(video_path: str, clips_data: list, output_dir: str) -> list:
    """Cuts and stitches video clips based on parsed AI data."""
    generated_clips = []
    if not os.path.exists(video_path):
        st.error(f"Source video not found at path: {video_path}")
        return []

    source_video = VideoFileClip(video_path)
    
    for i, clip_data in enumerate(clips_data):
        st.info(f"Processing Clip {i+1}/{len(clips_data)}: '{clip_data['title']}' ({clip_data['type']})")
        
        try:
            subclips = [source_video.subclip(start, end) for start, end in clip_data["timestamps"] if start < source_video.duration and end <= source_video.duration]

            if not subclips:
                st.error(f"No valid segments found for clip '{clip_data['title']}'. Skipping.")
                continue

            final_clip = concatenate_videoclips(subclips) if len(subclips) > 1 else subclips[0]
            output_filepath = os.path.join(output_dir, f"clip_{i+1}.mp4")
            
            final_clip.write_videofile(output_filepath, codec="libx264", audio_codec="aac", temp_audiofile=f'temp-audio_{i}.m4a', remove_temp=True)
            
            generated_clips.append({
                "path": output_filepath, "title": clip_data['title'], "type": clip_data['type'], "rationale": clip_data['rationale']
            })
            st.success(f"✅ Successfully generated clip: {clip_data['title']}")

        except Exception as e:
            st.error(f"Failed to generate clip '{clip_data['title']}': {e}")
        finally:
            if 'final_clip' in locals(): final_clip.close()
            for sc in subclips: sc.close()

    source_video.close()
    return generated_clips

# ---
# 3. STREAMLIT APP UI & LOGIC
# ---

def main():
    st.set_page_config(page_title="AI Shorts Assistant", layout="wide")
    st.title("🤖 AI Shorts Assistant")
    st.markdown("Combines AI-driven editing plans with automated video clipping. Upload a video file and its transcript to get started.")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # --- NEW: Prioritize File Upload ---
        st.subheader("1. Upload Video")
        st.info("🚀 This is the most reliable method.")
        uploaded_video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

        with st.expander("Alternative: Download from URL (May Fail)"):
            st.warning("YouTube downloads can be blocked. Use direct upload if possible.")
            video_source_type = st.radio("Video Source", ["YouTube", "Google Drive"])
            video_url = st.text_input("Video URL", placeholder="Enter URL here...")
        
        st.subheader("2. Upload Transcript")
        uploaded_transcript = st.file_uploader("Upload Transcript", type=["srt", "txt", "docx"])
        
        st.markdown("---")
        
        st.subheader("3. AI Settings")
        provider = st.selectbox("Choose AI Provider", ["Google", "OpenAI"])
        
        if provider == "OpenAI":
            model = st.selectbox("Choose model", fetch_openai_models(get_openai_api_key()), index=0)
        else: # Google
            model = st.selectbox("Choose model", fetch_gemini_models(get_google_api_key()), index=0)

        result_count = st.slider("Number of Shorts to Generate", 1, 10, 5)

    # --- Main App Logic ---
    if st.button("🚀 Generate Video Clips", type="primary"):
        # Validate inputs
        if not uploaded_video_file and not video_url:
            st.error("Please upload a video file or provide a URL.")
            return
        if not uploaded_transcript:
            st.error("Please upload a transcript file.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                video_path = ""
                # --- NEW: Logic to prioritize uploaded file ---
                if uploaded_video_file:
                    with st.spinner("Processing uploaded video file..."):
                        video_path = os.path.join(temp_dir, uploaded_video_file.name)
                        with open(video_path, "wb") as f:
                            f.write(uploaded_video_file.getbuffer())
                        st.success("✅ Uploaded video processed.")
                elif video_url:
                    with st.spinner(f"Downloading video from {video_source_type}... This may take a while."):
                        if video_source_type == "YouTube":
                            video_path = download_youtube_video(video_url, temp_dir)
                        else: # Google Drive
                            video_path = download_drive_file(video_url, temp_dir)
                    st.success(f"✅ Video downloaded successfully.")
                
                if not video_path or not os.path.exists(video_path):
                    st.error("Failed to obtain video file. Please check the source.")
                    return

                # Continue with the rest of the process
                transcript_text = read_transcript_file(uploaded_transcript)
                if not transcript_text: return

                with st.spinner(f"Analyzing transcript with {provider}'s {model}..."):
                    ai_response = analyze_transcript_with_llm(transcript_text, result_count, model, provider)
                if not ai_response: return
                st.success("✅ AI analysis complete.")
                with st.expander("View Raw AI Output"):
                    st.text_area("", ai_response, height=300)

                clips_to_generate = parse_ai_output(ai_response)
                if not clips_to_generate:
                    st.error("Could not parse any valid clips from the AI response.")
                    return
                st.success(f"✅ Parsed {len(clips_to_generate)} clip plans.")

                final_clips = generate_clips(video_path, clips_to_generate, temp_dir)

                if final_clips:
                    st.markdown("---")
                    st.header("🎬 Your Generated Clips")
                    for clip in final_clips:
                        st.subheader(f"{clip['title']}")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if os.path.exists(clip['path']):
                                st.video(clip['path'])
                                with open(clip['path'], "rb") as file:
                                    st.download_button(
                                        label="⬇️ Download Clip", data=file,
                                        file_name=f"{re.sub('[^A-Za-z0-9]+', '_', clip['title'])}.mp4",
                                        mime="video/mp4"
                                    )
                        with col2:
                            st.markdown(f"**Type:** `{clip['type']}`")
                            st.markdown("**Rationale for Virality:**")
                            st.info(clip['rationale'])
                else:
                    st.warning("No clips were successfully generated.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
