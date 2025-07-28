# app.py - AI Shorts Assistant
# Combines AI-driven editing plans with automated video clipping.

import os
import re
import tempfile
import streamlit as st
import shutil # Added for checking system dependencies

# This import is wrapped in a try/except block to provide a cleaner error
# if ffmpeg is missing, which is handled by a check in main().
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    # This will be caught by the check_ffmpeg() function in main(),
    # which provides a more user-friendly error message.
    pass

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

def check_ffmpeg():
    """Checks if ffmpeg is installed and accessible in the system's PATH."""
    return shutil.which("ffmpeg") is not None

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
    """Downloads a YouTube video to a specified path."""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(download_path, 'downloaded_video.mp4'),
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return os.path.join(download_path, 'downloaded_video.mp4')

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
    
    # OpenAI
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
        except OpenAIBadRequestError as e:
            st.error(f"OpenAI API Error: {e}. The selected model might not be available.")
            return None
        except Exception as e:
            st.error(f"An unexpected OpenAI API error occurred: {e}")
            return None

    # Google
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
            if not resp.parts:
                 st.error("The response was blocked by Google's safety filters. Try another model or provider.")
                 return None
            return resp.text
        except Exception as e:
            st.error(f"An unexpected Google AI API error occurred: {e}")
            return None
    return None

def parse_ai_output(text: str) -> list:
    """Parses the AI's markdown output into a list of structured clip data."""
    clips = []
    # Split the text by the main title header
    short_sections = re.split(r'Potential Short Title:', text)
    
    for i, section in enumerate(short_sections):
        if not section.strip():
            continue

        try:
            title_match = re.search(r'^(.*)', section, re.MULTILINE)
            duration_match = re.search(r'Estimated Duration:\s*(.*)', section, re.MULTILINE)
            type_match = re.search(r'Type:\s*(.*)', section, re.MULTILINE)
            rationale_match = re.search(r'Rationale for Virality:\s*([\s\S]*)', section, re.MULTILINE)
            
            title = title_match.group(1).strip() if title_match else f"Untitled Clip {i}"
            duration = duration_match.group(1).strip() if duration_match else "N/A"
            clip_type = type_match.group(1).strip() if type_match else "Direct Clip"
            rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

            # Extract timestamps from the markdown table
            table_rows = re.findall(r'\|\s*(.*?-->.*?)\s*\|', section, re.MULTILINE)
            timestamps = []
            for row in table_rows:
                start_str, end_str = [t.strip() for t in row.split('-->')]
                start_sec = time_to_seconds(start_str)
                end_sec = time_to_seconds(end_str)
                timestamps.append((start_sec, end_sec))

            if not timestamps:
                continue

            clips.append({
                "title": title,
                "duration": duration,
                "type": clip_type,
                "rationale": rationale,
                "timestamps": timestamps
            })
        except Exception as e:
            st.warning(f"Could not parse a clip section: {e}\nSection content: {section[:100]}...")
            
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
            subclips = []
            for start_time, end_time in clip_data["timestamps"]:
                # Ensure timestamps are within video duration
                if start_time < source_video.duration and end_time <= source_video.duration:
                    subclips.append(source_video.subclip(start_time, end_time))
                else:
                    st.warning(f"Timestamp ({start_time}-{end_time}) out of bounds for clip '{clip_data['title']}'. Skipping segment.")

            if not subclips:
                st.error(f"No valid segments found for clip '{clip_data['title']}'. Skipping.")
                continue

            # If it's a Franken-Clip with multiple segments, concatenate them
            if len(subclips) > 1:
                final_clip = concatenate_videoclips(subclips)
            else:
                final_clip = subclips[0]

            # Define output path and write file
            output_filename = f"clip_{i+1}.mp4"
            output_filepath = os.path.join(output_dir, output_filename)
            
            final_clip.write_videofile(output_filepath, codec="libx264", audio_codec="aac", temp_audiofile=f'temp-audio_{i}.m4a', remove_temp=True)
            
            generated_clips.append({
                "path": output_filepath,
                "title": clip_data['title'],
                "type": clip_data['type'],
                "rationale": clip_data['rationale']
            })
            st.success(f"✅ Successfully generated clip: {clip_data['title']}")

        except Exception as e:
            st.error(f"Failed to generate clip '{clip_data['title']}': {e}")
        finally:
            # Clean up memory
            if 'final_clip' in locals():
                final_clip.close()
            for sc in subclips:
                sc.close()

    source_video.close()
    return generated_clips

# ---
# 3. STREAMLIT APP UI & LOGIC
# ---

def main():
    st.set_page_config(page_title="AI Shorts Assistant", layout="wide")
    st.title("🤖 AI Shorts Assistant")
    st.markdown("Combines AI-driven editing plans with automated video clipping. Upload a video link and its transcript to get started.")

    # --- FFMPEG Dependency Check ---
    if not check_ffmpeg():
        st.error("CRITICAL ERROR: `ffmpeg` is not installed or not found in your system's PATH.")
        st.info("`ffmpeg` is a required system dependency for video processing with `moviepy`.")
        st.markdown("""
            ### How to Install `ffmpeg`
            
            **1. On Your Local Computer:**
            - **Windows:** Download the executables from [ffmpeg.org](https://ffmpeg.org/download.html), unzip them, and add the `bin` folder to your system's PATH environment variable.
            - **macOS (using Homebrew):** Open Terminal and run `brew install ffmpeg`.
            - **Linux (Debian/Ubuntu):** Open Terminal and run `sudo apt update && sudo apt install ffmpeg`.
            
            **2. On Streamlit Cloud:**
            - Create a file named `packages.txt` in the root of your GitHub repository.
            - Add one line to this file: `ffmpeg`
            - Commit and push this file. Streamlit Cloud will automatically install it for you.
            
            **After installing, you may need to restart your application or terminal.**
        """)
        return # Stop the app from executing further

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Video Source
        video_source_type = st.radio("Video Source", ["YouTube", "Google Drive"])
        video_url = st.text_input("Video URL", placeholder="Enter URL here...")
        
        # Transcript
        uploaded_transcript = st.file_uploader("Upload Transcript", type=["srt", "txt", "docx"])
        
        st.markdown("---")
        
        # AI Settings
        provider = st.selectbox("Choose AI Provider", ["Google", "OpenAI"])
        
        if provider == "OpenAI":
            available_models = fetch_openai_models(get_openai_api_key())
            model = st.selectbox("Choose model", available_models, index=0)
        else: # Google
            available_models = fetch_gemini_models(get_google_api_key())
            model = st.selectbox("Choose model", available_models, index=0)

        result_count = st.slider("Number of Shorts to Generate", 1, 10, 5)

    # --- Main App Logic ---
    if st.button("🚀 Generate Video Clips", type="primary"):
        if not video_url:
            st.error("Please provide a video URL.")
            return
        if not uploaded_transcript:
            st.error("Please upload a transcript file.")
            return

        # Create a temporary directory for all files for this run
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 1. Download Video
                with st.spinner(f"Downloading video from {video_source_type}... This may take a while."):
                    if video_source_type == "YouTube":
                        video_path = download_youtube_video(video_url, temp_dir)
                    else: # Google Drive
                        video_path = download_drive_file(video_url, temp_dir)
                st.success(f"✅ Video downloaded successfully to: {video_path}")

                # 2. Read Transcript
                transcript_text = read_transcript_file(uploaded_transcript)
                if not transcript_text:
                    st.error("Could not read transcript file.")
                    return

                # 3. Analyze with AI
                with st.spinner(f"Analyzing transcript with {provider}'s {model}..."):
                    ai_response = analyze_transcript_with_llm(transcript_text, result_count, model, provider)
                if not ai_response:
                    st.error("AI analysis failed. Please check the logs above.")
                    return
                st.success("✅ AI analysis complete.")
                with st.expander("View Raw AI Output"):
                    st.text_area("", ai_response, height=300)

                # 4. Parse AI Output
                clips_to_generate = parse_ai_output(ai_response)
                if not clips_to_generate:
                    st.error("Could not parse any valid clips from the AI response.")
                    return
                st.success(f"✅ Parsed {len(clips_to_generate)} clip plans from AI response.")

                # 5. Generate Clips
                final_clips = generate_clips(video_path, clips_to_generate, temp_dir)

                # 6. Display Results
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
                                        label="⬇️ Download Clip",
                                        data=file,
                                        file_name=f"{clip['title'].replace(' ', '_')}.mp4",
                                        mime="video/mp4"
                                    )
                            else:
                                st.error("Clip file not found.")
                        with col2:
                            st.markdown(f"**Type:** `{clip['type']}`")
                            st.markdown("**Rationale for Virality:**")
                            st.info(clip['rationale'])
                else:
                    st.warning("No clips were successfully generated.")

            except Exception as e:
                st.error(f"An unexpected error occurred during the process: {e}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
