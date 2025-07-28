# app.py - AI Shorts Assistant - Complete Updated Version
# Combines AI-driven editing plans with automated video clipping.

import os
import re
import tempfile
import streamlit as st
from datetime import datetime
import traceback

# All necessary libraries
from moviepy.editor import VideoFileClip, concatenate_videoclips
import yt_dlp
import gdown
from openai import OpenAI, BadRequestError as OpenAIBadRequestError
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as GoogleAPIErrors
import docx  # Requires python-docx

# ---
# 1. SYSTEM PROMPT - Enhanced for better timestamp extraction
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
    """Retrieve the OpenAI API key from Streamlit secrets."""
    return st.secrets.get("openai", {}).get("api_key", "")

def get_google_api_key() -> str:
    """Retrieve the Google AI API key from Streamlit secrets."""
    return st.secrets.get("google_ai", {}).get("api_key", "")

@st.cache_data(show_spinner=False)
def fetch_openai_models(api_key: str):
    """Fetch available GPT models, fallback to defaults on error."""
    default_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    if not api_key: 
        return default_models
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
    if not api_key: 
        return default_models
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name]
        model_ids = [m.split('/')[-1] for m in models]
        return sorted(list(set(model_ids + default_models))) if model_ids else default_models
    except Exception:
        st.error("Could not fetch Gemini models. Using default list.")
        return default_models

def read_transcript_file(uploaded_file) -> str:
    """Reads the content of an uploaded .srt, .txt, or .docx file."""
    try:
        if uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def extract_drive_id(url: str) -> str:
    """Extract Google Drive file ID from various URL formats."""
    patterns = [
        r'/file/d/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)',
        r'/d/([a-zA-Z0-9-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url  # Return as-is if no pattern matches

def download_youtube_video(url: str, download_path: str) -> str:
    """Downloads a YouTube video with enhanced anti-blocking measures."""
    output_template = os.path.join(download_path, 'downloaded_video.%(ext)s')
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        },
        'socket_timeout': 30,
        'retries': 3,
        'fragment_retries': 3,
        'skip_unavailable_fragments': True,
        'extractor_args': {
            'youtube': {
                'skip': ['hls', 'dash'],
                'player_client': ['android', 'web'],
            }
        },
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to test accessibility
            info_dict = ydl.extract_info(url, download=False)
            if not info_dict:
                raise Exception("Could not extract video information")
            
            # Download the video
            info_dict = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info_dict)
            
            if os.path.exists(downloaded_file):
                return downloaded_file
            else:
                # Fallback search
                for file in os.listdir(download_path):
                    if file.startswith("downloaded_video"):
                        return os.path.join(download_path, file)
                raise FileNotFoundError("Downloaded video file not found.")
                
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e).lower()
        if "403" in error_msg or "forbidden" in error_msg:
            raise Exception("YouTube download failed (403 Forbidden). This video may be private, region-locked, or YouTube is blocking server requests. Try using Google Drive instead.")
        elif "unavailable" in error_msg:
            raise Exception("This YouTube video is unavailable. It may be private, deleted, or region-locked.")
        elif "copyright" in error_msg:
            raise Exception("This video cannot be downloaded due to copyright restrictions.")
        else:
            raise Exception(f"YouTube download failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error during YouTube download: {str(e)}")

def download_drive_file(drive_url: str, download_path: str) -> str:
    """Downloads a Google Drive file with improved error handling."""
    try:
        file_id = extract_drive_id(drive_url)
        output_path = os.path.join(download_path, 'downloaded_video.mp4')
        
        # Create the direct download URL
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        gdown.download(download_url, output_path, quiet=False, fuzzy=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            raise Exception("Downloaded file is empty or doesn't exist")
            
    except Exception as e:
        raise Exception(f"Google Drive download failed: {str(e)}. Make sure the file is publicly accessible.")

def parse_srt_timestamp(timestamp_str: str) -> float:
    """Convert SRT timestamp format to seconds."""
    # Handle both comma and dot as decimal separator
    timestamp_str = timestamp_str.strip()
    
    # Split by arrow if it's a range
    if '-->' in timestamp_str:
        timestamp_str = timestamp_str.split('-->')[0].strip()
    
    # Replace comma with dot for milliseconds
    timestamp_str = timestamp_str.replace(',', '.')
    
    # Parse HH:MM:SS.mmm format
    try:
        time_parts = timestamp_str.split(':')
        if len(time_parts) == 3:
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds_ms = float(time_parts[2])
            return hours * 3600 + minutes * 60 + seconds_ms
        elif len(time_parts) == 2:
            minutes = int(time_parts[0])
            seconds_ms = float(time_parts[1])
            return minutes * 60 + seconds_ms
        else:
            return float(time_parts[0])
    except Exception as e:
        st.warning(f"Could not parse timestamp: {timestamp_str}")
        return 0.0

def extract_timestamps_from_srt(srt_content: str) -> list:
    """Extract all timestamps from SRT content for validation."""
    timestamps = []
    # Pattern to match SRT timestamp lines
    timestamp_pattern = r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})'
    
    matches = re.findall(timestamp_pattern, srt_content)
    for start_time, end_time in matches:
        start_seconds = parse_srt_timestamp(start_time)
        end_seconds = parse_srt_timestamp(end_time)
        timestamps.append((start_seconds, end_seconds, start_time, end_time))
    
    return timestamps

def analyze_transcript_with_llm(transcript: str, count: int, model_name: str, provider_name: str):
    """Generates shorts ideas using the selected AI provider."""
    user_content = f"{transcript}\n\nPlease generate {count} unique potential shorts following the exact format specified. Focus on the most viral-worthy moments with precise timestamps."
    
    if provider_name == "OpenAI":
        api_key = get_openai_api_key()
        if not api_key:
            st.error("OpenAI API key not set in Streamlit secrets.")
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
            st.error(f"OpenAI API error: {e}")
            return None

    elif provider_name == "Google":
        api_key = get_google_api_key()
        if not api_key:
            st.error("Google AI API key not set in Streamlit secrets.")
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
            st.error(f"Google AI API error: {e}")
            return None
    return None

def parse_ai_output(text: str) -> list:
    """Parse the AI output to extract clip information."""
    clips = []
    
    # Split by Short Title headers
    sections = re.split(r'\*\*Short Title:\*\*', text)
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        try:
            # Extract title
            title_match = re.search(r'^(.*?)(?=\*\*)', section, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else f"Clip {i}"
            
            # Extract duration
            duration_match = re.search(r'\*\*Estimated Duration:\*\*\s*(.*?)(?=\*\*)', section)
            duration = duration_match.group(1).strip() if duration_match else "Unknown"
            
            # Extract type
            type_match = re.search(r'\*\*Type:\*\*\s*(.*?)(?=\*\*)', section)
            clip_type = type_match.group(1).strip() if type_match else "Direct Clip"
            
            # Extract timestamps
            timestamp_section = re.search(r'\*\*Timestamps:\*\*(.*?)\*\*Script:\*\*', section, re.DOTALL)
            timestamps = []
            
            if timestamp_section:
                timestamp_text = timestamp_section.group(1)
                # Look for START: ... --> END: ... patterns
                timestamp_matches = re.findall(r'START:\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*END:\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', timestamp_text)
                
                for start_time, end_time in timestamp_matches:
                    start_seconds = parse_srt_timestamp(start_time)
                    end_seconds = parse_srt_timestamp(end_time)
                    timestamps.append((start_seconds, end_seconds))
            
            # Extract script
            script_match = re.search(r'\*\*Script:\*\*(.*?)\*\*Rationale:\*\*', section, re.DOTALL)
            script = script_match.group(1).strip() if script_match else ""
            
            # Extract rationale
            rationale_match = re.search(r'\*\*Rationale:\*\*(.*?)(?=\*\*|$)', section, re.DOTALL)
            rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."
            
            if timestamps:  # Only add clips with valid timestamps
                clips.append({
                    "title": title,
                    "duration": duration,
                    "type": clip_type,
                    "timestamps": timestamps,
                    "script": script,
                    "rationale": rationale
                })
                
        except Exception as e:
            st.warning(f"Could not parse clip section {i}: {e}")
    
    return clips

def generate_clips(video_path: str, clips_data: list, output_dir: str) -> list:
    """Generate video clips based on parsed AI data."""
    generated_clips = []
    
    if not os.path.exists(video_path):
        st.error(f"Source video not found: {video_path}")
        return []

    try:
        source_video = VideoFileClip(video_path)
        video_duration = source_video.duration
        st.info(f"Source video duration: {video_duration:.2f} seconds")
        
        for i, clip_data in enumerate(clips_data):
            st.info(f"Processing Clip {i+1}/{len(clips_data)}: '{clip_data['title']}' ({clip_data['type']})")
            
            try:
                subclips = []
                total_duration = 0
                
                for start_time, end_time in clip_data["timestamps"]:
                    # Validate timestamps
                    if start_time >= video_duration or end_time > video_duration:
                        st.warning(f"Timestamp ({start_time:.2f}-{end_time:.2f}) exceeds video duration ({video_duration:.2f}s). Skipping segment.")
                        continue
                    
                    if start_time >= end_time:
                        st.warning(f"Invalid timestamp range ({start_time:.2f}-{end_time:.2f}). Skipping segment.")
                        continue
                    
                    segment_duration = end_time - start_time
                    total_duration += segment_duration
                    
                    subclip = source_video.subclip(start_time, end_time)
                    subclips.append(subclip)
                    st.success(f"✓ Extracted segment: {start_time:.2f}s - {end_time:.2f}s ({segment_duration:.2f}s)")

                if not subclips:
                    st.error(f"No valid segments found for clip '{clip_data['title']}'. Skipping.")
                    continue

                # Concatenate clips if multiple segments
                if len(subclips) > 1:
                    final_clip = concatenate_videoclips(subclips)
                    st.info(f"Concatenated {len(subclips)} segments into Franken-Clip")
                else:
                    final_clip = subclips[0]

                # Define output path
                safe_title = re.sub(r'[^\w\s-]', '', clip_data['title']).strip()
                safe_title = re.sub(r'[-\s]+', '-', safe_title)
                output_filename = f"clip_{i+1}_{safe_title[:30]}.mp4"
                output_filepath = os.path.join(output_dir, output_filename)
                
                # Write video file
                st.info(f"Rendering clip... (Total duration: {total_duration:.2f}s)")
                final_clip.write_videofile(
                    output_filepath, 
                    codec="libx264", 
                    audio_codec="aac",
                    temp_audiofile=f'temp-audio_{i}.m4a',
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
                
                generated_clips.append({
                    "path": output_filepath,
                    "title": clip_data['title'],
                    "type": clip_data['type'],
                    "script": clip_data['script'],
                    "rationale": clip_data['rationale'],
                    "duration": f"{total_duration:.2f}s"
                })
                
                st.success(f"✅ Successfully generated: {clip_data['title']}")

            except Exception as e:
                st.error(f"Failed to generate clip '{clip_data['title']}': {e}")
            finally:
                # Clean up memory
                if 'final_clip' in locals():
                    final_clip.close()
                for sc in subclips:
                    sc.close()

        source_video.close()
        
    except Exception as e:
        st.error(f"Error loading source video: {e}")
    
    return generated_clips

# ---
# 3. STREAMLIT APP
# ---

def main():
    st.set_page_config(page_title="AI Shorts Assistant", layout="wide", page_icon="🎬")
    
    st.title("🎬 AI Shorts Assistant")
    st.markdown("**Generate viral YouTube Shorts from long-form content using AI analysis and automated video editing.**")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Video Source
        st.subheader("📹 Video Source")
        video_source = st.radio("Choose video source:", ["YouTube URL", "Google Drive Link"])
        video_url = st.text_input(
            "Video URL", 
            placeholder="Paste your YouTube or Google Drive URL here...",
            help="For Google Drive: Make sure the file is publicly accessible"
        )
        
        # Transcript Upload
        st.subheader("📄 Transcript")
        uploaded_transcript = st.file_uploader(
            "Upload Transcript File", 
            type=["srt", "txt", "docx"],
            help="Upload your video transcript with timestamps"
        )
        
        st.markdown("---")
        
        # AI Configuration
        st.subheader("🤖 AI Settings")
        provider = st.selectbox("AI Provider:", ["Google", "OpenAI"])
        
        if provider == "OpenAI":
            available_models = fetch_openai_models(get_openai_api_key())
            model = st.selectbox("Model:", available_models, index=0)
        else:
            available_models = fetch_gemini_models(get_google_api_key())
            model = st.selectbox("Model:", available_models, index=0)

        clips_count = st.slider("Number of Shorts to Generate:", 1, 10, 5)
        
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("- Ensure your transcript has proper timestamps")
        st.markdown("- Use public/accessible video links")
        st.markdown("- SRT format works best for timestamps")

    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate Video Clips", type="primary", use_container_width=True):
            # Validation
            if not video_url:
                st.error("❌ Please provide a video URL")
                return
            
            if not uploaded_transcript:
                st.error("❌ Please upload a transcript file")
                return
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Step 1: Read Transcript
                    with st.spinner("📖 Reading transcript..."):
                        transcript_content = read_transcript_file(uploaded_transcript)
                        if not transcript_content:
                            st.error("❌ Could not read transcript file")
                            return
                    
                    st.success(f"✅ Transcript loaded ({len(transcript_content)} characters)")
                    
                    # Step 2: Extract and validate timestamps
                    with st.spinner("🕐 Analyzing timestamps..."):
                        srt_timestamps = extract_timestamps_from_srt(transcript_content)
                        st.info(f"Found {len(srt_timestamps)} timestamp ranges in transcript")
                    
                    # Step 3: Download Video
                    with st.spinner(f"⬇️ Downloading video from {video_source}..."):
                        if video_source == "YouTube URL":
                            video_path = download_youtube_video(video_url, temp_dir)
                        else:
                            video_path = download_drive_file(video_url, temp_dir)
                    
                    st.success("✅ Video downloaded successfully")
                    
                    # Step 4: AI Analysis
                    with st.spinner(f"🧠 Analyzing transcript with {provider} {model}..."):
                        ai_response = analyze_transcript_with_llm(
                            transcript_content, clips_count, model, provider
                        )
                        if not ai_response:
                            st.error("❌ AI analysis failed")
                            return
                    
                    st.success("✅ AI analysis complete")
                    
                    # Show AI response in expander
                    with st.expander("🔍 View AI Analysis"):
                        st.text_area("Raw AI Output:", ai_response, height=300)
                    
                    # Step 5: Parse AI Output
                    with st.spinner("📝 Parsing AI recommendations..."):
                        clips_data = parse_ai_output(ai_response)
                        if not clips_data:
                            st.error("❌ Could not parse any valid clips from AI response")
                            return
                    
                    st.success(f"✅ Parsed {len(clips_data)} clip recommendations")
                    
                    # Step 6: Generate Clips
                    st.markdown("---")
                    st.header("🎬 Generating Video Clips")
                    
                    generated_clips = generate_clips(video_path, clips_data, temp_dir)
                    
                    # Step 7: Display Results
                    if generated_clips:
                        st.markdown("---")
                        st.header("🌟 Your Generated Shorts")
                        st.success(f"Successfully generated {len(generated_clips)} clips!")
                        
                        for i, clip in enumerate(generated_clips):
                            with st.container():
                                st.subheader(f"📱 {clip['title']}")
                                
                                col_video, col_info = st.columns([1, 1])
                                
                                with col_video:
                                    if os.path.exists(clip['path']):
                                        st.video(clip['path'])
                                        
                                        # Download button
                                        with open(clip['path'], "rb") as file:
                                            st.download_button(
                                                label="⬇️ Download Clip",
                                                data=file,
                                                file_name=f"{clip['title'].replace(' ', '_')}.mp4",
                                                mime="video/mp4",
                                                key=f"download_{i}"
                                            )
                                    else:
                                        st.error("❌ Clip file not found")
                                
                                with col_info:
                                    st.markdown(f"**Type:** `{clip['type']}`")
                                    st.markdown(f"**Duration:** `{clip['duration']}`")
                                    
                                    with st.expander("📜 Script"):
                                        st.text_area("", clip['script'], height=100, key=f"script_{i}")
                                    
                                    with st.expander("💡 Viral Rationale"):
                                        st.info(clip['rationale'])
                                
                                st.markdown("---")
                    else:
                        st.warning("⚠️ No clips were successfully generated")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    with st.expander("🐛 Error Details"):
                        st.code(traceback.format_exc())
    
    with col2:
        st.markdown("### 📋 How it works:")
        st.markdown("""
        1. **Upload your content** - Provide video URL and transcript
        2. **AI analyzes** - Identifies viral-worthy moments
        3. **Auto-generates** - Creates perfectly timed Short clips
        4. **Download & share** - Get ready-to-upload content
        """)
        
        st.markdown("### 🎯 Supported formats:")
        st.markdown("""
        **Video Sources:**
        - YouTube URLs
        - Google Drive links
        
        **Transcript Formats:**
        - .srt (SubRip)
        - .txt (plain text)
        - .docx (Word document)
        """)

if __name__ == "__main__":
    main()
