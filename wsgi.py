import yt_dlp
import json
import re
import calendar
import math
import logging
import os
import webvtt
from datetime import datetime, timedelta
from pathlib import Path
from stravalib import Client
from openai import OpenAI
from flask import Flask, request, Response

MAX_BAR_WIDTH = 16
PROMPT_FILES = {
    "timeline": "prompts/pose_timeline.txt",
    "intensity": "prompts/intensity.txt",
    "scores": "prompts/scores.txt",
    "summary": "prompts/summary.txt",
}

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)

# Create Flask app at module level for Gunicorn compatibility
app = Flask(__name__)


def bar_graph(data):
    """Generate ASCII bar graph from dictionary data."""
    full_block = "█"
    empty_block = "─"
    result = ""

    for measurement in sorted(data.keys(), reverse=True):
        bar_length = int(data[measurement] * MAX_BAR_WIDTH)
        bar = full_block * bar_length + empty_block * (MAX_BAR_WIDTH - bar_length)
        result += f"{bar} {measurement.title()}\n"

    return result


def interpolate_data(original):
    """Interpolate data points to fit MAX_BAR_WIDTH."""
    if not original:
        return [0] * MAX_BAR_WIDTH

    result = [0] * MAX_BAR_WIDTH
    step = (len(original) - 1) / (MAX_BAR_WIDTH - 1) if len(original) > 1 else 0

    for i in range(MAX_BAR_WIDTH):
        pos = i * step
        low_index = math.floor(pos)
        high_index = min(math.ceil(pos), len(original) - 1)
        frac = pos - low_index

        if low_index == high_index:
            result[i] = original[low_index]
        else:
            result[i] = (
                original[low_index]
                + (original[high_index] - original[low_index]) * frac
            )

    return result


def sparkline_graph(data):
    """Generate a sparkline graph from data."""
    BLOCK_CHARACTERS = "▁▂▃▄▅▆▇█"
    interpolated = interpolate_data(data)
    min_value = min(interpolated)
    max_value = max(interpolated)
    bin_width = 1.0 / (len(BLOCK_CHARACTERS) - 1)

    if max_value == min_value:
        return BLOCK_CHARACTERS[0] * MAX_BAR_WIDTH

    graph = []
    for value in interpolated:
        normalized = (value - min_value) / (max_value - min_value)
        graph.append(BLOCK_CHARACTERS[int(normalized / bin_width)])

    return "".join(graph)


def extract_youtube_id(url):
    """Extract YouTube ID from a URL."""
    pattern = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def format_datetime_readable():
    """Format current datetime in a readable format."""
    now = datetime.now()
    day_name = calendar.day_name[now.weekday()]
    hour = now.hour % 12 or 12
    period = "am" if now.hour < 12 else "pm"

    return f"{day_name} {hour}:{now.minute:02d}{period}"


def remove_markdown_code_blocks(text):
    """Remove markdown code blocks from text."""
    return re.sub(r"```[\w]*\n(.*?)```", r"\1", text, flags=re.DOTALL)


def fetch_youtube_subtitles(video_url):
    """Fetch subtitles from a YouTube video."""

    ydl_opts = {
        "forcefilename": True,
        "noprogress": True,
        "outtmpl": {"default": "%(id)s"},
        "paths": {"home": "subtitles/"},
        "quiet": True,
        "simulate": False,
        "skip_download": True,
        "subtitleslangs": ["en"],
        "writeautomaticsub": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        duration = info_dict["duration"]
        filename = ydl.prepare_filename(info_dict)

    vtt = webvtt.read(filename + ".en.vtt")
    vtt.save_as_srt(filename + ".en.srt")
    return filename + ".en.srt", duration


def send_llm_prompt(client, messages, prompt_content):
    """Send prompt to LLM and get response."""
    messages.append({"role": "user", "content": prompt_content})
    completion = client.chat.completions.create(
        model=app.config["MODEL"],
        messages=messages,
    )
    response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})

    return remove_markdown_code_blocks(response), messages


def run_ai_analysis(subtitle_file):
    """Run AI analysis on the video subtitles."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=app.config["OPENAI_API_KEY"],
    )

    # Initialize message history and results
    messages = []
    results = {}

    try:
        # Upload subtitle file
        client.files.create(
            file=Path(subtitle_file),
            purpose="assistants",
        )
        logging.info("Subtitles uploaded")

        # Intensity prompt
        prompt_content = Path(PROMPT_FILES["intensity"]).read_text()
        intensity_response, messages = send_llm_prompt(client, messages, prompt_content)

        # Add retry logic for intensity_data
        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                intensity_data = json.loads(intensity_response)
                results["intensity"] = intensity_data
                logging.info("Intensity analysis complete")
                break
            except json.JSONDecodeError:
                retries += 1
                if retries >= max_retries:
                    logging.error(
                        f"Failed to parse intensity response after {max_retries} attempts"
                    )
                    raise
                logging.warning(
                    f"Failed to parse intensity response, retrying ({retries}/{max_retries})"
                )
                intensity_response, messages = send_llm_prompt(
                    client, messages, prompt_content
                )

        # Scores prompt
        prompt_content = Path(PROMPT_FILES["scores"]).read_text()
        scores_response, messages = send_llm_prompt(client, messages, prompt_content)

        # Add retry logic for scores_data
        retries = 0
        while retries < max_retries:
            try:
                scores_data = json.loads(scores_response)
                results["scores"] = scores_data
                logging.info("Scores analysis complete")
                break
            except json.JSONDecodeError:
                retries += 1
                if retries >= max_retries:
                    logging.error(
                        f"Failed to parse scores response after {max_retries} attempts"
                    )
                    raise
                logging.warning(
                    f"Failed to parse scores response, retrying ({retries}/{max_retries})"
                )
                scores_response, messages = send_llm_prompt(
                    client, messages, prompt_content
                )

        # Summary prompt with formatted datetime
        summary_prompt = (
            Path(PROMPT_FILES["summary"]).read_text().format(format_datetime_readable())
        )
        title_response, messages = send_llm_prompt(client, messages, summary_prompt)

        # Add retry logic for title_data
        retries = 0
        while retries < max_retries:
            try:
                title_data = json.loads(title_response)
                break
            except json.JSONDecodeError:
                retries += 1
                if retries >= max_retries:
                    logging.error(
                        f"Failed to parse title response after {max_retries} attempts"
                    )
                    raise
                logging.warning(
                    f"Failed to parse title response, retrying ({retries}/{max_retries})"
                )
                title_response, messages = send_llm_prompt(
                    client, messages, summary_prompt
                )
        results["title"] = "[ai] " + title_data["title"]
        logging.info(f"Generated title: {results['title']}")
    except Exception as e:
        logging.error(f"Error running AI analyis: {e}")

    return results


def post_to_strava(video_url, title, duration, scores, intensity):
    """Post the activity to Strava."""
    description = f"""{bar_graph(scores["principles_summary"])}
{sparkline_graph(intensity)} Intensity

{bar_graph(scores["targets_summary"])}
youtube video: {extract_youtube_id(video_url)}
posted using a lil script
"""
    logging.info("Generated Strava description")

    strava_client = Client()
    try:
        token_response = strava_client.refresh_access_token(
            client_id=app.config["STRAVA_CLIENT_ID"],
            client_secret=app.config["STRAVA_CLIENT_SECRET"],
            refresh_token=app.config["STRAVA_REFRESH_TOKEN"],
        )
        strava_client.access_token = token_response["access_token"]

        activity_start_time = datetime.now() - timedelta(seconds=duration + 60)
        strava_client.create_activity(
            title,
            activity_start_time,
            duration,
            "Yoga",
            description=description,
        )
        logging.info(f"Activity posted to Strava: {title}")
        return True
    except Exception as e:
        logging.error(f"Error uploading to Strava: {str(e)}")
        return False


def process_youtube_url(video_url):
    """Process a YouTube URL and post to Strava."""
    try:
        logging.info(f"Processing video: {video_url}")

        subtitle_file, duration = fetch_youtube_subtitles(video_url)
        logging.info(f"Fetched subtitles: {subtitle_file}, duration: {duration}s")

        results = run_ai_analysis(subtitle_file)
        if results == {}:
            return False

        strava_posted = post_to_strava(
            video_url,
            results["title"],
            duration,
            results["scores"],
            results["intensity"],
        )
        if not strava_posted:
            return False

        logging.info("Process completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return False


# Load configuration at import time for Gunicorn
app.config.update(
    STRAVA_CLIENT_ID=int(os.environ["STRAVA_CLIENT_ID"]),
    STRAVA_CLIENT_SECRET=os.environ["STRAVA_CLIENT_SECRET"],
    STRAVA_REFRESH_TOKEN=os.environ["STRAVA_REFRESH_TOKEN"],
    OPENAI_API_KEY=os.environ["OPENAI_API_KEY"],
    YOGAVA_API_KEY=os.environ["YOGAVA_API_KEY"],
    MODEL=os.environ.get("MODEL", "google/gemini-2.0-flash-thinking-exp:free"),
)


@app.route("/submit", methods=["POST"])
def submit_video():
    """Flask endpoint to submit a YouTube video for processing."""
    logging.info(f"/submit request: {request.user_agent}")

    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header.split(" ")[-1] != app.config["YOGAVA_API_KEY"]:
        logging.info("Invalid yogava API key")
        return Response(response="Unauthorized", status=401)

    if not request.json or "video_url" not in request.json:
        logging.info("Incomplete message body")
        return Response(response="Missing video_url in request", status=400)

    video_url = request.json["video_url"]

    success = process_youtube_url(video_url)

    if success:
        return Response(response="OK", status=200)
    else:
        return Response(response="Internal server failure", status=500)


@app.route("/health", methods=["GET"])
def health_check():
    """Flask endpoint for service health checks."""
    logging.info(f"/health request: {request.user_agent}")
    return Response(response="Service is running", status=200)


def main():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    main()
