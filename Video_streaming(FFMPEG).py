import os
import json
import logging
from google.cloud import storage
import ffmpeg
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.api_core.exceptions import NotFound
from google.api_core.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import TemporaryDirectory
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)

# Retry configuration
retry = Retry(initial=1.0, maximum=10.0, multiplier=2.0)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


def check_video_integrity(file_path):
    try:
        ffmpeg.probe(file_path)
        return True
    except ffmpeg.Error:
        return False


def process_video(file_name, bucket_name):
    """Processes a video stored in a GCS bucket."""

    # Initialize Vertex AI
    project_id = os.getenv("PROJECT_ID", "sap-demo-416908")
    location = os.getenv("LOCATION", "us-central1")
    vertexai.init(project=project_id, location=location)
    vision_model = GenerativeModel("gemini-1.5-flash-002")

    storage_client = storage.Client()
    input_bucket = storage_client.bucket(bucket_name)
    output_bucket_name = os.getenv("OUTPUT_BUCKET", bucket_name)
    output_bucket = storage_client.bucket(output_bucket_name)

    with TemporaryDirectory() as tmp_dir:
        local_video_path = os.path.join(tmp_dir, file_name)
        blob = input_bucket.blob(file_name)

        # Check if the file exists in the bucket
        try:
            blob.reload(retry=retry)
        except NotFound:
            logging.error(f"File {file_name} does not exist in bucket {bucket_name}.")
            return

        try:
            logging.info(f"Downloading {file_name} from bucket {bucket_name}.")
            blob.download_to_filename(local_video_path, retry=retry)
            logging.info(f"Successfully downloaded {file_name}.")

            # Check video integrity
            if not check_video_integrity(local_video_path):
                logging.error(f"Video file {file_name} is corrupted.")
                return

            # Get video duration
            probe = ffmpeg.probe(local_video_path)
            duration = float(probe["format"]["duration"])
            interval = 10 * 60  # 10 minutes
            part = 1

            def process_segment(start_time, end_time, part):
                try:
                    segment_path = os.path.join(tmp_dir, f"{file_name}_part{part}.mp4")
                    logging.info(f"Processing segment {part}: {start_time}-{end_time} seconds.")

                    # Use ffmpeg-python to cut the video
                    (
                        ffmpeg.input(local_video_path, ss=start_time, to=end_time)
                        .output(segment_path, codec="libx264", audio_codec="aac")
                        .run(overwrite_output=True)
                    )

                    # Verify the segment file exists before uploading
                    if not os.path.exists(segment_path):
                        logging.error(f"Segment file {segment_path} does not exist.")
                        return

                    # Upload the segment to GCS
                    segment_blob = input_bucket.blob(f"temp/{file_name}_part{part}.mp4")
                    segment_blob.upload_from_filename(segment_path, retry=retry)
                    logging.info(f"Uploaded segment {part} to {segment_blob.name}.")

                    # Use Vertex AI's generative model for analysis
                    video_uri = f"gs://{bucket_name}/temp/{file_name}_part{part}.mp4"
                    response = vision_model.generate_content(
                        [
                            Part.from_uri(video_uri, mime_type="video/mp4"),
                            "Explain timestamp by timestamp all the events that take place and check if shoplifting is done or not?",
                        ]
                    )

                    # Parse response
                    analysis = response.text if response else "No analysis available."
                    video_segment_info = {
                        "file_name": file_name,
                        "part": part,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "analysis": analysis,
                    }

                    # Upload JSON result
                    json_file_name = f"{os.path.splitext(file_name)[0]}_part{part}.json"
                    json_blob = output_bucket.blob(f"processed/{json_file_name}")
                    json_blob.upload_from_string(json.dumps(video_segment_info), content_type="application/json")
                    logging.info(f"Uploaded analysis {json_file_name}.")

                except Exception as e:
                    logging.error(f"Error processing segment {part}: {str(e)}")
                    raise

            futures = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                for start_time in range(0, int(duration), interval):
                    end_time = min(start_time + interval, duration)
                    futures.append(executor.submit(process_segment, start_time, end_time, part))
                    part += 1

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Error in thread: {str(e)}")

        except Exception as e:
            logging.error(f"Error processing video {file_name}: {str(e)}")
            raise
        finally:
            logging.info(f"Cleaned up resources for {file_name}.")


if __name__ == "__main__":
    # Configuration from environment variables
    FILE_NAME = os.getenv("FILE_NAME", "ASMR Satisfying Video.mp4")
    BUCKET_NAME = os.getenv("INPUT_BUCKET", "video-streaming-test-json")

    process_video(FILE_NAME, BUCKET_NAME)
