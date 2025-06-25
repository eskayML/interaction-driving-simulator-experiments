# Communication Analysis Tool for Human-AI Interaction Driving Simulator Experiments

This tool is a web application for TRIP-LAB that analyzes communication data in group simulation settings by processing video files to understand group dynamics through advanced data analytics and AI modeling.

![banner](assets/image.png)
## Overview

The project consists of two main components:
1. A video processing pipeline that extracts audio, performs transcription, and analyzes sentiment
2. A visualization interface built with Gradio that displays analysis results through interactive plots

## Features

- Audio extraction from video files
- Speech-to-text transcription using OpenAI's Whisper model
- Sentiment analysis using RoBERTa-based model
- Tone intensity analysis using audio amplitude
- Interactive visualization of:
  - Word count distribution
  - Sentiment trends over time
  - Tone intensity patterns


## Installation

1. Clone the repository
2. Create a Virtual Environment and then  Install  the  required packages:
   ```bash
   pip install -r requirements.txt
   ```

Since we are using a local model for our transcription (speech-to-text) , it requires the command-line tool ffmpeg to be installed on your system, which is available from most package managers:

```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

## Usage


Run the application:
```bash
python app.py
```

Then:
1. Access the web interface
2. Either:
   - Enter a folder path containing .mp4 files, or
   - Upload video files directly through the interface




## Architecture

### Data Processing Pipeline

1. **Video Processing (`video_processing.py`)**
   - Handles video file input
   - Manages the processing pipeline and the diarization

2. **Audio Processing (`audio_processing.py`)**
   - Extracts audio from videos (makes use of ffmpeg)
   - Analyzes tone intensity

3. **Transcription (`transcription.py`)**
   - Uses Whisper model for speech-to-text (makes use of ffmpeg also)
   - Segments audio into 5-second chunks

4. **Sentiment Analysis (`sentiment_analysis.py`)**
   - Uses CardiffNLP's RoBERTa model
   - Provides sentiment scores (-1 to +1)


## Testing

The system has been tested with:
- Single video files
- Multiple video batch processing
- Various video lengths and formats
- Different speech patterns and languages


## Notes

- Sentiment analysis uses a local pre-trained model, not external APIs
- Processing time varies based on video length and system capabilities
- CSV outputs are saved with unique names based on input video filenames
