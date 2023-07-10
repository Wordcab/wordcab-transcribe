import requests
import json

def run_api_youtube(
    url:str, 
    source_lang:str = "en", 
    timestamps:str = "s", 
    word_timestamps:bool = False, 
    alignment:bool = False, 
    diarization:bool = False, 
):
    """
        Run API call for Youtube videos.

        Args:
            url (str): URL source of the Youtube video.
            source_lang = language of the URL source (defaulted to English)
            timestamps = time unit of the timestamps (defaulted to seconds)
            word_timestamps = associated words and their timestamps (defaulted to False)
            alignment = re-align timestamps (defaulted to False)
            diarization = speaker labels for utterances (defaulted to False) 

        Returns:
            YouTubeResponse
    """
    
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    params = {"url": url}
    data = {
        "alignment": alignment,  # Longer processing time but better timestamps
        "diarization": diarization,  # Longer processing time but speaker segment attribution
        "source_lang": source_lang,  # optional, default is "en"
        "timestamps": timestamps,  # optional, default is "s". Can be "s", "ms" or "hms".
        "word_timestamps": word_timestamps,  # optional, default is False
    }

    response = requests.post(
        "http://localhost:5001/api/v1/youtube",
        headers=headers,
        params=params,
        data=json.dumps(data),
    )

    
    if response.json == 200:
        r_json = response.json()
    else:
        raise ValueError("Unexpected JSON response")

    
    with open("youtube_video_output.json", "w", encoding="utf-8") as f:
        json.dump(r_json, f, indent=4, ensure_ascii=False)
        
        
def run_api_audioFile(
    file:str, 
    source_lang:str = "en",
    timestamps:str = "s",
    word_timestamps:bool = False,
    alignment:bool = False,
    diarization:bool = False,
    dual_channel:bool = False
 ):
     """
        Run API call for audio files.

        Args:
            url (str): URL source of the Youtube video.
            source_lang = language of the URL source (defaulted to English)
            timestamps = time unit of the timestamps (defaulted to seconds)
            word_timestamps = whether the timestamps are represented by words (defaulted to False)
            alignment = re-align timestamps (defaulted to False)
            diarization = speaker labels for utterances (defaulted to False) 
            dual_channel = defaulted to False

        Returns:
            AudioResponse
    """
        
    filepath = file  # or any other convertible format by ffmpeg
    data = {
        "alignment": alignment,  # Longer processing time but better timestamps
        "diarization": diarization,  # Longer processing time but speaker segment attribution
        "dual_channel": dual_channel,  # Only for stereo audio files with one speaker per channel
        "source_lang": source_lang,  # optional, default is "en"
        "timestamps": timestamps,  # optional, default is "s". Can be "s", "ms" or "hms".
        "word_timestamps": word_timestamps,  # optional, default is False
    }

    with open(filepath, "rb") as f:
        files = {"file": f}
        response = requests.post(
            "http://localhost:5001/api/v1/audio",
            files=files,
            data=data,
        )

    if response.json == 200:
        r_json = response.json()
    else:
        raise ValueError("Unexpected JSON response")

    filename = filepath.split(".")[0]
    with open(f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(r_json, f, indent=4, ensure_ascii=False)
        

# run API function that will delegate to other functions based on the endpoint         
def run_api (
    source : str,
    source_lang:str = "en",
    timestamps:str = "s",
    word_timestamps:bool = False,
    alignment:bool = False,
    diarization:bool = False,
    dual_channel:bool = False 
)
    if endpoint == "youtube":
        run_api_youtube(source, source_lang, timestamps, word_timestamps, alignment, diarization)
    elif endpoint == "audioFile":
        run_api_audioFile(source, source_lang, timestamps, word_timestamps, alignment, diarization, dual_channel)

