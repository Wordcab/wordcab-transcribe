There are four different ASR Engines and the right one is chosen based on the [`asr_type`](usage/env/#asr-type-configuration) parameter.

## Engines

* `ASRAsyncService`: the main engine that handles jobs and define the execution mode for transcription and diarization (post-processing is always locally done by this engine).

* `ASRLiveService`: the engine that handles live streaming requests.

* `ASRTranscriptionOnly`: the engine when you want to deploy a single transcription remote server.

* `ASRDiarizationOnly`: the engine when you want to deploy a single diarization remote server.

!!! warning
    The `ASRTranscriptionOnly` and `ASRDiarizationOnly` engines aren't meant to be used alone. They are used only when you want to deploy each service in a separate server and they will need to be used along with the `ASRAsyncService` engine.

## Endpoints

Each Engine has its own endpoints as described below.

### ASRAsyncService

#### Transcription endpoints

These endpoints are the main endpoints for transcribing audio files.

* `/audio` <span style="color: #66ccff;">[POST]</span> - The audio endpoint for transcribing local files.

```python
@router.post(
    "", response_model=Union[AudioResponse, str], status_code=http_status.HTTP_200_OK
)
async def inference_with_audio(
    background_tasks: BackgroundTasks,
    offset_start: Union[float, None] = Form(None),
    offset_end: Union[float, None] = Form(None),
    num_speakers: int = Form(-1),
    diarization: bool = Form(False),
    multi_channel: bool = Form(False),
    source_lang: str = Form("en"),
    timestamps: str = Form("s"),
    vocab: Union[List[str], None] = Form(None),
    word_timestamps: bool = Form(False),
    internal_vad: bool = Form(False),
    repetition_penalty: float = Form(1.2),
    compression_ratio_threshold: float = Form(2.4),
    log_prob_threshold: float = Form(-1.0),
    no_speech_threshold: float = Form(0.6),
    condition_on_previous_text: bool = Form(True),
    file: UploadFile = File(...),
) -> AudioResponse:
    """Inference endpoint with audio file."""
```

!!!note
    The local `/audio` endpoint is expecting a `file` parameter and all the other parameters are optional and have default values.

* `/audio-url` <span style="color: #66ccff;">[POST]</span> - The audio endpoint for transcribing remote files using a URL.

```python
@router.post("", response_model=AudioResponse, status_code=http_status.HTTP_200_OK)
async def inference_with_audio_url(
    background_tasks: BackgroundTasks,
    url: str,
    data: Optional[AudioRequest] = None,
) -> AudioResponse:
    """Inference endpoint with audio url."""
```

Here is the `AudioRequest` model which inherits from the `BaseRequest` model:

```python
class BaseRequest(BaseModel):
    """Base request model for the API."""

    offset_start: Union[float, None] = None
    offset_end: Union[float, None] = None
    num_speakers: int = -1
    diarization: bool = False
    source_lang: str = "en"
    timestamps: Timestamps = Timestamps.seconds
    vocab: Union[List[str], None] = None
    word_timestamps: bool = False
    internal_vad: bool = False
    repetition_penalty: float = 1.2
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True

class AudioRequest(BaseRequest):
    """Request model for the ASR audio file and url endpoint."""

    multi_channel: bool = False
```

Here is the `AudioResponse` model which inherits from the `BaseResponse` model:

```python
class BaseResponse(BaseModel):
    """Base response model, not meant to be used directly."""

    utterances: List[Utterance]
    audio_duration: float
    offset_start: Union[float, None]
    offset_end: Union[float, None]
    num_speakers: int
    diarization: bool
    source_lang: str
    timestamps: str
    vocab: Union[List[str], None]
    word_timestamps: bool
    internal_vad: bool
    repetition_penalty: float
    compression_ratio_threshold: float
    log_prob_threshold: float
    no_speech_threshold: float
    condition_on_previous_text: bool
    process_times: ProcessTimes


class AudioResponse(BaseResponse):
    """Response model for the ASR audio file and url endpoint."""

    multi_channel: bool
```

* `youtube` <span style="color: #66ccff;">[POST]</span> - The audio endpoint for transcribing YouTube videos using a YouTube video link.

```python
@router.post("", response_model=YouTubeResponse, status_code=http_status.HTTP_200_OK)
async def inference_with_youtube(
    background_tasks: BackgroundTasks,
    url: str,
    data: Optional[BaseRequest] = None,
) -> YouTubeResponse:
    """Inference endpoint with YouTube url."""
```

!!!note
    As you can see the only difference is that the YouTube endpoint has the same `BaseRequest` model as the `/audio-url` endpoint but without the `multi_channel` parameter.

Here is the `YouTubeResponse` model which inherits from the `BaseResponse` model:

```python
class YouTubeResponse(BaseResponse):
    """Response model for the ASR YouTube endpoint."""

    video_url: str
```

#### Management endpoints

These endpoints are used to manage the remote servers URLs, when you want to deploy the `ASRTranscriptionOnly` or `ASRDiarizationOnly` engines in separate servers.

* `/url` <span style="color: #ccff66;">[GET]</span> - This endpoint allow listing the remote servers URLs.

```python
@router.get(
    "",
    response_model=Union[List[HttpUrl], str],
    status_code=http_status.HTTP_200_OK,
)
async def get_url(task: Literal["transcription", "diarization"]) -> List[HttpUrl]:
    """Get Remote URL endpoint for remote transcription or diarization."""
```

* `/url/add` <span style="color: #66ccff;">[POST]</span> - This endpoint allow adding a remote server URL.

```python
@router.post(
    "/add",
    response_model=Union[UrlSchema, str],
    status_code=http_status.HTTP_200_OK,
)
async def add_url(data: UrlSchema) -> UrlSchema:
    """Add Remote URL endpoint for remote transcription or diarization."""
```

* `/url/remove` <span style="color: #66ccff;">[POST]</span> - This endpoint allow removing a remote server URL.

```python
@router.post(
    "/remove",
    response_model=Union[UrlSchema, str],
    status_code=http_status.HTTP_200_OK,
)
async def remove_url(data: UrlSchema) -> UrlSchema:
    """Remove Remote URL endpoint for remote transcription or diarization."""
```

Here is the `UrlSchema` model:

```python
class UrlSchema(BaseModel):
    """Request model for the add_url endpoint."""

    task: Literal["transcription", "diarization"]
    url: HttpUrl
```

The `url` parameter needs to be a valid URL (check pydantic [HttpUrl](https://docs.pydantic.dev/latest/api/networks/#pydantic.networks.HttpUrl)) and the `task` parameter needs to be either `transcription` or `diarization`.

### ASRLiveService

* `/live` <span style="color: #ff66cc;">[WEBSOCKET]</span> - The live streaming endpoint.

```python
@router.websocket("")
async def websocket_endpoint(source_lang: str, websocket: WebSocket) -> None:
    """Handle WebSocket connections."""
```

This endpoint expects a WebSocket connection and a `source_lang` parameter as a string, and will return the transcription results in real-time.

### ASRTranscriptionOnly

!!!warning
    This endpoint is not meant to be used alone. It is used only when you want to deploy the `ASRTranscriptionOnly` engine in a separate server and it will need to be used along with the `ASRAsyncService` engine.

* `/transcribe` <span style="color: #66ccff;">[POST]</span> - The transcription endpoint.

```python
@router.post(
    "",
    response_model=Union[TranscriptionOutput, List[TranscriptionOutput], str],
    status_code=http_status.HTTP_200_OK,
)
async def only_transcription(
    data: TranscribeRequest,
) -> Union[TranscriptionOutput, List[TranscriptionOutput]]:
    """Transcribe endpoint for the `only_transcription` asr type."""
```

This endpoint expects a `TranscribeRequest` and will return the transcription results as a `TranscriptionOutput` or a list of `TranscriptionOutput` if the task is a `multi_channel` task.

Here is the `TranscribeRequest` model:

```python
class TranscribeRequest(BaseModel):
    """Request model for the transcribe endpoint."""

    audio: Union[TensorShare, List[TensorShare]]
    compression_ratio_threshold: float
    condition_on_previous_text: bool
    internal_vad: bool
    log_prob_threshold: float
    no_speech_threshold: float
    repetition_penalty: float
    source_lang: str
    vocab: Union[List[str], None]
```

Here is the `TranscriptionOutput` model:

```python
class TranscriptionOutput(BaseModel):
    """Transcription output model for the API."""

    segments: List[Segment]
```

### ASRDiarizationOnly

!!! warning
    This endpoint is not meant to be used alone. It is used only when you want to deploy the `ASRDiarizationOnly` engine in a separate server and it will need to be used along with the `ASRAsyncService` engine.

* `/diarize` <span style="color: #66ccff;">[POST]</span> - The diarization endpoint.

```python
@router.post(
    "",
    response_model=Union[DiarizationOutput, str],
    status_code=http_status.HTTP_200_OK,
)
async def remote_diarization(
    data: DiarizationRequest,
) -> DiarizationOutput:
    """Diarize endpoint for the `only_diarization` asr type."""
```

This endpoint expects a `DiarizationRequest` and will return the diarization results as a `DiarizationOutput`.

Here is the `DiarizationRequest` model:

```python
class DiarizationRequest(BaseModel):
    """Request model for the diarize endpoint."""

    audio: TensorShare
    duration: float
    num_speakers: int
```

Here is the `DiarizationSegment` and `DiarizationOutput` models:

```python
class DiarizationSegment(NamedTuple):
    """Diarization segment model for the API."""

    start: float
    end: float
    speaker: int


class DiarizationOutput(BaseModel):
    """Diarization output model for the API."""

    segments: List[DiarizationSegment]
```

## Execution modes

The execution modes represent the way the tasks are executed, either locally or remotely.
For each task, one execution mode is defined.

There are two different execution modes: `LocalExecution` and `RemoteExecution`.

* `LocalExecution` is the default execution mode. It executes the pipeline on the local machine. It is useful for testing and debugging.

```python
class LocalExecution(BaseModel):
    """Local execution model."""

    index: Union[int, None]
```

The local execution is looking for any local GPU device. If there is no GPU device, it will use the CPU.
If there are multiple GPU devices, it will use all alternatively.
The `index` parameter keep the track of the GPU index assigned to the task.

* `RemoteExecution` executes the pipeline on a remote machine. It is useful for production and scaling.

!!! note
    The remote execution mode is only available if you have added `transcribe_server_urls` or `diarization_server_urls` in the configuration file or on the fly via the API. Check the [Environment variables](/usage/env) section for more information.

```python
class RemoteExecution(BaseModel):
    """Remote execution model."""

    url: str
```

The `url` parameter is the URL of the remote machine that will be used for the execution of a task.
