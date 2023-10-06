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

__Need to be updated__

### ASRLiveService

* `/live` - The live streaming endpoint.

```python
@router.websocket("")
async def websocket_endpoint(source_lang: str, websocket: WebSocket) -> None:
    """Handle WebSocket connections."""
```

This endpoint expects a WebSocket connection and a `source_lang` parameter as a string, and will return the transcription results in real-time.

### ASRTranscriptionOnly

* `/transcribe` - The transcription endpoint.

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

* `/diarize` - The diarization endpoint.

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
