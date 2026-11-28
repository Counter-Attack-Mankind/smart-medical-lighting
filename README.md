# Smart Medical Lighting System

Research and engineering prototype for multimodal perception and virtual adaptive lighting in mental-health nursing scenarios. The system observes body motion, facial appearance, speech, prosody, and environmental sounds. Its outputs are engineering candidates, not diagnoses, patient attribution, or clinical lighting prescriptions.

## Architecture

```text
live camera + microphone ----+
video file (OpenCV/FFmpeg) --+--> Pose / Face / Audio analyzers
image file (single snapshot)-+              |
                                            v
                                 FusionAnalyzer (Russell V/A)
                                            |
                                      temporal FSM
                                            |
                                 LightingController (virtual)
                                            |
                              FastAPI + WebSocket + Dashboard
```

`PerceptionSystem` owns the acquisition sources and initializes each model once per selected source. Browser clients only consume shared cached results. Switching source stops the old workers, clears temporal caches, resets the time base, and starts an isolated source session.

## Implemented functions

- MediaPipe Pose Landmarker Lite with posture, motion, walking, pacing, repetitive-motion, and transition rules.
- MediaPipe Face Landmarker with 478 landmarks, 52 blendshapes, expression candidates, facial activity, and head pose.
- Silero VAD, SenseVoiceSmall INT8, YAMNet, acoustic/prosodic features, speech state, and freshness/reliability metadata.
- Live camera/microphone, local video, and local image sources. Images are evaluated once and never replayed to manufacture temporal evidence.
- Video play, pause, replay, stop, progress, and optional audio-track processing. A video without an audio track does not open the microphone.
- Reliability- and freshness-weighted Russell valence/arousal fusion with independent dimension weights.
- Candidate/pending/confirmed finite-state machine and 30-second Russell trajectory.
- Manual or automatic virtual lighting with explicit speech requests taking priority and a rate-limited illuminance transition.
- Local Tabler dashboard with shared MJPEG previews, WebSocket metrics, and the latest 50 events.

## Fusion model

Each modality produces an optional score `x[m,d]` for dimension `d` and an observation reliability `r[m]`. Missing scores remain `None`. For each dimension:

```text
raw_weight[m,d] = base_weight[m] * reliability[m] * freshness[m]
effective_weight[m,d] = raw_weight[m,d] / sum(raw_weight[:,d])
fused[d] = sum(effective_weight[m,d] * x[m,d])
```

Fusion is unknown when the raw effective weight total is below `0.20`. Base weights are Pose 0.35, Face 0.25, speech content 0.20, and speech prosody 0.20. Pose primarily supplies arousal; face supplies valence and arousal; keyword/phrase rules supply speech-content valence and explicit rest/lighting intent; prosody and SenseVoice labels supply auxiliary acoustic evidence. YAMNet remains contextual evidence and does not receive a fifth base weight.

Scores are smoothed by a first-order exponential filter with a 1.0 s time constant. State confirmation is initially 3 s, minimum hold is 5 s, hysteresis is 0.05, and insufficient-information timeout is 3 s. Duplicate caches, paused video, and static images do not advance confirmation time.

All initial mappings and thresholds are experimental parameters in [`config/fusion_config.py`](config/fusion_config.py).

## Virtual lighting model

Confirmed candidate targets are:

| Confirmed state | Target |
|---|---:|
| tense / active | 200 lux |
| pleasant / active | 300 lux |
| low / negative | 500 lux |
| calm / relaxed | 150 lux |
| neutral | 300 lux |
| explicit rest intent | 100 lux |

Default is 300 lux, allowed range is 0--600 lux, control rate is 10 Hz, and both rise and fall rates are 30 lux/s. Manual mode always has priority. Unknown fusion holds the current target. These values are a visualization policy only; they are not measured physical illuminance or a clinical prescription. Parameters are in [`config/lighting_config.py`](config/lighting_config.py).

## Source analysis

- **Live:** camera and microphone are opened once and shared by all modules.
- **Video:** OpenCV decodes frames against one playback clock. FFmpeg extracts an optional audio track to 16 kHz mono Float32 for the existing audio pipeline. Install `ffmpeg` on `PATH`, or place the portable executable at `.tools/ffmpeg/bin/ffmpeg.exe`.
- **Image:** OpenCV decodes one frame. Pose/face static geometry is available, while motion, trajectories, and confirmed temporal state are not synthesized.

Uploaded browser files are written under `.temp/uploads/`, which is Git-ignored. They are not committed. The prototype does not persist raw live audio or microphone video.

### Using local material from the Dashboard

1. Start the Dashboard and use the compact source bar above the four status cards.
2. Select **video** to upload a local MP4, MOV, MKV, AVI, WebM, or M4V file. Analysis starts automatically after the models and optional audio track are ready. The original file name, playback state, elapsed time, total duration, and progress are displayed. Use play, pause, replay, or stop as needed.
3. Select **image** to upload a JPG, JPEG, PNG, BMP, or WebP file. The image is analyzed once for valid static Pose/Face features. It does not generate motion, trajectory, or state-confirmation evidence by repeating the same frame.
4. Select **live devices** to return to the camera and microphone. Video and image modes do not require a camera. Video without an audio track does not open the microphone.

The upload status line reports loading, success, missing audio, FFmpeg failure, and operation errors. Browser uploads use a random temporary server-side name, while the Dashboard retains and displays the original local file name. Temporary copies are removed when the source is replaced or the service stops.

## Stage 05.5.1 sensitivity tuning

The following values are initial engineering adjustments and still require real-camera calibration:

- Face: happy rules remain at a minimum score of `0.32` and a `0.40 s` confirmation time. Sad/angry/surprised minimum scores are `0.22/0.21/0.22`; confirmation times are `0.25/0.25/0.22 s`. Their necessary-action and switch margins were reduced, while multi-action evidence remains mandatory.
- Pose: the activity window changed from `5.0 s` to `2.5 s`, temporal coverage from `0.55` to `0.45`, motion confirmation from `0.45 s` to `0.25 s`, and high-activity enter/exit values from `1.15/0.90` to `0.78/0.60` body-scales per second. Activity full-scale visualization changed from `2.2` to `1.6`.
- Dashboard: the Pose card shows normalized activity score. The Face card shows all five independent rule scores; these are not probabilities and do not sum to one.

Chinese semantic rules are grouped in `config/fusion_config.py` as negative emotion, depressive expression, agitation, self-harm/suicide, conflict/threat, positive emotion, rest intent, and lighting requests. Phrase matching includes a small preceding-negation window, explicit exceptions such as `不想活了`, and reported/example context exclusions. The matched phrases are visible in the current speech card for manual tuning.

Self-harm/suicide and threat matches produce an independent **human review** alert. Their rule score is not included in Russell fusion and they do not directly change automatic lighting. History stores only the alert category, never the full sensitive transcript. These rules are recall-oriented prompts, not a diagnosis or emergency assessment system.

## Project layout

```text
config/                     analyzer, fusion, lighting, and runtime parameters
demos/                      pose_demo.py, face_demo.py, audio_demo.py
models/                     local ignored model files
src/acquisition/            camera and microphone capture
src/pose/                   pose detector/analyzer/visualizer
src/face/                   face detector/analyzer/visualizer
src/audio/                  audio detector/analyzer/visualizer
src/system/
  perception_system.py      lifecycle, scheduling, caches, source switching
  media_source.py           image/video clock and FFmpeg audio extraction
  fusion_analyzer.py        V/A mapping, dynamic weighting, temporal FSM
  lighting_controller.py    manual/automatic virtual control
web/                        FastAPI and offline Tabler dashboard
tests/                      focused unittest modules
```

## Setup and run

Windows 11 and Python 3.11 are the current development target.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Required model locations are defined in `config/pose_config.py`, `config/face_config.py`, and `config/audio_config.py`. Model weights are intentionally Git-ignored.

Start the dashboard with the retained one-click launcher:

```powershell
.\start_dashboard.bat
```

or directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Independent demos remain available:

```powershell
.\.venv\Scripts\python.exe -m demos.pose_demo
.\.venv\Scripts\python.exe -m demos.face_demo
.\.venv\Scripts\python.exe -m demos.audio_demo
```

Focused Stage 05.4--05.5 checks:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fusion_lighting -v
```

## Development status and limitations

Stages 01--05.5 are implemented at prototype level. Stage 05.6 ARM64 profiling remains pending. Fusion weights, keyword lists, thresholds, time constants, and illuminance targets require literature review and real experiments. Camera viewpoint, occlusion, distance, lighting, room acoustics, microphone response, ASR errors, and model-domain mismatch all affect results. FFmpeg-backed video audio and browser/hardware behavior must be verified manually on the target Windows machine.
