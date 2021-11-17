import logging
import logging.handlers
import queue
import threading
from _thread import start_new_thread
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List
from websocket import create_connection
from fastapi import WebSocketDisconnect
import json


import av
import numpy as np
import pydub
from pydub.audio_segment import AudioSegment
import streamlit as st

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    WebRtcMode,
    webrtc_streamer,
)
from streamlit_webrtc.component import WebRtcStreamerContext

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    """Downloads file from url"""
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def main():
    """Main loop that runs the app in streamlit"""
    st.header("Real Time Speech-to-Text")
    st.markdown(
        """
This demo app is using [DeepSpeech](https://github.com/mozilla/DeepSpeech),
an open speech-to-text engine.

A pre-trained model released with
[v0.9.3](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3),
trained on American English is being served.
"""
    )

    # https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
    MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
    LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
    download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

    lm_alpha = 0.931289039105002
    lm_beta = 1.1834137581510284
    beam = 100

    app_sst(
        str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
    )

def get_webrtc_context(key: str = "speech-to-text") -> WebRtcStreamerContext:
    """Build a context to manage connection by webrtc to the mic"""
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    client = ClientSettings(
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": False, "audio": True})
    # Unlike other Streamlit components, webrtc_streamer() requires the key argument as a unique identifier.
    # Set an arbitrary string to it.
    return webrtc_streamer(key=key,mode=WebRtcMode.SENDONLY,audio_receiver_size=1024,client_settings=client)

def cum_sound_chunks(audio_frames: List) -> AudioSegment:
    """cummulate sound frames"""
    sound_chunk = pydub.AudioSegment.empty()
    for audio_frame in audio_frames:
        sound = pydub.AudioSegment(
            data=audio_frame.to_ndarray().tobytes(),
            sample_width=audio_frame.format.bytes,
            frame_rate=audio_frame.sample_rate,
            channels=len(audio_frame.layout.channels),
        )
        sound_chunk += sound
    return sound_chunk

def read_transcript(client, responses) ->List[str]:
    try:
        response = client.recv()
        print(f"reciving response:\n\t{response}\n")
        responses.append(json.loads(response))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

def app_sst(model_path: str, endpoint: str, ORIGINAL_SR:int, VAD_SR: int):
    """Speech-to-text"""

    webrtc_ctx = get_webrtc_context()
    
    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None
    responses = []

    while True:
        if webrtc_ctx.audio_receiver:
            if client is None:
                # from deepspeech import Model
                # model = Model(model_path)
                # model.enableExternalScorer(lm_path)
                # model.setScorerAlphaBeta(lm_alpha, lm_beta)
                # model.setBeamWidth(beam)
                # stream = model.createStream()
                client = create_connection(f'{endpoint}?&source_sr={ORIGINAL_SR}&vad_sr={VAD_SR}')
                start_new_thread(read_transcript, (client,responses))
                status_indicator.write("Model loaded.")

            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")
            # Cumulate sound chunks
            sound_chunk = cum_sound_chunks(audio_frames)
            # Feed stream of audio to stt server
            if len(sound_chunk) > 0:
                # sound_chunk = sound_chunk.set_channels(1).set_frame_rate(model.sampleRate())
                buffer = np.array(sound_chunk.get_array_of_samples())
                # stream.feedAudioContent(buffer)
                # text = stream.intermediateDecode()
                # text_output.markdown(f"**Text:** {text}")
                # text_output.markdown(f"**Text:** {text}")
                client.send_binary(buffer)
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
