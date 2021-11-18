import logging
import logging.handlers
import queue
import threading
from _thread import start_new_thread
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, List
from starlette.websockets import WebSocket
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
from streamlit.report_thread import add_report_ctx


HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


def main():
    """Main loop that runs the app in streamlit"""
    st.header("Real Time Speech-to-Text")
    st.markdown(
        """
This demo app is using conformerctc
"""
    )

    app_sst(endpoint='ws://104.197.76.238:23000/ws', ORIGINAL_SR=48000, VAD_SR=48000)

def get_webrtc_context(key: str = "speech-to-text") -> WebRtcStreamerContext:
    """Build a context to manage connection by webrtc to the mic"""
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    client = ClientSettings(
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": False, "audio": True})
    # Unlike other Streamlit components, webrtc_streamer() requires the key argument as a unique identifier.
    # Set an arbitrary string to it.
    return webrtc_streamer(key=key,mode=WebRtcMode.SENDONLY,audio_receiver_size=1024,client_settings=client)

def read_transcript(client: WebSocket, responses: List) ->List[str]:
    """Read from websocket"""
    try:
        response = client.recv()
        # print(f"reciving response:\n\t{response}\n")
        text_output = st.empty()
        logging.info(f"response: {response}")
        text_output.markdown(f"**Text:** {response}")
        responses.append(json.loads(response))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

def start_read_thread(client: WebSocket, responses: List) -> None:
    """Init read thread"""
    thread = threading.Thread(target=read_transcript, args=(client,responses))
    add_report_ctx(thread)
    thread.start()

def init_client(endpoint: str, ORIGINAL_SR: int, VAD_SR: int, responses: List, status_indicator: Any) -> None:
    """Init websocket and read thread"""
    conn = f'{endpoint}?&source_sr={ORIGINAL_SR}&vad_sr={VAD_SR}'
    logger.info(f"Connecting to: {conn}")
    client = create_connection(conn)
    start_read_thread(client, responses)
    status_indicator.write("Model loaded.")
    return client

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

def send_audio_frames(audio_frames: List, client: WebSocket, ORIGINAL_SR: int) -> None:
    """Send audio frames to the websocket"""
    # Cumulate sound chunks
    sound_chunk = cum_sound_chunks(audio_frames)
    # Feed stream of audio to stt server
    if len(sound_chunk) > 0:
        sound_chunk = sound_chunk.set_channels(1).set_frame_rate(ORIGINAL_SR)
        # buffer = np.array(sound_chunk.get_array_of_samples())
        buffer = sound_chunk.raw_data
        client.send_binary(buffer)

def app_sst(endpoint: str, ORIGINAL_SR:int, VAD_SR: int):
    """Speech-to-text"""

    webrtc_ctx = get_webrtc_context()
    status_indicator = st.empty()
    if not webrtc_ctx.state.playing:
        return
    status_indicator.write("Loading...")
    client = None
    responses = []

    while True:
        if webrtc_ctx.audio_receiver:
            if client is None:
                client = init_client(endpoint, ORIGINAL_SR, VAD_SR, responses, status_indicator)
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue
            status_indicator.write("Running. Say something!")
            send_audio_frames(audio_frames, client, ORIGINAL_SR)
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
