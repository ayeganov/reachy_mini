"""Media constants for ZeroMQ IPC and TCP communication.

This module contains all the endpoint addresses, ports, and topics used
for media streaming between components.
"""

# IPC endpoints for local (same-machine) communication
VIDEO_IPC_ENDPOINT = "ipc:///tmp/reachy_video"
AUDIO_IPC_ENDPOINT = "ipc:///tmp/reachy_audio"

# TCP ports for remote (network) communication
VIDEO_TCP_PORT = 5555
AUDIO_TCP_PORT = 5556
AUDIO_OUTPUT_TCP_PORT = 5557

# ZeroMQ topics for pub/sub filtering
VIDEO_TOPIC = b"reachy_video"
AUDIO_TOPIC = b"reachy_audio"
AUDIO_OUTPUT_TOPIC = b"reachy_audio_out"
PLAY_SOUND_TOPIC = b"reachy_play_sound"
