import os
import tempfile
import time

import numpy as np
import pytest
import soundfile as sf

from reachy_mini.media.audio_sounddevice import SoundDeviceAudio
from reachy_mini.media.audio_utils import _process_card_number_output


@pytest.fixture
def audio_device():
    """Create SoundDeviceAudio instance for tests."""
    audio = SoundDeviceAudio()
    yield audio
    audio.stop_recording()
    audio.stop_playing()


@pytest.mark.audio
def test_play_sound_default_backend(audio_device) -> None:
    """Test playing a sound with SoundDeviceAudio."""
    # Use a short sound file present in your assets directory
    sound_file = "wake_up.wav"  # Change to a valid file if needed
    audio_device.start_playing()
    audio_device.play_sound(sound_file)
    print("Playing sound with SoundDeviceAudio...")
    # Wait a bit to let the sound play (non-blocking backend)
    time.sleep(2)
    audio_device.stop_playing()
    # No assertion: test passes if no exception is raised.
    # Sound should be audible if the audio device is correctly set up.


@pytest.mark.audio
def test_push_audio_sample_default_backend(audio_device) -> None:
    """Test pushing an audio sample with SoundDeviceAudio."""
    audio_device.start_playing()
    samplerate = audio_device.get_output_audio_samplerate()
    output_channels = audio_device.get_output_channels()

    # Generate 1 second of random audio matching output channels
    data = np.random.random((samplerate, output_channels)).astype(np.float32)
    audio_device.push_audio_sample(data)
    time.sleep(1)

    # Mono input (will be handled by audio device)
    data = np.random.random((samplerate, 1)).astype(np.float32)
    audio_device.push_audio_sample(data)
    time.sleep(1)

    audio_device.stop_playing()
    # No assertion: test passes if no exception is raised.


@pytest.mark.audio
def test_record_audio_and_file_exists(audio_device) -> None:
    """Test recording audio and check that the file exists and is not empty."""
    DURATION = 2  # seconds
    tmpfile = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmpfile.close()

    audio_device.start_recording()
    time.sleep(DURATION)
    audio_device.stop_recording()

    audio = audio_device.get_audio_sample()
    samplerate = audio_device.get_input_audio_samplerate()

    assert audio is not None
    sf.write(tmpfile.name, audio, samplerate)
    assert os.path.exists(tmpfile.name)
    assert os.path.getsize(tmpfile.name) > 0
    # comment the following line if you want to keep the file for inspection
    os.remove(tmpfile.name)


@pytest.mark.audio
def test_record_audio_without_start_recording(audio_device) -> None:
    """Test recording audio without starting recording."""
    audio = audio_device.get_audio_sample()
    assert audio is None


@pytest.mark.audio
def test_record_audio_above_max_queue_seconds(audio_device) -> None:
    """Test recording audio and check that the maximum queue seconds is respected."""
    audio_device._input_max_queue_seconds = 1
    audio_device.start_recording()
    time.sleep(5)
    audio = audio_device.get_audio_sample()
    audio_device.stop_recording()

    assert audio is not None
    assert audio.shape[0] < audio_device._input_max_queue_samples


@pytest.mark.audio
def test_DoA(audio_device) -> None:
    """Test Direction of Arrival (DoA) estimation."""
    doa = audio_device.get_DoA()
    assert doa is not None
    assert isinstance(doa, tuple)
    assert len(doa) == 2
    assert isinstance(doa[0], float)
    assert isinstance(doa[1], bool)


def test_get_respeaker_card_number() -> None:
    """Test getting the ReSpeaker card number."""
    alsa_output = "carte 5 : Audio [Reachy Mini Audio], périphérique 0 : USB Audio [USB Audio]"
    card_number = _process_card_number_output(alsa_output)
    assert isinstance(card_number, int)
    assert card_number == 5
    alsa_output = "card 0: Audio [Reachy Mini Audio], device 0: USB Audio [USB Audio]"
    card_number = _process_card_number_output(alsa_output)
    assert card_number == 0
    alsa_output = "card 3: PCH [HDA Intel PCH], device 0: ALC255 Analog [ALC255 Analog]"
    card_number = _process_card_number_output(alsa_output)
    assert card_number == 0
