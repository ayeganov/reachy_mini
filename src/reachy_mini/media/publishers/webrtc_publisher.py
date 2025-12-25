"""WebRTC publisher using GStreamer appsrc.

Subscribes to the IPC bus and pushes frames into a GStreamer pipeline
for WebRTC streaming, replacing direct camera ownership with IPC consumption.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.capture import (
    AudioMetadata,
    VideoMetadata,
)
from reachy_mini.media.publishers.base import PublisherBase, PublisherConfig

if TYPE_CHECKING:
    pass


@dataclass
class WebRTCPublisherConfig(PublisherConfig):
    """Configuration for WebRTC publisher.

    Attributes:
        video_width: Video frame width.
        video_height: Video frame height.
        video_fps: Video frames per second.
        video_bitrate: H264 encoder bitrate in bits per second.
        audio_sample_rate: Audio sample rate in Hz.
        audio_channels: Number of audio channels.
        run_signalling_server: Whether to run the built-in signalling server.
        producer_name: Name for the WebRTC producer.
        log_level: Logging level string.

    """

    video_width: int = 1920
    video_height: int = 1080
    video_fps: int = 30
    video_bitrate: int = 5_000_000
    audio_sample_rate: int = 16000
    audio_channels: int = 2
    run_signalling_server: bool = True
    producer_name: str = "reachymini"
    log_level: str = "INFO"


class WebRTCPublisher(PublisherBase):
    """WebRTC publisher using GStreamer appsrc.

    Subscribes to the IPC bus and pushes raw frames into GStreamer appsrc
    elements, which feed into the WebRTC encoding and streaming pipeline.

    This replaces the old approach of directly owning libcamerasrc/alsasrc.

    Example:
        config = WebRTCPublisherConfig(
            video_width=1920,
            video_height=1080,
            video_fps=30,
        )
        publisher = WebRTCPublisher(config)
        publisher.start()
        # ... WebRTC streaming is now active
        publisher.stop()

    """

    def __init__(
        self,
        config: Optional[WebRTCPublisherConfig] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize WebRTC publisher.

        Args:
            config: Publisher configuration.
            log_level: Logging level string.

        """
        self._webrtc_config = config or WebRTCPublisherConfig()
        super().__init__(config=self._webrtc_config, log_level=log_level)

        self._gst_initialized = False
        self._loop: Any = None
        self._loop_thread: Optional[threading.Thread] = None

        self._pipeline_sender: Any = None
        self._pipeline_receiver: Any = None
        self._bus_sender: Any = None
        self._bus_receiver: Any = None

        self._video_appsrc: Any = None
        self._audio_appsrc: Any = None

        self._frame_duration_ns: int = int(1e9 / self._webrtc_config.video_fps)
        self._video_pts: int = 0
        self._audio_pts: int = 0

    def _init_gstreamer(self) -> None:
        """Initialize GStreamer and GLib main loop."""
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")

        from gi.repository import GLib, Gst

        if not self._gst_initialized:
            Gst.init(None)
            self._gst_initialized = True

        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._loop.run,
            name="WebRTCPublisher_glib",
            daemon=True,
        )
        self._loop_thread.start()

    def _init_output(self) -> None:
        """Initialize GStreamer pipelines."""
        self._init_gstreamer()
        self._create_sender_pipeline()
        self._create_receiver_pipeline()

    def _cleanup_output(self) -> None:
        """Clean up GStreamer resources."""
        from gi.repository import Gst

        if self._pipeline_sender is not None:
            self._pipeline_sender.set_state(Gst.State.NULL)
            self._pipeline_sender = None

        if self._pipeline_receiver is not None:
            self._pipeline_receiver.set_state(Gst.State.NULL)
            self._pipeline_receiver = None

        if self._loop is not None:
            self._loop.quit()
            self._loop = None

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
            self._loop_thread = None

        self._video_appsrc = None
        self._audio_appsrc = None

    def _create_sender_pipeline(self) -> None:
        """Create the GStreamer sender pipeline with appsrc."""
        from gi.repository import GLib, Gst

        self._pipeline_sender = Gst.Pipeline.new("reachymini_webrtc_sender")
        self._bus_sender = self._pipeline_sender.get_bus()
        self._bus_sender.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        webrtcsink = self._create_webrtcsink()
        self._configure_video_appsrc(webrtcsink)
        self._configure_audio_appsrc(webrtcsink)

    def _create_receiver_pipeline(self) -> None:
        """Create the GStreamer receiver pipeline for incoming audio."""
        from gi.repository import GLib, Gst

        self._pipeline_receiver = Gst.Pipeline.new("reachymini_webrtc_receiver")
        self._bus_receiver = self._pipeline_receiver.get_bus()
        self._bus_receiver.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        udpsrc = Gst.ElementFactory.make("udpsrc")
        if udpsrc is None:
            self._logger.warning("Failed to create udpsrc, skipping receiver pipeline")
            return

        udpsrc.set_property("port", 5000)

        caps = Gst.Caps.from_string(
            "application/x-rtp,media=audio,encoding-name=OPUS,payload=96"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)

        rtpjitterbuffer = Gst.ElementFactory.make("rtpjitterbuffer")
        rtpjitterbuffer.set_property("latency", 200)

        rtpopusdepay = Gst.ElementFactory.make("rtpopusdepay")
        opusdec = Gst.ElementFactory.make("opusdec")
        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")

        alsasink = Gst.ElementFactory.make("alsasink")
        alsasink.set_property("device", "reachymini_audio_sink")
        alsasink.set_property("sync", False)

        elements = [
            udpsrc,
            capsfilter,
            rtpjitterbuffer,
            rtpopusdepay,
            opusdec,
            queue,
            audioconvert,
            audioresample,
            alsasink,
        ]

        for elem in elements:
            self._pipeline_receiver.add(elem)

        udpsrc.link(capsfilter)
        capsfilter.link(rtpjitterbuffer)
        rtpjitterbuffer.link(rtpopusdepay)
        rtpopusdepay.link(opusdec)
        opusdec.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(alsasink)

    def _create_webrtcsink(self) -> Any:
        """Create and configure webrtcsink element.

        Returns:
            The webrtcsink GStreamer element.

        """
        from gi.repository import Gst

        webrtcsink = Gst.ElementFactory.make("webrtcsink")
        if not webrtcsink:
            raise RuntimeError(
                "Failed to create webrtcsink element. "
                "Is the GStreamer webrtc rust plugin installed?"
            )

        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", self._webrtc_config.producer_name)
        webrtcsink.set_property("meta", meta_structure)
        webrtcsink.set_property(
            "run-signalling-server", self._webrtc_config.run_signalling_server
        )

        self._pipeline_sender.add(webrtcsink)
        return webrtcsink

    def _configure_video_appsrc(self, webrtcsink: Any) -> None:
        """Configure video appsrc pipeline.

        Args:
            webrtcsink: The webrtcsink element to link to.

        """
        from gi.repository import Gst

        width = self._webrtc_config.video_width
        height = self._webrtc_config.video_height
        fps = self._webrtc_config.video_fps

        self._video_appsrc = Gst.ElementFactory.make("appsrc", "video_appsrc")
        self._video_appsrc.set_property("is-live", True)
        self._video_appsrc.set_property("format", Gst.Format.TIME)
        self._video_appsrc.set_property("do-timestamp", False)

        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={width},height={height},"
            f"framerate={fps}/1"
        )
        self._video_appsrc.set_property("caps", caps)

        videoconvert = Gst.ElementFactory.make("videoconvert")

        caps_i420 = Gst.Caps.from_string(
            f"video/x-raw,format=I420,width={width},height={height},"
            f"framerate={fps}/1"
        )
        capsfilter_i420 = Gst.ElementFactory.make("capsfilter")
        capsfilter_i420.set_property("caps", caps_i420)

        x264enc = Gst.ElementFactory.make("x264enc")
        if x264enc is None:
            self._logger.warning("x264enc not available, trying v4l2h264enc")
            x264enc = Gst.ElementFactory.make("v4l2h264enc")
            if x264enc is not None:
                extra_controls = Gst.Structure.new_empty("extra-controls")
                extra_controls.set_value("repeat_sequence_header", 1)
                extra_controls.set_value(
                    "video_bitrate", self._webrtc_config.video_bitrate
                )
                x264enc.set_property("extra-controls", extra_controls)
        else:
            x264enc.set_property("tune", "zerolatency")
            x264enc.set_property(
                "bitrate", self._webrtc_config.video_bitrate // 1000
            )
            x264enc.set_property("speed-preset", "ultrafast")

        if x264enc is None:
            raise RuntimeError("No H264 encoder available (tried x264enc, v4l2h264enc)")

        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
        capsfilter_h264.set_property("caps", caps_h264)

        queue = Gst.ElementFactory.make("queue")

        elements = [
            self._video_appsrc,
            videoconvert,
            capsfilter_i420,
            x264enc,
            capsfilter_h264,
            queue,
        ]

        for elem in elements:
            self._pipeline_sender.add(elem)

        self._video_appsrc.link(videoconvert)
        videoconvert.link(capsfilter_i420)
        capsfilter_i420.link(x264enc)
        x264enc.link(capsfilter_h264)
        capsfilter_h264.link(queue)
        queue.link(webrtcsink)

        self._logger.info("Video appsrc configured: %dx%d@%dfps", width, height, fps)

    def _configure_audio_appsrc(self, webrtcsink: Any) -> None:
        """Configure audio appsrc pipeline.

        Args:
            webrtcsink: The webrtcsink element to link to.

        """
        from gi.repository import Gst

        sample_rate = self._webrtc_config.audio_sample_rate
        channels = self._webrtc_config.audio_channels

        self._audio_appsrc = Gst.ElementFactory.make("appsrc", "audio_appsrc")
        self._audio_appsrc.set_property("is-live", True)
        self._audio_appsrc.set_property("format", Gst.Format.TIME)
        self._audio_appsrc.set_property("do-timestamp", False)

        caps = Gst.Caps.from_string(
            f"audio/x-raw,format=F32LE,rate={sample_rate},"
            f"channels={channels},layout=interleaved"
        )
        self._audio_appsrc.set_property("caps", caps)

        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")
        queue = Gst.ElementFactory.make("queue")

        elements = [
            self._audio_appsrc,
            audioconvert,
            audioresample,
            queue,
        ]

        for elem in elements:
            self._pipeline_sender.add(elem)

        self._audio_appsrc.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(queue)
        queue.link(webrtcsink)

        self._logger.info(
            "Audio appsrc configured: %dHz, %dch", sample_rate, channels
        )

    def _on_bus_message(
        self,
        bus: Any,
        msg: Any,
        loop: Any,
    ) -> bool:
        """Handle GStreamer bus messages.

        Args:
            bus: GStreamer bus.
            msg: GStreamer message.
            loop: GLib main loop.

        Returns:
            True to continue receiving messages.

        """
        from gi.repository import Gst

        msg_type = msg.type
        if msg_type == Gst.MessageType.EOS:
            self._logger.warning("End-of-stream")
            return False
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error("GStreamer error: %s - %s", err, debug)
            return False
        return True

    def start(self) -> bool:
        """Start the WebRTC publisher.

        Returns:
            True if started successfully, False otherwise.

        """
        result = super().start()
        if result and self._pipeline_sender is not None:
            from gi.repository import Gst

            self._pipeline_sender.set_state(Gst.State.PLAYING)
            if self._pipeline_receiver is not None:
                self._pipeline_receiver.set_state(Gst.State.PLAYING)
            self._logger.info("WebRTC pipelines started")
        return result

    def stop(self) -> None:
        """Stop the WebRTC publisher."""
        from gi.repository import Gst

        if self._pipeline_sender is not None:
            self._pipeline_sender.set_state(Gst.State.NULL)
        if self._pipeline_receiver is not None:
            self._pipeline_receiver.set_state(Gst.State.NULL)

        super().stop()

    def _process_video_frame(
        self,
        frame: npt.NDArray[np.uint8],
        metadata: VideoMetadata,
    ) -> None:
        """Push video frame to GStreamer appsrc.

        Args:
            frame: Video frame as numpy array (H, W, C).
            metadata: Frame metadata.

        """
        if self._video_appsrc is None:
            return

        from gi.repository import Gst

        data = frame.tobytes()
        buffer = Gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)

        buffer.pts = self._video_pts
        buffer.duration = self._frame_duration_ns
        self._video_pts += self._frame_duration_ns

        ret = self._video_appsrc.emit("push-buffer", buffer)
        if ret != Gst.FlowReturn.OK:
            self._logger.warning("Video appsrc push failed: %s", ret)

    def _process_audio_chunk(
        self,
        audio: npt.NDArray[np.float32],
        metadata: AudioMetadata,
    ) -> None:
        """Push audio chunk to GStreamer appsrc.

        Args:
            audio: Audio samples as numpy array.
            metadata: Audio metadata.

        """
        if self._audio_appsrc is None:
            return

        from gi.repository import Gst

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        data = audio.tobytes()
        buffer = Gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)

        samples = metadata.samples
        duration_ns = int(samples * 1e9 / metadata.sample_rate)
        buffer.pts = self._audio_pts
        buffer.duration = duration_ns
        self._audio_pts += duration_ns

        ret = self._audio_appsrc.emit("push-buffer", buffer)
        if ret != Gst.FlowReturn.OK:
            self._logger.warning("Audio appsrc push failed: %s", ret)
