#!/usr/bin/env python3
"""Sound direction tracker with radar visualization.

This program detects sound direction using the ReSpeaker microphone array
and turns the robot head toward the sound source. A radar/compass overlay
visualizes the sound direction and robot head orientation.

Usage:
    python sound_tracker_radar.py [options]

Options:
    --threshold T   Sound detection RMS threshold (default: 0.02)
    --snap-angle A  Angle threshold for snap movement in degrees (default: 30)
    --snap-speed S  Duration for snap movements in seconds (default: 0.2)
    --smooth-speed S Duration for smooth movements in seconds (default: 0.5)
    --no-video      Run without video display (audio only)
    --log-level L   Logging level (default: INFO)
"""

import argparse
import logging
import signal
import sys
import time
from collections import deque
from contextlib import contextmanager
from typing import Generator, Optional

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini


class LoopTimer:
    """Simple timing utility for profiling loop iterations."""

    def __init__(self, history_size: int = 30) -> None:
        """Initialize timer with rolling history."""
        self.timings: dict[str, deque[float]] = {}
        self.history_size = history_size
        self._current_start: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """Context manager to measure a code block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if name not in self.timings:
                self.timings[name] = deque(maxlen=self.history_size)
            self.timings[name].append(elapsed_ms)

    def start(self, name: str) -> None:
        """Start timing a named section."""
        self._current_start[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing and record the elapsed time."""
        if name not in self._current_start:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._current_start[name]) * 1000
        if name not in self.timings:
            self.timings[name] = deque(maxlen=self.history_size)
        self.timings[name].append(elapsed_ms)
        del self._current_start[name]
        return elapsed_ms

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all timed sections."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                arr = list(times)
                stats[name] = {
                    "avg": sum(arr) / len(arr),
                    "min": min(arr),
                    "max": max(arr),
                    "last": arr[-1],
                }
        return stats

    def format_stats(self) -> str:
        """Format timing stats as a readable string."""
        stats = self.get_stats()
        if not stats:
            return "No timings recorded"

        lines = ["=== Loop Timing (ms) ==="]
        # Sort by average time descending (slowest first)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["avg"], reverse=True)
        for name, s in sorted_stats:
            lines.append(
                f"  {name:20s}: avg={s['avg']:7.2f}  last={s['last']:7.2f}  "
                f"min={s['min']:7.2f}  max={s['max']:7.2f}"
            )
        return "\n".join(lines)


class RadarOverlay:
    """Draws radar/compass visualization overlay on video frames."""

    def __init__(
        self,
        radius: int = 80,
        position: str = "bottom-right",
        margin: int = 20,
    ) -> None:
        """Initialize radar overlay.

        Args:
            radius: Radius of the radar circle in pixels.
            position: Position on frame ('bottom-right', 'bottom-left', etc.).
            margin: Margin from frame edge in pixels.

        """
        self.radius = radius
        self.position = position
        self.margin = margin

        # Colors (BGR format for OpenCV)
        self.bg_color = (40, 40, 40)  # Dark gray background
        self.border_color = (100, 100, 100)  # Light gray border
        self.sound_color = (0, 255, 0)  # Green for sound direction
        self.head_color = (255, 100, 0)  # Orange for head direction
        self.active_color = (0, 200, 255)  # Yellow for active sound
        self.text_color = (200, 200, 200)  # Light gray text

        self.sound_detected = False
        self.sound_doa = 0.0
        self.head_yaw = 0.0
        self.animation_phase = 0.0

    def update(
        self,
        sound_detected: bool,
        doa_rad: float,
        head_yaw: float,
    ) -> None:
        """Update radar state for next draw.

        Args:
            sound_detected: Whether sound above threshold was detected.
            doa_rad: Direction of arrival in radians.
            head_yaw: Current head yaw in radians.

        """
        self.sound_detected = sound_detected
        if sound_detected:
            self.sound_doa = doa_rad
        self.head_yaw = head_yaw
        self.animation_phase = (self.animation_phase + 0.15) % (2 * np.pi)

    def draw(self, frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Draw radar overlay on frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Frame with radar overlay.

        """
        h, w = frame.shape[:2]

        # Calculate radar center position
        cx, cy = self._get_center(w, h)

        # Create overlay for transparency effect
        overlay = frame.copy()

        # 1. Draw background circle
        cv2.circle(overlay, (cx, cy), self.radius, self.bg_color, -1)
        cv2.circle(overlay, (cx, cy), self.radius, self.border_color, 2)

        # 2. Draw compass tick marks and labels
        self._draw_compass(overlay, cx, cy)

        # 3. Draw sound direction if detected (as a wedge/cone)
        if self.sound_detected:
            self._draw_sound_indicator(overlay, cx, cy)
            self._draw_active_pulse(overlay, cx, cy)

        # 4. Draw head direction indicator (arrow)
        self._draw_head_indicator(overlay, cx, cy)

        # Blend overlay with original frame (transparency)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 5. Draw text labels (on top, no transparency)
        self._draw_labels(frame, cx, cy)

        return frame

    def _get_center(self, frame_w: int, frame_h: int) -> tuple[int, int]:
        """Calculate radar center based on position setting."""
        if self.position == "bottom-right":
            return (
                frame_w - self.radius - self.margin,
                frame_h - self.radius - self.margin - 30,
            )
        elif self.position == "bottom-left":
            return (
                self.radius + self.margin,
                frame_h - self.radius - self.margin - 30,
            )
        return (frame_w // 2, frame_h // 2)

    def _draw_compass(self, img: npt.NDArray[np.uint8], cx: int, cy: int) -> None:
        """Draw compass tick marks and L/F/R labels."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        # Draw tick marks at cardinal directions
        r_inner = self.radius - 10
        r_outer = self.radius - 3
        for angle in [0, 90, 180, 270]:
            rad = np.deg2rad(angle)
            x1 = int(cx + r_inner * np.sin(rad))
            y1 = int(cy - r_inner * np.cos(rad))
            x2 = int(cx + r_outer * np.sin(rad))
            y2 = int(cy - r_outer * np.cos(rad))
            cv2.line(img, (x1, y1), (x2, y2), self.border_color, 2)

        # Labels: F (front=up), L (left), R (right)
        # Front (top)
        cv2.putText(
            img,
            "F",
            (cx - 5, cy - self.radius + 20),
            font,
            font_scale,
            self.text_color,
            1,
        )
        # Left (DoA=0 -> left side of robot, which is left on radar)
        cv2.putText(
            img,
            "L",
            (cx - self.radius + 8, cy + 5),
            font,
            font_scale,
            self.text_color,
            1,
        )
        # Right (DoA=pi -> right side of robot)
        cv2.putText(
            img,
            "R",
            (cx + self.radius - 18, cy + 5),
            font,
            font_scale,
            self.text_color,
            1,
        )

    def _draw_head_indicator(
        self, img: npt.NDArray[np.uint8], cx: int, cy: int
    ) -> None:
        """Draw arrow showing current head direction."""
        # Head yaw: 0=forward, positive=left, negative=right
        # Radar convention: 0=up (forward), clockwise positive
        angle = -self.head_yaw

        length = self.radius - 20
        end_x = int(cx + length * np.sin(angle))
        end_y = int(cy - length * np.cos(angle))

        # Draw arrow with thicker line
        cv2.arrowedLine(
            img, (cx, cy), (end_x, end_y), self.head_color, 3, tipLength=0.25
        )

        # Draw center dot
        cv2.circle(img, (cx, cy), 5, self.head_color, -1)

    def _draw_sound_indicator(
        self, img: npt.NDArray[np.uint8], cx: int, cy: int
    ) -> None:
        """Draw wedge/cone showing sound direction."""
        # DoA semantics: 0=left, pi/2=front, pi=right
        # Convert to radar angle: radar 0=up (front), clockwise positive
        # DoA 0 (left) -> radar -90 (left)
        # DoA pi/2 (front) -> radar 0 (up)
        # DoA pi (right) -> radar +90 (right)
        radar_angle = self.sound_doa - np.pi / 2

        # Draw as a filled wedge
        wedge_half_width = np.deg2rad(15)  # 30 degree total wedge width

        # Calculate wedge points
        r = self.radius - 12
        points = [(cx, cy)]

        # Arc from start to end angle
        start_angle = radar_angle - wedge_half_width
        end_angle = radar_angle + wedge_half_width
        num_points = 20
        for i in range(num_points + 1):
            a = start_angle + (end_angle - start_angle) * i / num_points
            px = int(cx + r * np.sin(a))
            py = int(cy - r * np.cos(a))
            points.append((px, py))

        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [points_array], self.sound_color)

    def _draw_active_pulse(self, img: npt.NDArray[np.uint8], cx: int, cy: int) -> None:
        """Draw pulsing ring when sound is active."""
        pulse_factor = 0.5 + 0.5 * np.sin(self.animation_phase)
        pulse_radius = int(self.radius + 5 + 8 * pulse_factor)
        thickness = max(1, int(2 * pulse_factor + 1))
        cv2.circle(img, (cx, cy), pulse_radius, self.active_color, thickness)

    def _draw_labels(self, img: npt.NDArray[np.uint8], cx: int, cy: int) -> None:
        """Draw status labels below radar."""
        y_base = cy + self.radius + 15
        font = cv2.FONT_HERSHEY_SIMPLEX

        # DoA value in degrees
        doa_deg = np.rad2deg(self.sound_doa)
        doa_text = f"DoA: {doa_deg:6.1f} deg"
        cv2.putText(img, doa_text, (cx - 55, y_base), font, 0.45, (255, 255, 255), 1)

        # Sound status indicator
        status = "SOUND" if self.sound_detected else "quiet"
        color = self.sound_color if self.sound_detected else (100, 100, 100)
        cv2.putText(img, status, (cx - 25, y_base + 18), font, 0.45, color, 1)


class MovementController:
    """Hybrid movement controller: snappy for large changes, smooth for small."""

    def __init__(
        self,
        large_angle_threshold: float = np.deg2rad(30),
        snap_duration: float = 0.2,
        smooth_duration: float = 0.5,
        min_movement_threshold: float = np.deg2rad(5),
    ) -> None:
        """Initialize movement controller.

        Args:
            large_angle_threshold: Angle in radians above which to use snap movement.
            snap_duration: Duration for snap (large angle) movements.
            smooth_duration: Duration for smooth (small angle) movements.
            min_movement_threshold: Deadband - ignore changes smaller than this.

        """
        self.large_angle_threshold = large_angle_threshold
        self.snap_duration = snap_duration
        self.smooth_duration = smooth_duration
        self.min_movement_threshold = min_movement_threshold
        self.last_move_time = 0.0

    def calculate_movement(
        self, target_yaw: float, current_yaw: float
    ) -> tuple[float, float]:
        """Determine movement parameters based on angle difference.

        Args:
            target_yaw: Target head yaw in radians.
            current_yaw: Current head yaw in radians.

        Returns:
            Tuple of (target_yaw, duration). Duration of 0 means no movement.

        """
        angle_diff = abs(target_yaw - current_yaw)

        # Deadband - ignore very small changes
        if angle_diff < self.min_movement_threshold:
            return current_yaw, 0.0

        # Hybrid logic: snap for large, smooth for small
        if angle_diff > self.large_angle_threshold:
            return target_yaw, self.snap_duration
        else:
            return target_yaw, self.smooth_duration


class SoundTracker:
    """Main sound tracking application with radar visualization."""

    def __init__(
        self,
        mini: ReachyMini,
        threshold: float = 0.02,
        snap_angle_deg: float = 30.0,
        snap_speed: float = 0.2,
        smooth_speed: float = 0.5,
        video_enabled: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize sound tracker.

        Args:
            mini: ReachyMini instance.
            threshold: RMS threshold for sound detection.
            snap_angle_deg: Angle threshold for snap vs smooth movement.
            snap_speed: Duration for snap movements.
            smooth_speed: Duration for smooth movements.
            video_enabled: Whether to display video.
            logger: Optional logger instance.

        """
        self.mini = mini
        self.threshold = threshold
        self.video_enabled = video_enabled
        self.logger = logger or logging.getLogger(__name__)

        self.movement = MovementController(
            large_angle_threshold=np.deg2rad(snap_angle_deg),
            snap_duration=snap_speed,
            smooth_duration=smooth_speed,
        )

        self.radar = RadarOverlay(radius=80, position="bottom-right")

        self.running = False
        self.current_head_yaw = 0.0
        self.last_doa = np.pi / 2  # Default to front
        self.last_move_time = 0.0
        self.min_move_interval = 0.3  # Don't move more often than this

        # Timing instrumentation
        self.timer = LoopTimer(history_size=60)
        self.loop_count = 0
        self.timing_report_interval = 60  # Print timing every N loops

    def run(self) -> None:
        """Run the main application loop."""
        self.running = True

        assert self.mini.media

        if self.video_enabled:
            cv2.namedWindow("Sound Tracker", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Sound Tracker", 960, 540)

        self.logger.info("Sound tracker started. Press 'q' or ESC to quit.")
        self.logger.info(
            "Timing instrumentation enabled - stats printed every %d loops",
            self.timing_report_interval,
        )

        while self.running:
            self.timer.start("TOTAL_LOOP")

            # 1. Get audio sample with metadata
            with self.timer.measure("1_get_audio"):
                audio_result = self.mini.media.get_audio_sample_with_metadata()

            sound_detected = False
            doa_rad = self.last_doa

            if audio_result is not None:
                audio, metadata = audio_result

                # Check sound level (RMS)
                with self.timer.measure("2_calc_rms"):
                    rms = float(np.sqrt(np.mean(audio**2)))
                    sound_detected = rms > self.threshold

                # Get DoA from metadata
                if metadata.get("doa_rad") is not None:
                    doa_rad = float(metadata["doa_rad"])
                    self.last_doa = doa_rad

                if sound_detected:
                    self.logger.debug(
                        f"Sound detected: RMS={rms:.4f}, DoA={np.rad2deg(doa_rad):.1f}deg"
                    )

            # 2. Get current head pose and extract yaw
            with self.timer.measure("3_get_head_pose"):
                try:
                    T_current = self.mini.get_current_head_pose()
                    current_euler = R.from_matrix(T_current[:3, :3]).as_euler("xyz")
                    self.current_head_yaw = current_euler[2]  # Z-axis rotation (yaw)
                except Exception as e:
                    self.logger.warning(f"Could not get head pose: {e}")

            # 3. Move toward sound if detected
            with self.timer.measure("4_move_sound"):
                if sound_detected:
                    self._move_toward_sound(doa_rad)

            # 4. Update visualization
            with self.timer.measure("5_radar_update"):
                self.radar.update(sound_detected, doa_rad, self.current_head_yaw)

            # 5. Display video with overlay
            if self.video_enabled:
                with self.timer.measure("6_get_frame"):
                    frame = self.mini.media.get_frame() if self.mini.media else None

                if frame is not None:
                    with self.timer.measure("7_radar_draw"):
                        frame = self.radar.draw(frame)

                    with self.timer.measure("8_cv2_imshow"):
                        cv2.imshow("Sound Tracker", frame)

                with self.timer.measure("9_cv2_waitKey"):
                    key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:  # 'q' or ESC
                    self.stop()
            else:
                time.sleep(0.05)  # Small delay when not displaying video

            self.timer.stop("TOTAL_LOOP")

            # Print timing report periodically
            self.loop_count += 1
            if self.loop_count % self.timing_report_interval == 0:
                self.logger.info("\n%s", self.timer.format_stats())

    def _move_toward_sound(self, doa_rad: float) -> None:
        """Convert DoA to head position and move.

        Uses the same approach as sound_doa.py: create a point in head frame
        and use look_at_world() after transforming to world coordinates.

        Args:
            doa_rad: Direction of arrival in radians.

        """
        # Rate limit movements
        now = time.time()
        if now - self.last_move_time < self.min_move_interval:
            return

        # Calculate target yaw from DoA
        # DoA: 0=left, pi/2=front, pi=right
        # Convert to yaw: positive=left, 0=front, negative=right
        target_yaw = np.pi / 2 - doa_rad

        # Get movement parameters using hybrid controller
        with self.timer.measure("4a_calc_movement"):
            target, duration = self.movement.calculate_movement(
                target_yaw, self.current_head_yaw
            )

        if duration > 0:
            # Create a point in head frame based on DoA
            # This is the approach from sound_doa.py
            with self.timer.measure("4b_calc_target"):
                p_head = np.array([np.sin(doa_rad), np.cos(doa_rad), 0.0])

                # Transform to world coordinates
                T_world_head = self.mini.get_current_head_pose()
                R_world_head = T_world_head[:3, :3]
                p_world = R_world_head @ p_head

            self.logger.debug(
                f"Moving to DoA={np.rad2deg(doa_rad):.1f}deg, "
                f"world point=({p_world[0]:.2f}, {p_world[1]:.2f}, {p_world[2]:.2f}), "
                f"duration={duration:.2f}s"
            )

            try:
                # THIS IS LIKELY THE BLOCKING CALL
                with self.timer.measure("4c_look_at_world"):
                    self.mini.look_at_world(
                        x=float(p_world[0]),
                        y=float(p_world[1]),
                        z=float(p_world[2]),
                        duration=0,
                    )
                self.last_move_time = now
            except Exception as e:
                self.logger.warning(f"Movement failed: {e}")

    def stop(self) -> None:
        """Stop the tracker."""
        self.running = False
        if self.video_enabled:
            cv2.destroyAllWindows()
        self.logger.info("Sound tracker stopped.")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Sound direction tracker with radar visualization"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Sound detection RMS threshold (default: 0.02)",
    )
    parser.add_argument(
        "--snap-angle",
        type=float,
        default=30.0,
        help="Angle threshold for snap movement in degrees (default: 30)",
    )
    parser.add_argument(
        "--snap-speed",
        type=float,
        default=0.2,
        help="Duration for snap movements in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--smooth-speed",
        type=float,
        default=0.5,
        help="Duration for smooth movements in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Run without video display",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    tracker: Optional[SoundTracker] = None

    def signal_handler(sig: int, frame: object) -> None:
        """Handle shutdown signals."""
        logger.info("Received signal %d, shutting down...", sig)
        if tracker is not None:
            tracker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Connecting to Reachy Mini...")
        with ReachyMini(
            localhost_only=False,
            log_level=args.log_level,
            automatic_body_yaw=True,
        ) as mini:
            if mini.media is None:
                logger.error("Media is not available. Check daemon configuration.")
                sys.exit(1)

            tracker = SoundTracker(
                mini=mini,
                threshold=args.threshold,
                snap_angle_deg=args.snap_angle,
                snap_speed=args.snap_speed,
                smooth_speed=args.smooth_speed,
                video_enabled=not args.no_video,
                logger=logger,
            )

            logger.info("Sound Tracker starting...")
            logger.info("  Threshold: %.3f", args.threshold)
            logger.info("  Snap angle: %.1f deg", args.snap_angle)
            logger.info("  Snap speed: %.2f s", args.snap_speed)
            logger.info("  Smooth speed: %.2f s", args.smooth_speed)
            logger.info("  Video: %s", "enabled" if not args.no_video else "disabled")
            logger.info("Press 'q' or ESC to quit")

            tracker.run()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
