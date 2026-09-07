from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "client" / "arx"))

from tau0vla_protocol import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    ActionChunk,
    ActionEMA,
    ChunkScheduler,
    RollingDelayEstimator,
)


def _chunk(request_id: int, *, delay: int = 0, prefix=None, offset: float = 0.0) -> ActionChunk:
    actions = np.repeat(np.arange(ACTION_HORIZON, dtype=np.float32)[:, None], ACTION_DIM, axis=1)
    actions += offset
    if prefix is not None:
        actions[:delay] = prefix
    return ActionChunk(
        actions=actions,
        request_id=request_id,
        sample_monotonic_ns=1,
        round_trip_ms=50.0,
        inference_ms=40.0,
        model_id="test",
        rtc_delay=delay,
        action_prefix=None if prefix is None else np.asarray(prefix, dtype=np.float32),
    )


class DelayEstimatorTest(unittest.TestCase):
    def test_rolling_max_margin_and_window(self):
        estimator = RollingDelayEstimator([10.0, 70.0], margin_ms=10.0, window_size=3)
        self.assertEqual(estimator.predict(), 3)
        estimator.observe(20.0)
        estimator.observe(30.0)  # evicts 10, but 70 remains
        self.assertEqual(estimator.predict(), 3)
        estimator.observe(40.0)  # evicts 70
        self.assertEqual(estimator.predict(), 2)


class SchedulerRTCTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = ChunkScheduler(replan_steps=5, blend_steps=2)
        self.scheduler.adopt(_chunk(1), initial=True)
        self.ema = ActionEMA(arm_alpha=0.5, gripper_alpha=0.5)
        self.ema.reset(np.zeros(ACTION_DIM, dtype=np.float32))

    def test_preview_has_no_scheduler_or_ema_side_effects(self):
        before_remaining = self.scheduler.remaining
        first = self.scheduler.preview(3, self.ema)
        second = self.scheduler.preview(3, self.ema)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(self.scheduler.remaining, before_remaining)
        actual = self.ema.apply_scheduled(self.scheduler.next_action())
        np.testing.assert_array_equal(actual, first[0])

    def test_tick_aligned_adoption_bypasses_prefix_ema_and_blends_postfix_only(self):
        for _ in range(5):
            self.ema.apply_scheduled(self.scheduler.next_action())
        delay = 4
        prefix = self.scheduler.preview(delay, self.ema)
        response = _chunk(2, delay=delay, prefix=prefix, offset=100.0)
        # Two commands were actually published while inference was in flight.
        for _ in range(2):
            self.ema.apply_scheduled(self.scheduler.next_action())
        info = self.scheduler.adopt(response, actual_elapsed=2)
        self.assertTrue(info.adopted)
        self.assertEqual(info.skipped, 2)
        self.assertEqual(info.blended_steps, 2)
        first = self.scheduler.next_action()
        second = self.scheduler.next_action()
        self.assertTrue(first.ema_passthrough)
        self.assertTrue(second.ema_passthrough)
        np.testing.assert_array_equal(self.ema.apply_scheduled(first), prefix[2])
        np.testing.assert_array_equal(self.ema.apply_scheduled(second), prefix[3])
        postfix = self.scheduler.next_action()
        self.assertFalse(postfix.ema_passthrough)
        self.assertLess(postfix.blend_alpha, 1.0)

    def test_delay_miss_keeps_old_buffer_for_immediate_retry(self):
        for _ in range(5):
            self.scheduler.next_action()
        prefix = self.scheduler.preview(2, self.ema)
        before = self.scheduler.remaining
        info = self.scheduler.adopt(_chunk(2, delay=2, prefix=prefix), actual_elapsed=3)
        self.assertFalse(info.adopted)
        self.assertEqual(info.drop_reason, "delay_miss")
        self.assertEqual(self.scheduler.remaining, before)
        self.assertTrue(self.scheduler.should_request(False, 3))

    def test_execution_horizon_bounds_future_request(self):
        for _ in range(27):
            self.scheduler.next_action()
        self.assertTrue(self.scheduler.can_request(3))
        self.assertFalse(self.scheduler.can_request(4))

    def test_30hz_jittered_responses_match_published_prefix_history(self):
        # Advance to the execution-horizon replan point, then inject several
        # response latencies measured in actual 30 Hz publication ticks.
        for _ in range(5):
            self.ema.apply_scheduled(self.scheduler.next_action())
        for request_id, (delay, elapsed) in enumerate(((4, 2), (5, 5), (6, 3)), start=2):
            prefix = self.scheduler.preview(delay, self.ema)
            published_during_request = [
                self.ema.apply_scheduled(self.scheduler.next_action()) for _ in range(elapsed)
            ]
            response = _chunk(request_id, delay=delay, prefix=prefix, offset=100 * request_id)
            info = self.scheduler.adopt(response, actual_elapsed=elapsed)
            self.assertTrue(info.adopted)
            np.testing.assert_array_equal(
                np.asarray(published_during_request, dtype=np.float32), prefix[:elapsed]
            )
            # Consume to this chunk's execution horizon before its successor.
            while self.scheduler.current_source_index < self.scheduler.replan_steps:
                self.ema.apply_scheduled(self.scheduler.next_action())


if __name__ == "__main__":
    unittest.main()
