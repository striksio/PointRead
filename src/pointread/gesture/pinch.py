import time


class PinchDetector:
    """Detects a double-pinch of index fingertip and thumb tip.

    A pinch is index and thumb overlapping past a threshold. Two pinches
    within a time window toggle an active state, like a double click. A
    lone pinch expires after the window and is forgotten.

    update() returns "on", "off", or None. Call it once per frame with the
    two fingertip positions, or None for either when not confidently seen.
    """

    def __init__(self, radius=40, enter_pct=40.0, window=0.5):
        self.r = radius
        self.enter = enter_pct
        self.window = window
        self.was_over = False
        self.armed_at = None
        self.active = False

    def overlap_pct(self, index_tip, thumb_tip):
        x1, y1 = index_tip
        x2, y2 = thumb_tip
        d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return max(0.0, (2 * self.r - d) / (2 * self.r) * 100)

    def update(self, index_tip, thumb_tip):
        if index_tip is None or thumb_tip is None:
            self.was_over = False
            self._expire()
            return None

        pct = self.overlap_pct(index_tip, thumb_tip)
        is_over = pct >= self.enter

        event = None
        if is_over and not self.was_over:
            now = time.time()
            if self.armed_at is not None and now - self.armed_at <= self.window:
                self.armed_at = None
                self.active = not self.active
                event = "on" if self.active else "off"
            else:
                self.armed_at = now
        self.was_over = is_over
        self._expire()
        return event

    def _expire(self):
        if self.armed_at is not None and time.time() - self.armed_at > self.window:
            self.armed_at = None
