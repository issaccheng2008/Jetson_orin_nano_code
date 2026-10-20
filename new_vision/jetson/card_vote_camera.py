"""Fresh secondary-camera frames for the stopped card vote only."""


class CardVoteCamera:
    def __init__(self, reader):
        self.reader = reader

    def sample(self, not_before_s, timeout_s=0.02):
        """Return a newly captured vote frame, or a reason to abstain.

        The main camera still owns lane tracking, approach detection, QR and
        recording. A frame captured before re-pose DONE + settling is never a
        valid vote even if the image is read from the camera later.
        """
        ok, frame, metadata = self.reader.read(timeout=timeout_s)
        if not ok or frame is None:
            return None, metadata, 'read_unavailable'
        captured = metadata.get('camera_capture_started_s')
        if captured is None or captured < not_before_s:
            return None, metadata, 'captured_before_vote'
        return frame, metadata, 'fresh'

    def close(self):
        return self.reader.close()
