import cv2


class VideoStreamPair:
    """Continuously reads two video files and loops them when they end."""

    def __init__(self, left_path: str, right_path: str, target_size=None):
        """
        Args:
            left_path: Path to the left video.
            right_path: Path to the right video.
            target_size: Optional (width, height) to resize both frames to.
        """
        self.left_path = left_path
        self.right_path = right_path
        self.target_size = target_size

        self.left_cap = cv2.VideoCapture(self.left_path)
        self.right_cap = cv2.VideoCapture(self.right_path)
        if not self.left_cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.left_path}")
        if not self.right_cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.right_path}")

    def _read_looped(self, cap: cv2.VideoCapture, label: str):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame from {label}")
        return frame

    def read(self):
        """
        Returns:
            Tuple of (left_frame, right_frame) resized to target_size if provided.
        """
        left = self._read_looped(self.left_cap, "left video")
        right = self._read_looped(self.right_cap, "right video")

        if self.target_size:
            left = cv2.resize(left, self.target_size)
            right = cv2.resize(right, self.target_size)
        elif left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))

        return left, right

    def release(self):
        self.left_cap.release()
        self.right_cap.release()
