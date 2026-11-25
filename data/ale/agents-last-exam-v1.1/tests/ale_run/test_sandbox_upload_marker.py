"""YiCloud upload slot is released only after the sandbox agent starts."""

from unittest.mock import patch

from ale_run.executors.sandbox import _mark_upload_ready


def test_upload_ready_marker_is_opt_in(tmp_path):
    marker = tmp_path / "ready"
    with patch.dict("os.environ", {}, clear=True):
        _mark_upload_ready()
    assert not marker.exists()

    with patch.dict("os.environ", {"ALE_UPLOAD_READY_MARKER": str(marker)}):
        _mark_upload_ready()
    assert marker.is_file()
