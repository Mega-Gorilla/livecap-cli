"""FileTranscriptionPipeline の作業ディレクトリの作成 (Issue #506)。

作業ディレクトリは、今までどおりシステムの一時ディレクトリ (``tempfile.gettempdir()``) に作る。
``tempfile.mkdtemp`` は Windows で、ACL で作成を拒否されたディレクトリでは返らないので、
上限つきの :func:`livecap_cli.resources.bounded_tempfile.mkdtemp` を使う。ここでは呼び出し側の
規則を固定する。実際に ACL で拒否したディレクトリでの確認は
``tests/core/resources/test_bounded_tempfile.py`` にある。
"""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path
from typing import List

import pytest

from livecap_cli.resources import bounded_tempfile
from livecap_cli.transcription.file_pipeline import FileTranscriptionPipeline
from tests.core.resources.acl_support import shadow_module


def test_temp_root_is_made_in_the_system_temp_dir_and_removed_on_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    pipeline = FileTranscriptionPipeline()
    temp_root = pipeline._temp_root
    try:
        assert temp_root.parent == tmp_path
        assert temp_root.name.startswith("livecap-file-pipeline-")
        assert temp_root.is_dir() and list(temp_root.iterdir()) == []
    finally:
        pipeline.close()

    assert not temp_root.exists()


def test_denied_creation_fails_on_the_first_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """作成を拒否されたら (``os.mkdir`` が ``PermissionError``)、やり直さずに 1 回で送出する。

    ``tempfile.mkdtemp`` は Windows でこれを名前の衝突とみなし、約 21 億回まで再試行していた。
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    calls: List[str] = []
    denied = PermissionError(errno.EACCES, "Access is denied")

    def mkdir(path, *args):
        calls.append(str(path))
        raise denied

    shadow_module(monkeypatch, bounded_tempfile, "os", mkdir=mkdir)

    with pytest.raises(PermissionError) as excinfo:
        FileTranscriptionPipeline()

    assert excinfo.value is denied
    assert len(calls) == 1
    assert Path(calls[0]).parent == tmp_path
    assert os.listdir(tmp_path) == []
