"""Unit tests for QDC transient-error classification and retry wrapper."""

from __future__ import annotations

import json
import socket
import sys
import zipfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qdc_jobs as m


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)


def _sdk_bare(message: str, cause: BaseException | None = None) -> Exception:
    """Mimic the QDC SDK, which flattens failures into a bare Exception."""
    err = Exception(message)
    if cause is not None:
        err.__cause__ = cause
    return err


@pytest.mark.parametrize("code", [403, 429, 500, 502, 503, 504])
def test_retryable_status_codes(code):
    err = _sdk_bare(f"failed with status code {code} and message: boom")
    assert m._matched_retryable_status_code(err) == code


def test_400_not_retryable():
    assert m._matched_retryable_status_code(_sdk_bare("status code 400")) is None


def test_network_errors_via_cause_chain():
    assert m._transient_network_error_name(
        _sdk_bare("x", cause=socket.gaierror("dns"))) == "gaierror"
    assert m._transient_network_error_name(
        _sdk_bare("x", cause=httpx.ConnectError("refused"))) == "ConnectError"


def test_local_oserror_not_transient():
    assert m._transient_network_error_name(
        _sdk_bare("x", cause=FileNotFoundError("nope"))) is None


def test_malformed_response_detected():
    # The exact failure seen in CI run 36085729471 (QCS9075M).
    err = _sdk_bare("x", cause=json.JSONDecodeError("Expecting value", "", 0))
    assert m._is_malformed_response(err)
    assert m._is_malformed_response(zipfile.BadZipFile("truncated"))
    assert not m._is_malformed_response(ValueError("other"))


def test_retry_recovers_from_empty_json_body():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise json.JSONDecodeError("Expecting value", "", 0)
        return "ok"

    assert m._call_with_retry(flaky, "download") == "ok"
    assert len(calls) == 3


def test_retry_gives_up_after_max_attempts():
    calls = []

    def always():
        calls.append(1)
        raise _sdk_bare("status code 503")

    with pytest.raises(Exception, match="503"):
        m._call_with_retry(always, "x")
    assert len(calls) == m.CALL_MAX_RETRIES


def test_permanent_error_fails_fast():
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        m._call_with_retry(bad, "x")
    assert len(calls) == 1


def test_describe_error_never_echoes_message():
    assert "secret" not in m._describe_error(_sdk_bare("status code 500 secret-token"))
    assert "secret" not in m._describe_error(RuntimeError("secret-token"))
