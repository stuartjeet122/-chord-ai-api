from __future__ import annotations

import pytest
from pydantic import ValidationError

from music_api.config import Environment, Settings


def test_settings_normalize_api_prefix() -> None:
    assert Settings(api_prefix="api/v1/").api_prefix == "/api/v1"


def test_settings_reject_invalid_port() -> None:
    with pytest.raises(ValidationError):
        Settings(api_port=70000)


def test_docs_default_off_in_production() -> None:
    assert Settings(app_env=Environment.PRODUCTION).should_enable_docs is False
