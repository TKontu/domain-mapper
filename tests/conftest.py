"""Suite-wide test fixtures."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Restore the root logger's exact handler list and level after each test.

    The ``landscape`` callback calls ``core.log.configure_logging`` on every CLI
    invocation, replacing root handlers and level. Handlers added during the test
    are closed; pre-existing handlers that were removed are re-added.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)
