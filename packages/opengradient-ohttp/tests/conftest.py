from __future__ import annotations

import pytest
from _fakes import FakeGateway


@pytest.fixture(scope="session")
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture(scope="session")
def other_gateway() -> FakeGateway:
    return FakeGateway(endpoint="https://gw-2.test")
