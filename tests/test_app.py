import pytest

from prediction.app import health


pytestmark = pytest.mark.asyncio  # @pytest.mark.asyncio on all tests


async def test_health():
    result = await health()
    assert result == {"status": "ok"}
