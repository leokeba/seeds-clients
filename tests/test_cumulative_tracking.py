"""Tests for cumulative tracking in clients."""

import shutil
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from seeds_clients import CumulativeTracking, Message, OpenAIClient


class TestClientCumulativeTracking:
    """Test cumulative tracking integration in OpenAI client."""

    @pytest.fixture
    def cache_dir(self) -> Generator[str, None, None]:
        """Create and cleanup temp cache directory."""
        cache_dir = tempfile.mkdtemp(prefix="test_cumulative_")
        yield cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def client(self, cache_dir: str) -> Generator[OpenAIClient, None, None]:
        """Create test client with caching."""
        client = OpenAIClient(api_key="test-key", cache_dir=cache_dir)
        yield client
        if client.cache:
            client.cache.close()

    @pytest.fixture
    def mock_response(self) -> dict[str, Any]:
        """Create mock API response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    def test_client_has_cumulative_tracking(self, client: OpenAIClient) -> None:
        """Test that client has cumulative tracking initialized."""
        assert hasattr(client, "cumulative_tracking")
        assert hasattr(client, "_cumulative_tracking")
        assert isinstance(client.cumulative_tracking, CumulativeTracking)

    def test_cumulative_tracking_starts_empty(self, client: OpenAIClient) -> None:
        """Test that cumulative tracking starts with zero values."""
        tracking = client.cumulative_tracking
        assert tracking.total_request_count == 0
        assert tracking.api_request_count == 0
        assert tracking.cached_request_count == 0
        assert tracking.total_gwp_kgco2eq == 0.0
        assert tracking.total_cost_usd == 0.0

    @patch("httpx.Client.post")
    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_api_request_accumulates(
        self,
        mock_llm_impacts: MagicMock,
        mock_post: MagicMock,
        client: OpenAIClient,
        mock_response: dict[str, Any],
    ) -> None:
        """Test that API requests accumulate tracking data."""
        # Setup mocks
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value=mock_response),
            raise_for_status=MagicMock(),
        )
        
        # Mock EcoLogits impacts
        mock_impacts = MagicMock()
        mock_impacts.energy = MagicMock(value=0.001)
        mock_impacts.gwp = MagicMock(value=0.0005)
        mock_impacts.adpe = None
        mock_impacts.pe = None
        mock_impacts.usage = MagicMock(
            energy=MagicMock(value=0.0008),
            gwp=MagicMock(value=0.0003),
            adpe=None,
            pe=None,
        )
        mock_impacts.embodied = MagicMock(
            gwp=MagicMock(value=0.0002),
            adpe=None,
            pe=None,
        )
        mock_impacts.warnings = None
        mock_impacts.errors = None
        mock_llm_impacts.return_value = mock_impacts
        
        # Make first request
        response = client.generate(
            [Message(role="user", content="Hello")],
            use_cache=False,
        )
        
        tracking = client.cumulative_tracking
        assert tracking.api_request_count == 1
        assert tracking.cached_request_count == 0
        assert tracking.api_gwp_kgco2eq == pytest.approx(0.0005)
        assert tracking.api_gwp_usage_kgco2eq == pytest.approx(0.0003)
        assert tracking.api_gwp_embodied_kgco2eq == pytest.approx(0.0002)

    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_cached_request_accumulates_separately(
        self,
        mock_llm_impacts: MagicMock,
        client: OpenAIClient,
        mock_response: dict[str, Any],
    ) -> None:
        """Test that cached requests accumulate in cached tracking."""
        # Mock EcoLogits - must return None to avoid pickling issues with cache
        # The cached response will get tracking data from _parse_response instead
        mock_llm_impacts.return_value = None
        
        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = MagicMock(
                json=MagicMock(return_value=mock_response.copy()),
                raise_for_status=MagicMock(),
            )
            
            messages = [Message(role="user", content="Hello")]
            
            # First request - API call
            response1 = client.generate(messages, use_cache=True)
            assert response1.cached is False
            
            # Second request - should be cached
            response2 = client.generate(messages, use_cache=True)
            assert response2.cached is True
            
            tracking = client.cumulative_tracking
            assert tracking.api_request_count == 1
            assert tracking.cached_request_count == 1
            assert tracking.total_request_count == 2
            
            # Note: Without EcoLogits, tracking still has cost from pricing
            # The cost is calculated in _parse_response
            assert tracking.total_cost_usd > 0

    @patch("httpx.Client.post")
    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_multiple_requests_accumulate(
        self,
        mock_llm_impacts: MagicMock,
        mock_post: MagicMock,
        client: OpenAIClient,
        mock_response: dict[str, Any],
    ) -> None:
        """Test that multiple requests accumulate correctly."""
        # Setup mocks
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value=mock_response),
            raise_for_status=MagicMock(),
        )
        
        # Mock EcoLogits impacts
        mock_impacts = MagicMock()
        mock_impacts.energy = MagicMock(value=0.001)
        mock_impacts.gwp = MagicMock(value=0.0005)
        mock_impacts.adpe = None
        mock_impacts.pe = None
        mock_impacts.usage = None
        mock_impacts.embodied = None
        mock_impacts.warnings = None
        mock_impacts.errors = None
        mock_llm_impacts.return_value = mock_impacts
        
        # Make multiple requests
        for i in range(5):
            client.generate(
                [Message(role="user", content=f"Request {i}")],
                use_cache=False,
            )
        
        tracking = client.cumulative_tracking
        assert tracking.api_request_count == 5
        assert tracking.api_gwp_kgco2eq == pytest.approx(0.0025)
        assert tracking.api_prompt_tokens == 50  # 10 per request

    def test_reset_cumulative_tracking(self, client: OpenAIClient) -> None:
        """Test resetting cumulative tracking."""
        # Manually accumulate some data
        client._cumulative_tracking.api_request_count = 10
        client._cumulative_tracking.api_gwp_kgco2eq = 0.005
        client._cumulative_tracking.cached_request_count = 5
        
        # Reset
        client.reset_cumulative_tracking()
        
        tracking = client.cumulative_tracking
        assert tracking.api_request_count == 0
        assert tracking.cached_request_count == 0
        assert tracking.api_gwp_kgco2eq == 0.0

    def test_get_cumulative_tracking_method(self, client: OpenAIClient) -> None:
        """Test get_cumulative_tracking() method returns same object as property."""
        assert client.get_cumulative_tracking() is client.cumulative_tracking
        assert client.get_cumulative_tracking() is client._cumulative_tracking


class TestCumulativeTrackingWithAsyncClient:
    """Test cumulative tracking with async operations."""

    @pytest.fixture
    def cache_dir(self) -> Generator[str, None, None]:
        """Create and cleanup temp cache directory."""
        cache_dir = tempfile.mkdtemp(prefix="test_cumulative_async_")
        yield cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def client(self, cache_dir: str) -> Generator[OpenAIClient, None, None]:
        """Create test client with caching."""
        client = OpenAIClient(api_key="test-key", cache_dir=cache_dir)
        yield client
        if client.cache:
            client.cache.close()

    @pytest.fixture
    def mock_response(self) -> dict[str, Any]:
        """Create mock API response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    async def test_async_api_request_accumulates(
        self,
        mock_llm_impacts: MagicMock,
        mock_post: MagicMock,
        client: OpenAIClient,
        mock_response: dict[str, Any],
    ) -> None:
        """Test that async API requests accumulate tracking data."""
        # Setup mocks
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value=mock_response),
            raise_for_status=MagicMock(),
        )
        
        # Mock EcoLogits impacts
        mock_impacts = MagicMock()
        mock_impacts.energy = MagicMock(value=0.001)
        mock_impacts.gwp = MagicMock(value=0.0005)
        mock_impacts.adpe = None
        mock_impacts.pe = None
        mock_impacts.usage = None
        mock_impacts.embodied = None
        mock_impacts.warnings = None
        mock_impacts.errors = None
        mock_llm_impacts.return_value = mock_impacts
        
        # Make async request
        response = await client.agenerate(
            [Message(role="user", content="Hello")],
            use_cache=False,
        )
        
        tracking = client.cumulative_tracking
        assert tracking.api_request_count == 1
        assert tracking.api_gwp_kgco2eq == pytest.approx(0.0005)

    @pytest.mark.asyncio
    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    async def test_async_cached_request_accumulates(
        self,
        mock_llm_impacts: MagicMock,
        client: OpenAIClient,
        mock_response: dict[str, Any],
    ) -> None:
        """Test that async cached requests accumulate separately."""
        from unittest.mock import AsyncMock
        
        # Mock EcoLogits - must return None to avoid pickling issues with cache
        mock_llm_impacts.return_value = None
        
        # Create async mock for the HTTP client
        mock_async_response = MagicMock()
        mock_async_response.json.return_value = mock_response.copy()
        mock_async_response.raise_for_status = MagicMock()
        
        mock_async_client = MagicMock()
        mock_async_client.post = AsyncMock(return_value=mock_async_response)
        
        with patch.object(client, "_get_async_client", return_value=mock_async_client):
            messages = [Message(role="user", content="Hello async")]
            
            # First async request - API call
            response1 = await client.agenerate(messages, use_cache=True)
            assert response1.cached is False
            
            # Second async request - should be cached
            response2 = await client.agenerate(messages, use_cache=True)
            assert response2.cached is True
            
            tracking = client.cumulative_tracking
            assert tracking.api_request_count == 1
            assert tracking.cached_request_count == 1
            assert tracking.total_request_count == 2


class TestCodeCarbonCumulativeTracking:
    """Test cumulative tracking with CodeCarbon metrics.

    CodeCarbon provides hardware-measured carbon emissions:
    - Tracks usage phase only (no embodied emissions)
    - Includes CPU/GPU/RAM energy breakdown
    """

    @pytest.fixture
    def cache_dir(self) -> Generator[str, None, None]:
        """Create and cleanup temp cache directory."""
        cache_dir = tempfile.mkdtemp(prefix="test_codecarbon_cumulative_")
        yield cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def mock_codecarbon_response(self) -> dict[str, Any]:
        """Create mock API response with CodeCarbon x_carbon_trace."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama3-70b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Model Garden!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
            "x_carbon_trace": {
                "emissions_g_co2": 0.52,  # 0.00052 kg
                "energy_consumed_wh": 1.5,  # 0.0015 kWh
                "cpu_energy_wh": 0.5,
                "gpu_energy_wh": 0.8,
                "ram_energy_wh": 0.2,
                "duration_seconds": 0.5,
                "cpu_power_watts": 85.0,
                "gpu_power_watts": 250.0,
                "ram_power_watts": 15.0,
                "completion_tokens": 8,
                "measured": True,
                "tracking_active": True,
            },
        }

    def test_cumulative_tracking_with_codecarbon_tracking_data(self) -> None:
        """Test CumulativeTracking.accumulate handles CodeCarbon TrackingData correctly."""
        from seeds_clients.core.types import CumulativeTracking, TrackingData

        # Create TrackingData as it would come from CodeCarbon
        # (gwp_usage = total, gwp_embodied = None)
        tracking_data = TrackingData(
            energy_kwh=0.0015,
            gwp_kgco2eq=0.00052,
            cost_usd=0.001,
            prompt_tokens=10,
            completion_tokens=8,
            provider="model_garden",
            model="llama3-70b",
            tracking_method="codecarbon",
            duration_seconds=0.5,
            energy_usage_kwh=0.0015,  # Same as total for CodeCarbon
            gwp_usage_kgco2eq=0.00052,  # Same as total for CodeCarbon
            gwp_embodied_kgco2eq=None,  # CodeCarbon doesn't measure embodied
            cpu_energy_kwh=0.0005,
            gpu_energy_kwh=0.0008,
            ram_energy_kwh=0.0002,
        )

        cumulative = CumulativeTracking()

        # Accumulate API request
        cumulative.accumulate(tracking_data, cached=False)

        # Verify accumulation
        assert cumulative.api_request_count == 1
        assert cumulative.api_energy_kwh == pytest.approx(0.0015)
        assert cumulative.api_gwp_kgco2eq == pytest.approx(0.00052)
        assert cumulative.api_gwp_usage_kgco2eq == pytest.approx(0.00052)
        assert cumulative.api_energy_usage_kwh == pytest.approx(0.0015)
        assert cumulative.api_gwp_embodied_kgco2eq == 0.0  # Should remain 0, not None

    def test_cumulative_tracking_multiple_codecarbon_requests(self) -> None:
        """Test accumulating multiple CodeCarbon requests."""
        from seeds_clients.core.types import CumulativeTracking, TrackingData

        cumulative = CumulativeTracking()

        # Simulate 3 API requests with CodeCarbon tracking
        for i in range(3):
            tracking_data = TrackingData(
                energy_kwh=0.001,
                gwp_kgco2eq=0.0005,
                cost_usd=0.01,
                prompt_tokens=10,
                completion_tokens=5,
                provider="model_garden",
                model="llama3-70b",
                tracking_method="codecarbon",
                duration_seconds=0.5,
                energy_usage_kwh=0.001,
                gwp_usage_kgco2eq=0.0005,
                gwp_embodied_kgco2eq=None,
            )
            cumulative.accumulate(tracking_data, cached=False)

        # Verify accumulated values
        assert cumulative.api_request_count == 3
        assert cumulative.api_energy_kwh == pytest.approx(0.003)
        assert cumulative.api_gwp_kgco2eq == pytest.approx(0.0015)
        assert cumulative.api_gwp_usage_kgco2eq == pytest.approx(0.0015)
        assert cumulative.api_cost_usd == pytest.approx(0.03)
        assert cumulative.api_prompt_tokens == 30
        # Embodied should remain 0 since CodeCarbon doesn't track it
        assert cumulative.api_gwp_embodied_kgco2eq == 0.0

    def test_cumulative_tracking_cached_codecarbon(self) -> None:
        """Test cached CodeCarbon requests are tracked separately."""
        from seeds_clients.core.types import CumulativeTracking, TrackingData

        cumulative = CumulativeTracking()

        tracking_data = TrackingData(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            cost_usd=0.01,
            prompt_tokens=10,
            completion_tokens=5,
            provider="model_garden",
            model="llama3-70b",
            tracking_method="codecarbon",
            duration_seconds=0.5,
            energy_usage_kwh=0.001,
            gwp_usage_kgco2eq=0.0005,
            gwp_embodied_kgco2eq=None,
        )

        # Accumulate as API request
        cumulative.accumulate(tracking_data, cached=False)

        # Accumulate as cached request
        cumulative.accumulate(tracking_data, cached=True)

        # Verify separation
        assert cumulative.api_request_count == 1
        assert cumulative.cached_request_count == 1
        assert cumulative.total_request_count == 2

        # API emissions
        assert cumulative.api_gwp_kgco2eq == pytest.approx(0.0005)
        assert cumulative.api_gwp_usage_kgco2eq == pytest.approx(0.0005)

        # Cached (avoided) emissions
        assert cumulative.cached_gwp_kgco2eq == pytest.approx(0.0005)
        assert cumulative.cached_gwp_usage_kgco2eq == pytest.approx(0.0005)

        # Totals
        assert cumulative.total_gwp_kgco2eq == pytest.approx(0.001)
        assert cumulative.emissions_avoided_kgco2eq == pytest.approx(0.0005)

    def test_mixed_codecarbon_and_ecologits_tracking(self) -> None:
        """Test accumulating mixed CodeCarbon and EcoLogits tracking data."""
        from seeds_clients.core.types import CumulativeTracking, TrackingData

        cumulative = CumulativeTracking()

        # CodeCarbon request (usage only)
        codecarbon_tracking = TrackingData(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            cost_usd=0.01,
            prompt_tokens=10,
            completion_tokens=5,
            provider="model_garden",
            model="llama3-70b",
            tracking_method="codecarbon",
            duration_seconds=0.5,
            energy_usage_kwh=0.001,
            gwp_usage_kgco2eq=0.0005,
            gwp_embodied_kgco2eq=None,
        )

        # EcoLogits request (usage + embodied)
        ecologits_tracking = TrackingData(
            energy_kwh=0.002,
            gwp_kgco2eq=0.001,
            cost_usd=0.02,
            prompt_tokens=20,
            completion_tokens=10,
            provider="openai",
            model="gpt-4.1",
            tracking_method="ecologits",
            duration_seconds=1.0,
            energy_usage_kwh=0.0016,
            gwp_usage_kgco2eq=0.0006,
            gwp_embodied_kgco2eq=0.0004,
        )

        cumulative.accumulate(codecarbon_tracking, cached=False)
        cumulative.accumulate(ecologits_tracking, cached=False)

        # Verify combined totals
        assert cumulative.api_request_count == 2
        assert cumulative.api_energy_kwh == pytest.approx(0.003)
        assert cumulative.api_gwp_kgco2eq == pytest.approx(0.0015)

        # Usage phase: 0.0005 (CodeCarbon) + 0.0006 (EcoLogits) = 0.0011
        assert cumulative.api_gwp_usage_kgco2eq == pytest.approx(0.0011)

        # Embodied: 0 (CodeCarbon) + 0.0004 (EcoLogits) = 0.0004
        assert cumulative.api_gwp_embodied_kgco2eq == pytest.approx(0.0004)


class TestCumulativeTrackingUsageEmbodiedBreakdown:
    """Test usage vs embodied phase breakdown in cumulative tracking."""

    @pytest.fixture
    def cache_dir(self) -> Generator[str, None, None]:
        """Create and cleanup temp cache directory."""
        cache_dir = tempfile.mkdtemp(prefix="test_breakdown_")
        yield cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def client(self, cache_dir: str) -> Generator[OpenAIClient, None, None]:
        """Create test client with caching."""
        client = OpenAIClient(api_key="test-key", cache_dir=cache_dir)
        yield client
        if client.cache:
            client.cache.close()

    @pytest.fixture
    def mock_response(self) -> dict[str, Any]:
        """Create mock API response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    @patch("httpx.Client.post")
    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_usage_embodied_breakdown_accumulated(
        self,
        mock_llm_impacts: MagicMock,
        mock_post: MagicMock,
        client: OpenAIClient,
        mock_response: dict[str, Any],
    ) -> None:
        """Test that usage and embodied phases are accumulated separately."""
        # Setup mocks
        mock_post.return_value = MagicMock(
            json=MagicMock(return_value=mock_response),
            raise_for_status=MagicMock(),
        )
        
        # Mock EcoLogits impacts with usage and embodied breakdown
        mock_impacts = MagicMock()
        mock_impacts.energy = MagicMock(value=0.001)
        mock_impacts.gwp = MagicMock(value=0.0005)
        mock_impacts.adpe = None
        mock_impacts.pe = None
        mock_impacts.usage = MagicMock(
            energy=MagicMock(value=0.0008),
            gwp=MagicMock(value=0.0003),
            adpe=None,
            pe=None,
        )
        mock_impacts.embodied = MagicMock(
            gwp=MagicMock(value=0.0002),
            adpe=None,
            pe=None,
        )
        mock_impacts.warnings = None
        mock_impacts.errors = None
        mock_llm_impacts.return_value = mock_impacts
        
        # Make two requests
        for i in range(2):
            client.generate(
                [Message(role="user", content=f"Request {i}")],
                use_cache=False,
            )
        
        tracking = client.cumulative_tracking
        
        # Check total
        assert tracking.api_gwp_kgco2eq == pytest.approx(0.001)  # 0.0005 * 2
        
        # Check usage phase
        assert tracking.api_gwp_usage_kgco2eq == pytest.approx(0.0006)  # 0.0003 * 2
        assert tracking.api_energy_usage_kwh == pytest.approx(0.0016)  # 0.0008 * 2
        
        # Check embodied phase
        assert tracking.api_gwp_embodied_kgco2eq == pytest.approx(0.0004)  # 0.0002 * 2
        
        # Check totals include both phases
        assert tracking.total_gwp_usage_kgco2eq == pytest.approx(0.0006)
        assert tracking.total_gwp_embodied_kgco2eq == pytest.approx(0.0004)
