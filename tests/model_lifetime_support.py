"""Controlled HTTP resources for public model-lifetime tests; no network contact."""

import asyncio

import httpx

from bio_neb.agentic_journey.adapters import create_nebius_model_endpoint
from tests.model_endpoint_fakes import ScriptedModelEndpoint


class LoopTransport(httpx.AsyncBaseTransport):
    """Release must await work on the loop which acquired and used the resource."""

    def __init__(self, handle=None, *, close_error=None):
        self.loop = asyncio.get_running_loop()
        self.handle = handle
        self.close_error = close_error
        self.requests = 0
        self.close_started = 0
        self.closed = False

    async def handle_async_request(self, request):
        assert asyncio.get_running_loop() is self.loop
        assert not self.closed
        self.requests += 1
        if self.handle is None:
            return httpx.Response(200, text="controlled model transport")
        response = await self.handle(request)
        # A MockTransport response is unread; the client expects a readable stream.
        await response.aread()
        return response

    async def aclose(self):
        assert asyncio.get_running_loop() is self.loop
        self.close_started += 1
        finished = self.loop.create_future()
        self.loop.call_soon(finished.set_result, None)
        await finished
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class OwnedModelEndpoint(ScriptedModelEndpoint):
    """A scripted endpoint whose transport has the production resource obligations.

    Each Adapter it builds gets its own loop-bound transport; `transport` is the
    one Adapter's.
    """

    def __init__(
        self, respond, *, close_error=None, adapter=create_nebius_model_endpoint
    ):
        super().__init__(respond, adapter=adapter)
        self.close_error = close_error
        self.transports: list[LoopTransport] = []

    def open_transport(self):
        transport = LoopTransport(self.handle, close_error=self.close_error)
        self.transports.append(transport)
        return transport

    @property
    def transport(self) -> LoopTransport:
        (only,) = self.transports
        return only
