"""The API key a client sends: ``api_key``, or
``$LOOPER_API_KEY`` when it is None, as ``Authorization: Bearer`` on every
request; with neither, no credential."""

import asyncio

import httpx
import pytest

from looper_sdk import ServiceClient
from looper_sdk.proto.v1 import sessions_pb2
from looper_sdk.transport import PROTOBUF


def answer(message) -> httpx.Response:
    return httpx.Response(
        200, content=message.SerializeToString(), headers={"content-type": PROTOBUF}
    )


def sessions(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/sessions":
        session = sessions_pb2.Session(id="s", project="p", heartbeat_interval_s=3600)
        return answer(sessions_pb2.OpenSessionResponse(session=session))
    return answer(sessions_pb2.CloseSessionResponse())


def authorizations(config: ServiceClient.Config) -> list[str | None]:
    """The Authorization header of each request a client sends while it opens
    and closes its session."""
    seen: list[str | None] = []

    def api(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return sessions(request)

    async def run() -> None:
        runtime = ServiceClient.Runtime(transport=httpx.MockTransport(api))
        service = await config.build(runtime)
        await service.close()

    asyncio.run(run())
    return seen


def test_the_key_goes_on_every_request(monkeypatch):
    monkeypatch.setenv("LOOPER_API_KEY", "sk-from-env")
    config = ServiceClient.Config(base_url="http://looper", project="p", api_key="sk-given")
    assert authorizations(config) == ["Bearer sk-given", "Bearer sk-given"]


@pytest.mark.parametrize("base_url", [None, "http://127.0.0.1:8080", "https://other.example"])
def test_the_hosted_default_and_explicit_urls_reach_the_expected_api(base_url):
    seen = []

    def api(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return sessions(request)

    async def run():
        config = (
            ServiceClient.Config(project="p", api_key="sk-given")
            if base_url is None
            else ServiceClient.Config(project="p", api_key="sk-given", base_url=base_url)
        )
        runtime = ServiceClient.Runtime(transport=httpx.MockTransport(api))
        async with await config.build(runtime):
            pass

    asyncio.run(run())
    expected = base_url or "https://api.looper.goaly.ai"
    assert seen == [
        (f"{expected}/v1/sessions", "Bearer sk-given"),
        (f"{expected}/v1/sessions/s:close", "Bearer sk-given"),
    ]


def test_without_one_the_client_reads_looper_api_key(monkeypatch):
    monkeypatch.setenv("LOOPER_API_KEY", "sk-from-env")
    config = ServiceClient.Config(base_url="http://looper", project="p")
    assert authorizations(config) == ["Bearer sk-from-env", "Bearer sk-from-env"]


def test_with_neither_the_client_sends_no_credential(monkeypatch):
    monkeypatch.delenv("LOOPER_API_KEY", raising=False)
    config = ServiceClient.Config(base_url="http://looper", project="p")
    assert authorizations(config) == [None, None]


def test_a_config_never_shows_its_key():
    config = ServiceClient.Config(base_url="http://looper", project="p", api_key="sk-secret")
    assert "sk-secret" not in repr(config)


@pytest.mark.parametrize("key", ["", "sk-a b", "sk-a\nInjected: 1"])
def test_a_key_is_one_token(key):
    with pytest.raises(ValueError, match="api_key"):
        ServiceClient.Config(base_url="http://looper", project="p", api_key=key)


def test_a_key_from_the_environment_is_one_token_too(monkeypatch):
    monkeypatch.setenv("LOOPER_API_KEY", "sk-a b")
    config = ServiceClient.Config(base_url="http://looper", project="p")
    with pytest.raises(ValueError, match="LOOPER_API_KEY"):
        authorizations(config)
