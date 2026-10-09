from typing import Any

import litellm
from litellm.llms.openai.openai import OpenAIChatCompletion
from openai import AsyncOpenAI, omit


async def _empty_api_key() -> str:
    return ""


class UnauthenticatedClient(AsyncOpenAI):
    def __init__(self, base_url: str) -> None:
        transport = OpenAIChatCompletion._get_async_http_client()
        self._owns_transport = transport is None or transport is not litellm.aclient_session
        headers: dict[str, Any] = {"Authorization": omit}
        super().__init__(
            api_key=_empty_api_key,
            admin_api_key="",
            base_url=base_url,
            default_headers=headers,
            http_client=transport,
            max_retries=0,
        )

    async def close(self) -> None:
        if self._owns_transport:
            await super().close()


def unauthenticated_client(base_url: str) -> AsyncOpenAI:
    return UnauthenticatedClient(base_url)
