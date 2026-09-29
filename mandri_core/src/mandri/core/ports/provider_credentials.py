"""Provider credential materialization port."""


class CredentialResolverPort:
    """Supplies the live credential for a provider whose key is not stored in config."""

    def access_token(self, provider_name: str) -> str:
        raise NotImplementedError

    async def ensure_fresh(self, provider_name: str) -> None:
        """Rotate the stored credential when it is close to expiry."""
        raise NotImplementedError

    def clear(self, provider_name: str) -> None:
        raise NotImplementedError
