import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI-compatible implementation of the embedding provider.
    Works with any API exposing an OpenAI-compatible `/embeddings` endpoint
    (OpenAI, Azure OpenAI, Ollama, LM Studio, vLLM, etc.).

    :param model_name: The name of the embedding model to use.
    :param api_key: The API key for the embedding endpoint.
    :param base_url: The base URL of the OpenAI-compatible API, e.g. "https://api.openai.com/v1".
    :param vector_size: The dimensionality of the embeddings produced by the model.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        vector_size: int,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.vector_size = vector_size
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=60.0
        )

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = await self._client.post(
            "/embeddings",
            json={"model": self.model_name, "input": inputs},
        )
        response.raise_for_status()
        data = response.json()["data"]
        # Preserve input order as guaranteed by the "index" field
        data.sort(key=lambda item: item["index"])
        embeddings = [item["embedding"] for item in data]
        for embedding in embeddings:
            if len(embedding) != self.vector_size:
                raise ValueError(
                    f"Embedding size mismatch: model '{self.model_name}' returned "
                    f"{len(embedding)} dimensions, but EMBEDDING_VECTOR_SIZE is set "
                    f"to {self.vector_size}. Set EMBEDDING_VECTOR_SIZE to match the "
                    f"model's dimensionality."
                )
        return embeddings

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        return await self._embed(documents)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        embeddings = await self._embed([query])
        return embeddings[0]

    def get_vector_name(self) -> str:
        """Return the name of the vector for the Qdrant collection."""
        model_name = self.model_name.split("/")[-1].lower()
        return f"openai-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self.vector_size
