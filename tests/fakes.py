'''Test doubles: a scripted, tool-calling chat model and a scripted embedding model.

The suite must run with no API key and no network, so every model call is replayed
from a script. `ScriptedChatModel` supports `bind_tools` and
`with_structured_output`, which is all `create_agent` and the two decision nodes
need. `HashingEmbeddings` does the same job for the SOP index: a real Chroma
collection, built and searched locally.
'''

from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any, Dict, Iterator, List, Optional, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda


class ScriptedChatModel(BaseChatModel):
    '''Replays a list of AIMessages, one per call, in order.

    `calls` records the messages each invocation was given, so a test can assert on
    what the middleware actually assembled.
    '''

    script: List[AIMessage] = []
    cursor: Dict[str, int] = {}
    calls: List[List[BaseMessage]] = []
    label: str = "scripted"

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _next(self, messages: List[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        index = self.cursor.get("i", 0)
        self.cursor["i"] = index + 1
        message = self.script[index] if index < len(self.script) else AIMessage(content="done")
        return message.model_copy(update={"id": str(uuid.uuid4())})

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        text = message.content if isinstance(message.content, str) else ""
        chunk_size = 8
        pieces = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]
        for piece in pieces:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece, id=message.id))
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", id=message.id, tool_calls=message.tool_calls)
            )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        '''Return whatever `structured` was seeded with for this schema.

        Parameters:
        ---------
        schema (Any): the schema the caller asked to bind.
        **kwargs: accepted and ignored, to match the real signature.

        Returns:
        ----------
        runnable (Runnable): whatever `structured` was seeded with for that schema.
        '''

        seeded = getattr(self, "_structured", {})
        value = seeded.get(getattr(schema, "__name__", str(schema)))

        def _respond(_input: Any) -> Any:
            if value is None:
                raise AssertionError(
                    f"No structured output seeded for {getattr(schema, '__name__', schema)}"
                )
            return value() if callable(value) else value

        return RunnableLambda(_respond)


def scripted(*messages: AIMessage, label: str = "scripted") -> ScriptedChatModel:
    return ScriptedChatModel(script=list(messages), cursor={}, calls=[], label=label)


def tool_call(name: str, args: Dict[str, Any], call_id: str) -> Dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def seed_structured(model: ScriptedChatModel, **by_schema_name: Any) -> ScriptedChatModel:
    '''Attach structured-output responses, keyed by schema class name.

    Parameters:
    ---------
    model (ScriptedChatModel): the scripted model to attach responses to.
    **by_schema_name: one response per structured-output schema, keyed by its class name.

    Returns:
    ----------
    model (ScriptedChatModel): the same model, so seeding can be chained.
    '''

    object.__setattr__(model, "_structured", dict(by_schema_name))
    return model


class HashingEmbeddings(Embeddings):
    '''Deterministic bag-of-words embeddings, so a test can build and search a real Chroma collection.

    Words are hashed into a fixed number of buckets and the vector is L2-normalised,
    so two texts sit closer together the more vocabulary they share. It is not a
    semantic model, and that is the point: a test asserting on retrieval order
    should depend on nothing but its own inputs.

    Parameters:
    ---------
    dimensions (int): how many buckets to hash words into.
    '''

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> List[float]:
        buckets = [0.0] * self.dimensions
        word = ""
        for character in (text or "").lower() + " ":
            if character.isalnum():
                word += character
                continue
            if word:
                digest = hashlib.sha256(word.encode("utf-8")).digest()
                buckets[digest[0] % self.dimensions] += 1.0
                word = ""

        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0:
            # An empty document still needs a direction, or cosine distance is undefined.
            buckets[0] = 1.0
            return buckets
        return [value / norm for value in buckets]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)
