# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

from haystack import AsyncPipeline, Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.document_stores.in_memory import InMemoryDocumentStore


async def log_chunk(chunk: StreamingChunk) -> None:
    """Init-time streaming callback; composed alongside the pipeline forwarder."""
    # print(f"[init-cb saw {len(chunk.content)} chars]")


async def happy_path() -> None:
    """RAG over an in-memory BM25 store; streams the LLM answer as it arrives."""
    document_store = InMemoryDocumentStore()
    document_store.write_documents(
        [
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome."),
        ]
    )

    prompt_template = [
        ChatMessage.from_user(
            "Given these documents, answer the question. Lonag answer, at least 3 sentences\n"
            "Documents:\n"
            "{% for doc in documents %}{{ doc.content }}\n{% endfor %}\n"
            "Question: {{ question }}\n"
            "Answer:"
        )
    ]

    pipe = AsyncPipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    pipe.add_component("prompt_builder", ChatPromptBuilder(template=prompt_template))
    pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4.1-nano", streaming_callback=log_chunk))
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm")

    question = "Who lives in Paris?"
    handle = pipe.stream(data={"retriever": {"query": question}, "prompt_builder": {"question": question}})

    print("--- streaming ---")
    async for chunk in handle:
        print(chunk.content, end="", flush=True)

    print("\n\n--- final result ---")
    print(handle.result["llm"]["replies"][0].text)


async def error_path() -> None:
    """Trigger a real OpenAI error mid-call and show that it surfaces through `async for`."""
    pipe = AsyncPipeline()
    pipe.add_component("llm", OpenAIChatGenerator(model="not-a-real-model-xyz"))

    handle = pipe.stream(data={"llm": {"messages": [ChatMessage.from_user("hello")]}})

    print("--- error path ---")
    try:
        async for chunk in handle:
            print(chunk.content, end="", flush=True)
    except Exception as e:
        print(f"caught {type(e).__name__}: {e}")
        print()
    else:
        print("no error raised (unexpected)")

    try:
        _ = handle.result
    except Exception as e:
        print(f"handle.result also raised: {type(e).__name__}")


async def main() -> None:
    """Run both demo paths in sequence."""
    await happy_path()
    print("\n\n")
    await error_path()


if __name__ == "__main__":
    asyncio.run(main())
