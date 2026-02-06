"""
MoltbotAgent — Conversational AI with RAG, tool use, and session memory.
=========================================================================

Replaces the stateless /chat endpoint with a persistent, tool-capable agent.
"""

import json

import httpx
from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.config import Settings
from app.logging_config import get_logger
from app.models import Document, ChatSession, ChatMessage

log = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Moltbot, a Senior Systems Architect and Trading Sentinel. "
    "You are concise, technical, and cynical. You prioritize facts over pleasantries. "
    "Answer in Markdown. Use code blocks for technical content. "
    "You have access to tools for searching documents, retrieving stats, and fetching document details. "
    "Use them when the user asks about documents, system status, or stored knowledge."
)

# Tool definitions for Ollama's tool-use API
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search indexed documents using semantic vector similarity. Use when the user asks about documents, knowledge, or information that might be stored.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_stats",
            "description": "Get current system statistics including document count, memory count, and system status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Get full details of a specific document by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "integer",
                        "description": "The document ID",
                    }
                },
                "required": ["doc_id"],
            },
        },
    },
]


class MoltbotAgent:
    """
    Stateful chat agent with RAG, tool use, and session persistence.

    Flow:
        1. Load history from DB (last 10 messages)
        2. RAG: pgvector cosine search -> top 20 -> batch re-rank -> top 5
        3. Build context: system prompt + RAG docs + history
        4. Call Ollama /api/chat with tools array
        5. If tool_calls -> execute tools -> loop back to step 4
        6. Save all messages to DB
        7. Return final response
    """

    MAX_TOOL_ROUNDS = 3
    HISTORY_LIMIT = 10

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ):
        self.session_factory = session_factory
        self.settings = settings

    async def chat(
        self,
        user_message: str,
        session_id: int | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, int | None]:
        """
        Process a user message and return (response_text, tokens_used).
        Creates a new session if session_id is None.
        """
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Get or create session
        async with self.session_factory() as db:
            if session_id:
                result = await db.execute(
                    select(ChatSession).where(ChatSession.id == session_id)
                )
                session = result.scalar_one_or_none()
                if not session:
                    session = ChatSession(title=user_message[:80], system_prompt=system_prompt)
                    db.add(session)
                    await db.commit()
                    await db.refresh(session)
                    session_id = session.id
            else:
                session = ChatSession(title=user_message[:80], system_prompt=system_prompt)
                db.add(session)
                await db.commit()
                await db.refresh(session)
                session_id = session.id

            # Load recent history
            history = await self._load_history(db, session_id)

            # RAG: find relevant context
            rag_context = await self._rag_search(user_message)

            # Build messages for Ollama
            messages = self._build_messages(system_prompt, rag_context, history, user_message)

            # Save user message
            user_msg = ChatMessage(
                session_id=session_id,
                role="user",
                content=user_message,
            )
            db.add(user_msg)
            await db.commit()

        # Call Ollama with tool loop
        response_text, tokens = await self._call_with_tools(messages, session_id)

        # Save assistant response
        async with self.session_factory() as db:
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=response_text,
                tokens_used=tokens,
            )
            db.add(assistant_msg)
            await db.commit()

        return response_text, tokens

    async def _load_history(self, db: AsyncSession, session_id: int) -> list[dict]:
        """Load the last N messages from the session."""
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(self.HISTORY_LIMIT)
        )
        messages = list(reversed(result.scalars().all()))
        return [{"role": m.role, "content": m.content} for m in messages]

    async def _rag_search(self, query: str) -> str:
        """
        Perform RAG: embed query -> pgvector top 20 -> batch re-rank -> top 5.
        Returns formatted context string.
        """
        from app.services import generate_embedding

        embedding = await generate_embedding(query)
        if not embedding:
            return ""

        async with self.session_factory() as db:
            result = await db.execute(
                text("""
                    SELECT id, filename, summary,
                           1 - (embedding <=> :query_embedding) as similarity
                    FROM documents
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> :query_embedding
                    LIMIT 20
                """),
                {"query_embedding": str(embedding)},
            )
            rows = result.fetchall()

        if not rows:
            return ""

        # Batch re-rank: send all summaries to Ollama in one call
        ranked = await self._batch_rerank(query, rows)

        # Format top 5 as context
        context_parts = []
        for row in ranked[:5]:
            summary = row["summary"] or "No summary"
            context_parts.append(f"[Doc #{row['id']}] {row['filename']} (relevance: {row['score']}/10):\n{summary}")

        return "\n\n".join(context_parts)

    async def _batch_rerank(self, query: str, rows: list) -> list[dict]:
        """
        Re-rank documents by sending all summaries to Ollama in one call.
        Returns sorted list of dicts with id, filename, summary, score.
        """
        if not rows:
            return []

        docs_text = "\n".join([
            f"{i+1}. [{row.filename}]: {(row.summary or 'No summary')[:200]}"
            for i, row in enumerate(rows)
        ])

        rerank_prompt = (
            f"Rate each document's relevance to the query \"{query}\" on a scale of 1-10. "
            f"Respond with ONLY a JSON array of scores, e.g. [8, 3, 7, ...]. No other text.\n\n"
            f"Documents:\n{docs_text}"
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.settings.ollama_generate_url,
                    json={
                        "model": self.settings.ollama_llm_model,
                        "prompt": rerank_prompt,
                        "stream": False,
                    },
                )
                if response.status_code == 200:
                    raw = response.json().get("response", "").strip()
                    # Try to extract JSON array from response
                    scores = self._parse_scores(raw, len(rows))
                    ranked = []
                    for i, row in enumerate(rows):
                        ranked.append({
                            "id": row.id,
                            "filename": row.filename,
                            "summary": row.summary,
                            "score": scores[i] if i < len(scores) else 5,
                        })
                    ranked.sort(key=lambda x: x["score"], reverse=True)
                    return ranked
        except Exception as e:
            log.warning("rerank_failed", error=str(e))

        # Fallback: use vector similarity order
        return [
            {"id": row.id, "filename": row.filename, "summary": row.summary, "score": 5}
            for row in rows
        ]

    @staticmethod
    def _parse_scores(raw: str, expected_count: int) -> list[int]:
        """Parse a JSON array of scores from LLM output."""
        try:
            # Find the first [ ... ] in the response
            start = raw.index("[")
            end = raw.index("]", start) + 1
            scores = json.loads(raw[start:end])
            return [max(1, min(10, int(s))) for s in scores]
        except (ValueError, json.JSONDecodeError):
            return [5] * expected_count

    def _build_messages(
        self,
        system_prompt: str,
        rag_context: str,
        history: list[dict],
        user_message: str,
    ) -> list[dict]:
        """Build the messages array for Ollama."""
        messages = [{"role": "system", "content": system_prompt}]

        if rag_context:
            messages.append({
                "role": "system",
                "content": f"Relevant documents from the knowledge base:\n\n{rag_context}",
            })

        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _call_with_tools(
        self,
        messages: list[dict],
        session_id: int,
    ) -> tuple[str, int | None]:
        """
        Call Ollama with tool support. Loops if the model returns tool_calls.
        Returns (response_text, tokens_used).
        """
        tokens_total = 0

        for _ in range(self.MAX_TOOL_ROUNDS):
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.settings.ollama_llm_model,
                    "messages": messages,
                    "stream": False,
                    "tools": TOOLS,
                }
                response = await client.post(
                    self.settings.ollama_chat_url,
                    json=payload,
                )

                if response.status_code != 200:
                    log.error("ollama_chat_error", status=response.status_code)
                    return f"Ollama error: {response.status_code}", None

                data = response.json()
                msg = data.get("message", {})
                tokens_total += data.get("eval_count", 0)

                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    # No tool calls — return the content
                    return msg.get("content", ""), tokens_total or None

                # Process tool calls
                messages.append(msg)  # Add assistant message with tool_calls

                for tool_call in tool_calls:
                    fn = tool_call.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})

                    log.info("tool_call", tool=name, args=args)
                    result = await self._execute_tool(name, args)

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result),
                    })

                    # Save tool interaction to DB
                    async with self.session_factory() as db:
                        tool_msg = ChatMessage(
                            session_id=session_id,
                            role="tool",
                            content=json.dumps(result),
                            tool_calls={"name": name, "args": args},
                        )
                        db.add(tool_msg)
                        await db.commit()

        # If we exhaust tool rounds, return the last content
        return "I've reached the maximum tool interaction limit. Here's what I found so far.", tokens_total or None

    async def _execute_tool(self, name: str, args: dict) -> dict:
        """Execute a tool by name and return the result."""
        if name == "search_documents":
            return await self._tool_search_documents(args.get("query", ""))
        elif name == "get_system_stats":
            return await self._tool_get_system_stats()
        elif name == "get_document":
            return await self._tool_get_document(args.get("doc_id", 0))
        else:
            return {"error": f"Unknown tool: {name}"}

    async def _tool_search_documents(self, query: str) -> dict:
        """Tool: search documents via pgvector."""
        from app.services import generate_embedding

        embedding = await generate_embedding(query)
        if not embedding:
            return {"error": "Embedding service unavailable", "results": []}

        async with self.session_factory() as db:
            result = await db.execute(
                text("""
                    SELECT id, filename, summary,
                           1 - (embedding <=> :query_embedding) as similarity
                    FROM documents
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> :query_embedding
                    LIMIT 5
                """),
                {"query_embedding": str(embedding)},
            )
            rows = result.fetchall()

        return {
            "results": [
                {
                    "id": row.id,
                    "filename": row.filename,
                    "summary": row.summary or "",
                    "similarity": round(float(row.similarity), 3),
                }
                for row in rows
            ]
        }

    async def _tool_get_system_stats(self) -> dict:
        """Tool: get system statistics."""
        async with self.session_factory() as db:
            result = await db.execute(select(func.count(Document.id)))
            doc_count = result.scalar() or 0

            result = await db.execute(
                select(func.count(Document.id)).where(Document.status == "completed")
            )
            completed = result.scalar() or 0

            result = await db.execute(
                select(func.count(Document.id)).where(Document.status == "processing")
            )
            processing = result.scalar() or 0

        return {
            "documents_total": doc_count,
            "documents_completed": completed,
            "documents_processing": processing,
            "status": "ONLINE",
        }

    async def _tool_get_document(self, doc_id: int) -> dict:
        """Tool: get full document details."""
        async with self.session_factory() as db:
            result = await db.execute(
                select(Document).where(Document.id == doc_id)
            )
            doc = result.scalar_one_or_none()

        if not doc:
            return {"error": f"Document {doc_id} not found"}

        return {
            "id": doc.id,
            "filename": doc.filename,
            "status": doc.status,
            "summary": doc.summary or "",
            "content_type": doc.content_type or "",
            "file_size": doc.file_size,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }
