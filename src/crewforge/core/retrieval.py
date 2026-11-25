"""Provider-independent external document ingestion and allowlisted HTTPS retrieval."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from crewforge.core.evidence_verifier import EvidenceRetriever, RetrievedSource
from crewforge.schemas import Evidence, EvidenceReliability, ExecutionEvent
from crewforge.storage.database import Database

_QUERY_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/csv",
        "text/html",
        "application/json",
    }
)


class ReferenceDocument(BaseModel):
    """Operator-supplied reference document or text bundle."""

    doc_id: str = Field(min_length=1)
    title: str = ""
    content: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    reliability: EvidenceReliability = EvidenceReliability.MEDIUM
    retrieved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class RetrievalResult(BaseModel):
    """Explicit external retrieval result capturing failure and provenance states."""

    status: Literal["ok", "unavailable", "blocked", "failed"]
    sources: list[RetrievedSource] = Field(default_factory=list)
    error: str | None = None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_document(doc: ReferenceDocument, max_chunk_chars: int = 2000) -> list[dict[str, Any]]:
    """Split a reference document into immutable excerpt chunks preserving word boundaries.

    Documents under max_chunk_chars are kept whole to preserve complete context.
    Longer documents are chunked along paragraph or sentence boundaries, never slicing words.
    """
    text = doc.content.strip()
    if not text:
        return []
    if len(text) <= max_chunk_chars:
        return [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source_ref": doc.source_ref,
                "excerpt": text,
                "source_location": f"{doc.source_ref}#chars:0-{len(text)}",
                "content_hash": _sha256_text(text),
                "retrieved_at": doc.retrieved_at,
                "reliability": doc.reliability,
            }
        ]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[dict[str, Any]] = []
    current_para_group: list[str] = []
    current_chars = 0
    start_offset = 0

    for para in paragraphs:
        if len(para) > max_chunk_chars:
            if current_para_group:
                combined = "\n\n".join(current_para_group)
                loc = f"{doc.source_ref}#chars:{start_offset}-{start_offset + len(combined)}"
                chunks.append(
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "source_ref": doc.source_ref,
                        "excerpt": combined,
                        "source_location": loc,
                        "content_hash": _sha256_text(combined),
                        "retrieved_at": doc.retrieved_at,
                        "reliability": doc.reliability,
                    }
                )
                start_offset += len(combined) + 2
                current_para_group = []
                current_chars = 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_group: list[str] = []
            sent_chars = 0
            for sent in sentences:
                if sent_chars + len(sent) > max_chunk_chars and sent_group:
                    sent_combined = " ".join(sent_group)
                    loc = (
                        f"{doc.source_ref}#chars:{start_offset}-{start_offset + len(sent_combined)}"
                    )
                    chunks.append(
                        {
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "source_ref": doc.source_ref,
                            "excerpt": sent_combined,
                            "source_location": loc,
                            "content_hash": _sha256_text(sent_combined),
                            "retrieved_at": doc.retrieved_at,
                            "reliability": doc.reliability,
                        }
                    )
                    start_offset += len(sent_combined) + 1
                    sent_group = []
                    sent_chars = 0
                sent_group.append(sent)
                sent_chars += len(sent) + 1
            if sent_group:
                sent_combined = " ".join(sent_group)
                loc = f"{doc.source_ref}#chars:{start_offset}-{start_offset + len(sent_combined)}"
                chunks.append(
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "source_ref": doc.source_ref,
                        "excerpt": sent_combined,
                        "source_location": loc,
                        "content_hash": _sha256_text(sent_combined),
                        "retrieved_at": doc.retrieved_at,
                        "reliability": doc.reliability,
                    }
                )
                start_offset += len(sent_combined) + 2
        elif current_chars + len(para) > max_chunk_chars:
            combined = "\n\n".join(current_para_group)
            loc = f"{doc.source_ref}#chars:{start_offset}-{start_offset + len(combined)}"
            chunks.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source_ref": doc.source_ref,
                    "excerpt": combined,
                    "source_location": loc,
                    "content_hash": _sha256_text(combined),
                    "retrieved_at": doc.retrieved_at,
                    "reliability": doc.reliability,
                }
            )
            start_offset += len(combined) + 2
            current_para_group = [para]
            current_chars = len(para)
        else:
            current_para_group.append(para)
            current_chars += len(para) + 2

    if current_para_group:
        combined = "\n\n".join(current_para_group)
        loc = f"{doc.source_ref}#chars:{start_offset}-{start_offset + len(combined)}"
        chunks.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source_ref": doc.source_ref,
                "excerpt": combined,
                "source_location": loc,
                "content_hash": _sha256_text(combined),
                "retrieved_at": doc.retrieved_at,
                "reliability": doc.reliability,
            }
        )
    return chunks


class LocalDocumentRetriever(EvidenceRetriever):
    """Deterministic operator-supplied document/text bundle retriever."""

    def __init__(self, documents: list[ReferenceDocument] | None = None) -> None:
        self._documents: list[ReferenceDocument] = list(documents or [])
        self._chunks: list[dict[str, Any]] = []
        for doc in self._documents:
            self._chunks.extend(_chunk_document(doc))

    @classmethod
    def from_paths(
        cls,
        paths: list[str | Path],
        *,
        max_file_bytes: int = 524_288,
    ) -> LocalDocumentRetriever:
        """Load local UTF-8 text/markdown/JSON documents safely without shell/code execution."""
        docs: list[ReferenceDocument] = []
        for idx, raw_path in enumerate(paths, start=1):
            p = Path(raw_path).expanduser().resolve()
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Reference document does not exist: '{raw_path}'.")
            data = p.read_bytes()
            if len(data) > max_file_bytes:
                raise ValueError(
                    f"Reference document '{p.name}' exceeds max size ({max_file_bytes} bytes)."
                )
            text = data.decode("utf-8")
            if p.suffix.lower() == ".json":
                try:
                    parsed_json = json.loads(text)
                    text = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                except ValueError:
                    pass
            docs.append(
                ReferenceDocument(
                    doc_id=f"doc_file_{idx}",
                    title=p.name,
                    content=text,
                    source_ref=p.name,
                    reliability=EvidenceReliability.HIGH,
                )
            )
        return cls(docs)

    @property
    def retrieval_available(self) -> bool:
        return bool(self._chunks)

    @property
    def documents(self) -> list[ReferenceDocument]:
        return list(self._documents)

    async def retrieve(
        self,
        query: str,
        *,
        run_id: str,
        limit: int = 3,
    ) -> list[RetrievedSource]:
        if not self._chunks:
            return []
        q_tokens = set(_QUERY_WORD_RE.findall(query.lower()))
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, chunk in enumerate(self._chunks):
            c_tokens = set(_QUERY_WORD_RE.findall(chunk["excerpt"].lower()))
            overlap = len(q_tokens & c_tokens) if q_tokens else 1
            scored.append((overlap, -idx, chunk))
        scored.sort(reverse=True)
        selected = [c for ov, _, c in scored[:limit] if (ov > 0 or not q_tokens)]
        if not selected and self._chunks:
            selected = [self._chunks[0]]
        return [
            RetrievedSource(
                source_ref=c["source_ref"],
                title=c["title"],
                excerpt=c["excerpt"],
                retrieved_at=c["retrieved_at"],
                source_location=c["source_location"],
                content_hash=c["content_hash"],
                reliability=c["reliability"],
            )
            for c in selected
        ]

    async def ingest_into_ledger(
        self,
        *,
        database: Database,
        run_id: str,
        actor: str = "lead",
        limit: int = 12,
    ) -> list[Evidence]:
        """Persist verbatim operator-document chunks as immutable verified Evidence."""
        persisted: list[Evidence] = []
        for chunk in self._chunks[:limit]:
            evd = Evidence(
                run_id=run_id,
                source_type="external_retrieved",
                source_ref=chunk["source_ref"],
                excerpt=chunk["excerpt"],
                reliability=chunk["reliability"],
                verified_in_input=True,
                content_hash=chunk["content_hash"],
                retrieved_at=chunk["retrieved_at"],
                source_location=chunk["source_location"],
                retrieval_status="ok",
                created_by=actor,
            )
            await database.insert_evidence(evd)
            persisted.append(evd)
        if persisted:
            await database.append_event(
                ExecutionEvent(
                    run_id=run_id,
                    event_type="documents_ingested",
                    actor=actor,
                    message=f"Ingested {len(persisted)} verified reference excerpt(s).",
                    payload={
                        "evidence_ids": [e.evidence_id for e in persisted],
                        "sources": list(dict.fromkeys(e.source_ref for e in persisted)),
                    },
                )
            )
        return persisted


def _is_public_ip(ip_str: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return bool(
        ip_obj.is_global
        and not ip_obj.is_private
        and not ip_obj.is_loopback
        and not ip_obj.is_link_local
        and not ip_obj.is_multicast
        and not ip_obj.is_reserved
        and not ip_obj.is_unspecified
    )


def validate_allowlisted_https_url(
    url: str,
    *,
    allowed_hosts: list[str],
    resolver: Any | None = None,
) -> tuple[bool, str | None]:
    """Validate destination scheme, exact host allowlist, and non-private DNS resolution.

    Limitations: An initial DNS resolution before opening an HTTP connection verifies current
    public IP allocation, but does not completely eliminate time-of-check-to-time-of-use DNS
    rebinding without socket-level IP binding. For this reason, HTTP retrieval remains disabled
    by default (enable_http_retrieval=False) and fails closed for untrusted destinations.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() != "https":
        return False, "Only HTTPS URLs are permitted for external retrieval."
    if parsed.username or parsed.password:
        return False, "Credentials in retrieval URLs are forbidden."
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return False, "Missing destination hostname."
    normalized_allowlist = {h.strip().lower().rstrip(".") for h in allowed_hosts if h.strip()}
    if not normalized_allowlist or host not in normalized_allowlist:
        return False, f"Destination host '{host}' is not in explicit retrieval allowlist."

    # Disallow raw IP literals in hostnames to preserve hostname/TLS identity checks.
    try:
        ipaddress.ip_address(host)
        return False, "Direct IP-literal URLs are disallowed."
    except ValueError:
        pass

    dns_fn = resolver or socket.getaddrinfo
    try:
        addr_infos = dns_fn(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        return False, f"DNS resolution failed for '{host}': {exc}"

    ips = [info[4][0] for info in addr_infos if info and len(info) >= 5 and info[4]]
    if not ips:
        return False, f"DNS resolution returned no addresses for '{host}'."
    for ip_addr in ips:
        if not _is_public_ip(ip_addr):
            return False, f"Blocked non-public/SSRF address '{ip_addr}' for host '{host}'."
    return True, None


class AllowlistedHttpRetriever(EvidenceRetriever):
    """Disabled-by-default HTTPS retriever enforcing SSRF, redirect, size, and host caps."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        allowed_hosts: list[str] | None = None,
        timeout_seconds: float = 5.0,
        max_bytes: int = 262_144,
        transport: httpx.AsyncBaseTransport | None = None,
        dns_resolver: Any | None = None,
    ) -> None:
        self._enabled = enabled
        self._allowed_hosts = [h.strip().lower() for h in (allowed_hosts or []) if h.strip()]
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes
        self._transport = transport
        self._dns_resolver = dns_resolver

    @property
    def retrieval_available(self) -> bool:
        return self._enabled and bool(self._allowed_hosts)

    async def fetch_url(self, url: str) -> RetrievalResult:
        """Fetch one allowlisted HTTPS document cleanly without redirects or credential leaks."""
        if not self.retrieval_available:
            return RetrievalResult(
                status="unavailable",
                error="HTTP external retrieval is disabled or no allowed hosts are configured.",
            )

        ok, err = validate_allowlisted_https_url(
            url, allowed_hosts=self._allowed_hosts, resolver=self._dns_resolver
        )
        if not ok:
            return RetrievalResult(status="blocked", error=err)

        headers = {
            "User-Agent": "CrewForge-EvidenceRetriever/1.0",
            "Accept": "text/plain, text/markdown, application/json, text/html",
        }
        try:
            async with (
                httpx.AsyncClient(
                    follow_redirects=False,
                    timeout=httpx.Timeout(self._timeout_seconds),
                    headers=headers,
                    transport=self._transport,
                ) as client,
                client.stream("GET", url) as resp,
            ):
                if 300 <= resp.status_code < 400:
                    return RetrievalResult(
                        status="blocked",
                        error="HTTP redirects are disallowed for SSRF protection.",
                    )
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and ctype not in _ALLOWED_CONTENT_TYPES:
                    return RetrievalResult(
                        status="blocked",
                        error=f"Disallowed content type '{ctype}'.",
                    )
                chunks: list[bytes] = []
                total = 0
                async for part in resp.aiter_bytes():
                    total += len(part)
                    if total > self._max_bytes:
                        return RetrievalResult(
                            status="failed",
                            error=f"Response exceeded max size limit ({self._max_bytes} bytes).",
                        )
                    chunks.append(part)
            body = b"".join(chunks).decode("utf-8", errors="replace").strip()
            if not body:
                return RetrievalResult(status="failed", error="Empty response body.")
            excerpt = body[:800]
            now = datetime.now(UTC).isoformat()
            return RetrievalResult(
                status="ok",
                sources=[
                    RetrievedSource(
                        source_ref=urlparse(url).hostname or url,
                        title=urlparse(url).path or "/",
                        excerpt=excerpt,
                        retrieved_at=now,
                        source_location=f"{url}#chars:0-{len(excerpt)}",
                        content_hash=_sha256_text(excerpt),
                        reliability=EvidenceReliability.MEDIUM,
                    )
                ],
            )
        except httpx.HTTPError as exc:
            return RetrievalResult(status="failed", error=f"HTTP retrieval error: {exc}")

    async def retrieve(
        self,
        query: str,
        *,
        run_id: str,
        limit: int = 3,
    ) -> list[RetrievedSource]:
        return []
