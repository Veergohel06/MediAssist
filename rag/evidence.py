"""Evidence sufficiency, source validation, and structured citation auditing."""

import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

CITATION_ID_PATTERN = re.compile(r"\b(?:[a-z0-9_-]+:p\d+|[a-z0-9_-]+_\d+)\b", re.IGNORECASE)
BRACKETED_METADATA_PATTERN = re.compile(
    r"\[\s*([^\]]*?(?:[a-z0-9_-]+:p\d+|[a-z0-9_-]+_\d+)[^\]]*?)\s*\]",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)
VALID_STATUSES = {"sufficient", "insufficient", "conflicting", "stale"}
OVERCONFIDENCE_PATTERNS = (
    r"\byou definitely have\b",
    r"\bthis proves\b",
    r"\bguaranteed to cure\b",
    r"\bwill cure\b",
)


class MedicalResponseContract(BaseModel):
    """Pydantic schema for LangChain structured output validation of clinical responses."""

    answer: str = Field(description="Rich markdown medical explanation or clinical response")
    cited_passage_ids: List[str] = Field(
        default_factory=list,
        description="List of verified passage identifiers substantiating the answer",
    )
    evidence_sufficiency: Literal["sufficient", "insufficient", "conflicting", "stale"] = Field(
        default="insufficient",
        description="Audit status of evidence sufficiency",
    )
    practice_mode: Optional[str] = Field(
        default="general_tutoring",
        description="Active pedagogical mode: general_tutoring, viva_voce, mcq_practice, clinical_case",
    )
    interactive_mcq: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured interactive multiple choice practice item",
    )
    clinical_case: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured progressive clinical case details",
    )


MEDICAL_RESPONSE_PARSER = PydanticOutputParser(pydantic_object=MedicalResponseContract)


def is_valid_source_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_stale(chunk: Dict[str, Any], max_age_days: int) -> bool:
    retrieval_date = str(chunk.get("retrieval_date", "")).strip()
    if not retrieval_date:
        return False
    try:
        parsed = datetime.strptime(retrieval_date, "%Y-%m-%d").date()
    except ValueError:
        return True
    return parsed < date.today() - timedelta(days=max_age_days)


def assess_evidence(chunks: Sequence[Dict[str, Any]], max_age_days: int = 730) -> Dict[str, Any]:
    """Assess whether retrieved passages are usable evidence."""
    valid_chunks = []
    invalid_source_ids = []
    seen_ids = set()
    stale_ids = []
    conflict_groups: Dict[str, set[str]] = {}

    for chunk in chunks:
        passage_id = str(chunk.get("passage_id") or chunk.get("chunk_id") or "")
        if not passage_id or passage_id in seen_ids:
            continue
        seen_ids.add(passage_id)
        is_uploaded = bool(chunk.get("is_uploaded_document") or chunk.get("source_type") == "uploaded_study_note")
        if not is_uploaded and not is_valid_source_url(str(chunk.get("source_url", ""))):
            invalid_source_ids.append(passage_id)
            continue
        valid_chunks.append(chunk)
        if _is_stale(chunk, max_age_days):
            stale_ids.append(passage_id)
        group = chunk.get("conflict_group")
        stance = chunk.get("stance")
        if group and stance:
            conflict_groups.setdefault(str(group), set()).add(str(stance))

    conflicting_groups = sorted(group for group, stances in conflict_groups.items() if len(stances) > 1)
    if not valid_chunks:
        status = "insufficient"
    elif conflicting_groups:
        status = "conflicting"
    elif stale_ids:
        status = "stale"
    else:
        status = "sufficient"

    return {
        "status": status,
        "is_sufficient": status == "sufficient",
        "retrieved_count": len(chunks),
        "valid_source_count": len(valid_chunks),
        "invalid_source_ids": invalid_source_ids,
        "stale_ids": stale_ids,
        "conflicting_groups": conflicting_groups,
        "note": "Evidence status is an audit signal, not a measure of clinical truth.",
    }


def audit_citations(answer: str, chunks: Sequence[Dict[str, Any]], cited_ids: Iterable[str] = ()) -> Dict[str, Any]:
    """Validate passage IDs, URLs, and source provenance against retrieved chunks."""
    valid_ids = {
        str(chunk.get("passage_id") or chunk.get("chunk_id") or "").lower()
        for chunk in chunks
    }
    valid_urls = {
        str(chunk.get("source_url", "")).strip()
        for chunk in chunks
        if is_valid_source_url(str(chunk.get("source_url", "")))
    }
    inline_ids = set(CITATION_ID_PATTERN.findall(answer or ""))
    structured_ids = {str(item) for item in cited_ids if item}
    cited = {item.lower() for item in inline_ids | structured_ids}
    answer_urls = set(URL_PATTERN.findall(answer or ""))
    valid_citations = sorted(cited & valid_ids)
    invalid_citations = sorted(cited - valid_ids)
    unknown_urls = sorted(answer_urls - valid_urls)
    confidence_flags = [pattern for pattern in OVERCONFIDENCE_PATTERNS if re.search(pattern, answer or "", re.IGNORECASE)]
    return {
        "cited_passage_ids": sorted(cited),
        "valid_citation_ids": valid_citations,
        "invalid_citation_ids": invalid_citations,
        "cited_urls": sorted(answer_urls),
        "unknown_urls": unknown_urls,
        "unsupported_confidence_flags": confidence_flags,
        "all_citations_valid": not invalid_citations and not unknown_urls,
        "has_citation": bool(cited or answer_urls),
    }


def parse_structured_response(raw_text: str) -> Dict[str, Any]:
    """Parse the model JSON contract with LangChain PydanticOutputParser, conservatively falling back to plain text."""
    if not isinstance(raw_text, str):
        raw_text = str(raw_text) if raw_text is not None else ""
    raw_text = raw_text.strip()
    cleaned = raw_text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        parsed_contract: MedicalResponseContract = MEDICAL_RESPONSE_PARSER.parse(cleaned)
        return {
            "answer": parsed_contract.answer,
            "cited_passage_ids": parsed_contract.cited_passage_ids,
            "evidence_sufficiency": parsed_contract.evidence_sufficiency,
            "practice_mode": parsed_contract.practice_mode or "general_tutoring",
            "interactive_mcq": parsed_contract.interactive_mcq,
            "clinical_case": parsed_contract.clinical_case,
            "structured_output_valid": True,
        }
    except (OutputParserException, ValueError, TypeError, json.JSONDecodeError):
        # Conservative direct JSON fallback
        try:
            raw_obj = json.loads(cleaned)
            if isinstance(raw_obj, dict) and "answer" in raw_obj:
                ev_suff = str(raw_obj.get("evidence_sufficiency") or "insufficient")
                valid_suff = ev_suff in VALID_STATUSES
                return {
                    "answer": str(raw_obj.get("answer", "")),
                    "cited_passage_ids": list(raw_obj.get("cited_passage_ids") or []),
                    "evidence_sufficiency": ev_suff if valid_suff else "insufficient",
                    "practice_mode": str(raw_obj.get("practice_mode") or "general_tutoring"),
                    "interactive_mcq": raw_obj.get("interactive_mcq"),
                    "clinical_case": raw_obj.get("clinical_case"),
                    "structured_output_valid": valid_suff,
                }
        except Exception:
            pass
        return {
            "answer": raw_text,
            "cited_passage_ids": [],
            "evidence_sufficiency": "insufficient",
            "practice_mode": "general_tutoring",
            "interactive_mcq": None,
            "clinical_case": None,
            "structured_output_valid": False,
        }


def sanitize_and_render_citations(answer: str, chunks: Sequence[Dict[str, Any]] = ()) -> str:
    """
    Remove or replace raw internal citation identifiers, document chunk IDs, page tokens,
    or retrieval metadata (e.g., [sleep_hygiene_and_disorders:p0003], [doc:p0001]) with
    clean, user-friendly source attributions or clickable markdown links.

    - Resolves citations against verified retrieved chunks using source titles and URLs.
    - If the source title is already mentioned in nearby context, omits the redundant token.
    - If a citation cannot be resolved or verified, omits the broken citation rather than
      displaying raw metadata.
    - Never invents citations, page numbers, links, or references.
    - Sweeps the entire text for unrendered citation tokens and cleans formatting/punctuation.
    """
    if not answer:
        return ""

    # Map verified chunks by normalized IDs
    chunks_by_id: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        pid = str(chunk.get("passage_id") or "").strip().lower()
        cid = str(chunk.get("chunk_id") or "").strip().lower()
        if pid:
            chunks_by_id[pid] = chunk
        if cid:
            chunks_by_id[cid] = chunk

    def _replace_bracket(match: re.Match) -> str:
        content = match.group(1)
        found_ids = CITATION_ID_PATTERN.findall(content)
        if not found_ids:
            return match.group(0)

        # Context lookback to check if source was already attributed in preceding text
        start_idx = match.start()
        last_para = answer.rfind("\n\n", 0, start_idx)
        para_start = last_para + 2 if last_para != -1 else 0
        lookback_start = min(para_start, max(0, start_idx - 600))
        lookback = answer[lookback_start:start_idx].lower()

        rendered_attributions = []
        seen_titles = set()

        for raw_id in found_ids:
            chunk = chunks_by_id.get(raw_id.lower())
            if not chunk:
                # Omit unresolvable / invalid citation tokens
                continue

            title = str(chunk.get("source_title", "")).strip()
            url = str(chunk.get("source_url", "")).strip()
            page_num = chunk.get("page_number")
            if page_num and "page" not in title.lower() and "p." not in title.lower():
                title = f"{title} (p. {page_num})"
            if not title:
                continue

            # Check if title (or major title prefix) is already mentioned in nearby context
            title_lower = title.lower()
            colon_part = title_lower.split(":")[0].strip()
            dash_part = title_lower.split(" - ")[0].strip()
            already_mentioned = (
                title_lower in lookback
                or (len(colon_part) >= 8 and colon_part in lookback)
                or (len(dash_part) >= 8 and dash_part in lookback)
            )
            if already_mentioned:
                continue

            if title in seen_titles:
                continue
            seen_titles.add(title)

            if url and is_valid_source_url(url):
                rendered_attributions.append(f"[{title}]({url})")
            else:
                rendered_attributions.append(title)

        if not rendered_attributions:
            return ""

        return " (" + "; ".join(rendered_attributions) + ")"

    # 1. Process bracketed citations
    cleaned = BRACKETED_METADATA_PATTERN.sub(_replace_bracket, answer)

    # 2. Safety sweep: resolve any stray unbracketed citation IDs to verified title/link or strip if unresolvable
    def _replace_unbracketed(match: re.Match) -> str:
        raw_id = match.group(0).lower()
        chunk = chunks_by_id.get(raw_id)
        if chunk:
            title = str(chunk.get("source_title", "")).strip()
            url = str(chunk.get("source_url", "")).strip()
            if title:
                start_idx = match.start()
                lookback = cleaned[max(0, start_idx - 300):start_idx].lower()
                if title.lower() in lookback:
                    return ""
                if url and is_valid_source_url(url):
                    return f"[{title}]({url})"
                return title
        return ""

    cleaned = CITATION_ID_PATTERN.sub(_replace_unbracketed, cleaned)

    # 3. Clean up formatting artifacts: empty brackets, empty parens, punctuation spacing
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\(\s*\((.*?)\)\s*\)", r"(\1)", cleaned)

    # Spacing cleanup before punctuation: e.g. "disorder ." -> "disorder."
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)

    # Remove extra spaces
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Remove whitespace at end of lines
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    # Remove leading spaces on newlines
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)

    return cleaned.strip()
