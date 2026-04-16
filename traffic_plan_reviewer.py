
import argparse
import json
import os
import re
from typing import Any, TypedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from pdf_markdown_extractor import pdf_to_documents




class ReviewInputError(ValueError):
    pass


class ReviewFinding(BaseModel):
    category: str = Field(default="general")
    severity: str = Field(default="medium")
    title: str
    issue: str
    affected_documents: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    guideline_basis: list[str] = Field(default_factory=list)
    recommendation: str
    confidence: str = Field(default="medium")


class ReviewReportModel(BaseModel):
    overview: str = Field(default="")
    assumptions: list[str] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ReviewSubReportModel(BaseModel):
    summary: str = Field(default="")
    assumptions: list[str] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class TrafficReviewState(TypedDict):
    project_context: str
    reference_context: str
    additional_focus_text: str
    standards_review: dict[str, Any]
    common_review: dict[str, Any]
    final_report: dict[str, Any]


def _model_response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(str(item) for item in content)
    return str(content)


def _normalize_path(path: str | None) -> str:
    return os.path.abspath(path or "").replace("\\", "/").lower()


def _chunk_key(doc: Document) -> tuple[str, Any, Any]:
    metadata = doc.metadata or {}
    return (
        _normalize_path(metadata.get("source", "")),
        metadata.get("page"),
        metadata.get("block_id"),
    )


def _severity_rank(value: str) -> int:
    ranks = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return ranks.get((value or "").lower(), 4)


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object")
    return json.loads(text[start:end + 1])


def _strip_thinking_and_fences(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_candidates(text: str) -> list[str]:
    cleaned = _strip_thinking_and_fences(text)
    candidates: list[str] = []

    for fenced in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL):
        candidates.append(fenced.strip())

    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start:end + 1].strip())

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, offset = decoder.raw_decode(cleaned[match.start():])
            if isinstance(parsed, dict):
                candidates.append(cleaned[match.start():match.start() + offset].strip())
                break
        except Exception:
            continue

    unique: list[str] = []
    seen = set()
    for candidate in candidates:
        key = candidate.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _default_structured_response(parser: JsonOutputParser, raw_text: str) -> dict[str, Any]:
    model = getattr(parser, "pydantic_object", None)
    cleaned = _strip_thinking_and_fences(raw_text)
    summary_text = cleaned[:3000].strip() or "Model returned an unstructured response that could not be parsed as JSON."

    if model is ReviewSubReportModel:
        return {
            "summary": summary_text,
            "assumptions": ["The review model returned non-JSON output; findings below may be incomplete."],
            "findings": [],
            "recommendations": [],
        }

    return {
        "overview": summary_text,
        "assumptions": ["The review model returned non-JSON output; the structured report was recovered in fallback mode."],
        "findings": [],
        "recommendations": [],
    }


def _repair_llm_json_response(llm: Any, raw_text: str, parser: JsonOutputParser) -> dict[str, Any] | None:
    repair_prompt = f"""
Convert the following model output into a valid JSON object.

Rules:
- Return JSON only.
- Do not include markdown fences.
- Preserve the meaning of the original output.
- If details are missing, use empty arrays and concise strings.

Required schema:
{parser.get_format_instructions()}

Model output to convert:
{raw_text}
"""

    try:
        repair_response = llm.invoke(repair_prompt)
        repaired_text = _model_response_to_text(repair_response)
        repaired_clean = _strip_thinking_and_fences(repaired_text)
        parsed = parser.parse(repaired_clean)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, BaseModel):
            return parsed.model_dump()
        return dict(parsed)
    except Exception:
        for candidate in _extract_json_candidates(raw_text):
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None


def _parse_llm_json_response(llm: Any, prompt: str, parser: JsonOutputParser) -> dict[str, Any]:
    response = llm.invoke(prompt)
    raw_text = _model_response_to_text(response)
    cleaned_text = _strip_thinking_and_fences(raw_text)
    try:
        parsed = parser.parse(cleaned_text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, BaseModel):
            return parsed.model_dump()
        return dict(parsed)
    except Exception:
        for candidate in _extract_json_candidates(raw_text):
            try:
                return json.loads(candidate)
            except Exception:
                continue

        repaired = _repair_llm_json_response(llm, cleaned_text[:12000], parser)
        if repaired is not None:
            return repaired

        return _default_structured_response(parser, raw_text)


def _query_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2]


def _lexical_score(text: str, query: str) -> int:
    lowered = text.lower()
    score = 0
    for term in _query_terms(query):
        if term in lowered:
            score += lowered.count(term)
    return score


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    unique = []
    seen = set()
    for doc in documents:
        key = _chunk_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _select_review_chunks(
    documents: list[Document],
    queries: list[str],
    reranker: Any = None,
    per_query_limit: int = 4,
    total_limit: int = 14,
) -> list[Document]:
    if not documents:
        return []

    selected: list[Document] = []
    seen = set()

    for query in queries:
        ranked = sorted(
            documents,
            key=lambda doc: (
                _lexical_score(doc.page_content, query),
                len(doc.page_content or ""),
            ),
            reverse=True,
        )
        candidates = [doc for doc in ranked[:40] if _lexical_score(doc.page_content, query) > 0]
        if not candidates:
            candidates = ranked[: min(25, len(ranked))]

        if reranker and candidates:
            pairs = [[query, (doc.page_content or "")[:2500]] for doc in candidates]
            scores = reranker.predict(pairs)
            ordered = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
            candidates = [doc for doc, _ in ordered[:per_query_limit]]
        else:
            candidates = candidates[:per_query_limit]

        for doc in candidates:
            key = _chunk_key(doc)
            if key in seen:
                continue
            seen.add(key)
            selected.append(doc)
            if len(selected) >= total_limit:
                return selected

    return selected


def _retrieve_reference_chunks(
    vectorstore: Any,
    queries: list[str],
    reranker: Any = None,
    per_query_fetch: int = 10,
    per_query_limit: int = 4,
    total_limit: int = 16,
) -> list[Document]:
    if not vectorstore:
        return []

    selected: list[Document] = []
    seen = set()

    for query in queries:
        matches = vectorstore.similarity_search(query, k=per_query_fetch)
        if reranker and matches:
            pairs = [[query, (doc.page_content or "")[:2500]] for doc in matches]
            scores = reranker.predict(pairs)
            ordered = sorted(zip(matches, scores), key=lambda item: float(item[1]), reverse=True)
            matches = [doc for doc, _ in ordered[:per_query_limit]]
        else:
            matches = matches[:per_query_limit]

        for doc in matches:
            key = _chunk_key(doc)
            if key in seen:
                continue
            seen.add(key)
            selected.append(doc)
            if len(selected) >= total_limit:
                return selected

    return selected


def _format_chunks(label: str, documents: list[Document], excerpt_limit: int = 1200) -> str:
    if not documents:
        return f"[{label}] No evidence extracted."

    lines: list[str] = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source_name = os.path.basename(metadata.get("source", "")) or label
        raw_page = metadata.get("page")
        page = raw_page + 1 if isinstance(raw_page, int) else raw_page
        block_type = metadata.get("block_type", "text")
        excerpt = re.sub(r"\s+", " ", doc.page_content or "").strip()
        excerpt = excerpt[:excerpt_limit]
        lines.append(
            f"[{label} {index}] source={source_name}; page={page}; block_type={block_type}; excerpt={excerpt}"
        )
    return "\n".join(lines)


def _build_recommendations(findings: list[dict[str, Any]], existing: list[str]) -> list[str]:
    recommendations: list[str] = []
    seen = set()

    for value in existing:
        cleaned = (value or "").strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            recommendations.append(cleaned)

    for finding in findings:
        cleaned = (finding.get("recommendation") or "").strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            recommendations.append(cleaned)

    return recommendations


def _normalize_findings(raw_findings: list[Any]) -> list[dict[str, Any]]:
    findings = [
        item.model_dump() if isinstance(item, BaseModel) else dict(item)
        for item in (raw_findings or [])
    ]
    findings.sort(key=lambda item: _severity_rank(item.get("severity", "medium")))
    return findings


def _extract_tgs_identifier(documents: list[Document]) -> str:
    if not documents:
        return ""

    ordered = sorted(documents, key=lambda doc: (doc.metadata or {}).get("page", 10**9))
    patterns = [
        r"\bTGS\s*#?\s*([A-Z0-9][A-Z0-9\-\/]*)\b",
        r"\b(CC\d{3,}[A-Z0-9\-\/]*)\b",
    ]

    for doc in ordered:
        text = re.sub(r"\s+", " ", doc.page_content or " ")
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = (match.group(1) or "").strip().upper()
                if candidate and candidate not in {"TGS", "TMP", "CTMP"}:
                    return candidate
    return ""


def _filter_tgs_options_analysis_pages(documents: list[Document]) -> tuple[list[Document], list[int]]:
    if not documents:
        return [], []

    pages_to_skip: set[tuple[str, int]] = set()
    skipped_page_numbers: set[int] = set()

    for doc in documents:
        metadata = doc.metadata or {}
        source_key = _normalize_path(metadata.get("source", ""))
        page = metadata.get("page")
        if not isinstance(page, int):
            continue

        lowered = (doc.page_content or "").lower()
        if "options analysis" in lowered and "risk assessment" in lowered:
            pages_to_skip.add((source_key, page))
            skipped_page_numbers.add(page + 1)

    if not pages_to_skip:
        return documents, []

    filtered_docs: list[Document] = []
    for doc in documents:
        metadata = doc.metadata or {}
        source_key = _normalize_path(metadata.get("source", ""))
        page = metadata.get("page")
        if isinstance(page, int) and (source_key, page) in pages_to_skip:
            continue
        filtered_docs.append(doc)

    return filtered_docs, sorted(skipped_page_numbers)


def _load_review_document(
    doc_type: str,
    path: str | None,
    *,
    include_images: bool = True,
    vision_model: str | None = None,
    vision_base_url: str | None = None,
    fallback_vision_model: str | None = None,
    fallback_vision_base_url: str | None = None,
    max_images_per_pdf: int = 4,
    vision_request_timeout_seconds: int = 90,
) -> list[Document]:
    if not path:
        return []
    if not os.path.exists(path):
        raise ReviewInputError(f"File not found for {doc_type}: {path}")

    docs = pdf_to_documents(
        path,
        include_images=include_images,
        vision_model=vision_model,
        vision_base_url=vision_base_url,
        fallback_vision_model=fallback_vision_model,
        fallback_vision_base_url=fallback_vision_base_url,
        max_images_per_pdf=max_images_per_pdf,
        vision_request_timeout_seconds=vision_request_timeout_seconds,
    )
    if not docs:
        raise ReviewInputError(f"No reviewable content extracted from {doc_type}: {path}")

    for doc in docs:
        doc.metadata["review_doc_type"] = doc_type
    return docs


class StandardsComplianceAgent:
    def __init__(self, llm: Any):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ReviewSubReportModel)

    def invoke(self, state: TrafficReviewState) -> dict[str, Any]:
        prompt = f"""
You are Agent 1, a traffic management standards compliance reviewer.

Your task is limited to checking the supplied TGS/TMP/CTMP package against the reference guidance evidence.

Project evidence may include extracted text, tables, and image-derived descriptions from plan sheets or diagrams.
Treat image-derived descriptions as visual evidence of signage, device layouts, lane closures, barriers, tapers, pedestrian paths, legends, and dimensions when they are specific enough.

Focus on:
- compliance with Austroads, national/federal, and Queensland guidance shown in the evidence
- whether the TGS design matches the TMP where the reference guidance clearly matters
- missing controls, missing dimensions, missing approvals, traffic staging issues, speed management, pedestrian and cyclist handling, emergency access, and device/layout issues

Rules:
- Use only the supplied evidence.
- If compliance cannot be verified from the evidence, say so explicitly.
- Findings must be concrete and correction-focused.
- Severity must be one of: critical, high, medium, low.
- Confidence must be one of: high, medium, low.

{self.parser.get_format_instructions()}

Additional focus: {state['additional_focus_text']}

Project evidence:
{state['project_context']}

Reference guidance evidence:
{state['reference_context']}
"""
        return _parse_llm_json_response(self.llm, prompt, self.parser)


class CommonPracticeReviewAgent:
    def __init__(self, llm: Any):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ReviewSubReportModel)

    def invoke(self, state: TrafficReviewState) -> dict[str, Any]:
        prompt = f"""
You are Agent 2, a practical traffic management review agent.

Your task is to review the supplied TGS/TMP/CTMP package against common implementation and review issues, even when the issue is not tied to one specific standard citation.

Project evidence may include extracted text, tables, and image-derived descriptions from plan sheets or diagrams.
Use those image-derived descriptions when checking signage presence, device placement, staging clarity, note conflicts, or visually obvious layout omissions.

Look for:
- TGS and TMP mismatches
- unclear staging or sequencing
- conflicting notes or inconsistent work hours
- missing sign schedules, taper lengths, buffer details, lane status, dimensions, or transitions
- weak pedestrian/cyclist continuity
- emergency access issues
- field constructability problems
- ambiguous responsibilities, approvals, or assumptions
- any review comment a senior designer or checker would normally raise before issue

Rules:
- Use only the supplied evidence.
- If a concern cannot be verified, state that clearly instead of guessing.
- Findings must be actionable and correction-focused.
- Severity must be one of: critical, high, medium, low.
- Confidence must be one of: high, medium, low.

{self.parser.get_format_instructions()}

Additional focus: {state['additional_focus_text']}

Project evidence:
{state['project_context']}
"""
        return _parse_llm_json_response(self.llm, prompt, self.parser)


class LeadReviewerAgent:
    def __init__(self, llm: Any):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=ReviewReportModel)

    def invoke(self, state: TrafficReviewState) -> dict[str, Any]:
        standards_json = json.dumps(state.get("standards_review", {}), indent=2)
        common_json = json.dumps(state.get("common_review", {}), indent=2)

        prompt = f"""
You are Agent 3, the lead traffic review engineer.

Combine Agent 1 and Agent 2 outputs into one final review report.

Responsibilities:
- merge overlapping findings
- keep the strongest evidence and highest defensible severity when duplicates exist
- preserve corrections that the planner or designer can act on immediately
- produce a balanced overall view of whether the package is ready, risky, or incomplete

Rules:
- Do not invent evidence beyond what the upstream agents reported.
- Prefer concise, professional findings.
- Recommendations should be deduplicated and written as direct correction actions.
- Severity must be one of: critical, high, medium, low.
- Confidence must be one of: high, medium, low.

{self.parser.get_format_instructions()}

Agent 1 standards review:
{standards_json}

Agent 2 practical/common review:
{common_json}
"""
        return _parse_llm_json_response(self.llm, prompt, self.parser)


class AgenticTrafficPlanReviewer:
    def __init__(self, llm: Any):
        self.standards_agent = StandardsComplianceAgent(llm)
        self.common_agent = CommonPracticeReviewAgent(llm)
        self.lead_agent = LeadReviewerAgent(llm)

    def _run_agents_parallel(self, initial_state: TrafficReviewState) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run Agent 1 and Agent 2 concurrently — they are independent so there is no reason to wait."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_standards = executor.submit(self.standards_agent.invoke, initial_state)
            future_common = executor.submit(self.common_agent.invoke, initial_state)
            standards_review = future_standards.result()
            common_review = future_common.result()

        return standards_review, common_review

    def invoke(self, project_context: str, reference_context: str, additional_focus_text: str) -> dict[str, Any]:
        initial_state: TrafficReviewState = {
            "project_context": project_context,
            "reference_context": reference_context,
            "additional_focus_text": additional_focus_text,
            "standards_review": {},
            "common_review": {},
            "final_report": {},
        }

        # Agent 1 and Agent 2 run in parallel; Agent 3 waits for both then merges.
        standards_review, common_review = self._run_agents_parallel(initial_state)

        merged_state: TrafficReviewState = {
            **initial_state,
            "standards_review": standards_review,
            "common_review": common_review,
            "final_report": {},
        }
        final_report = self.lead_agent.invoke(merged_state)

        return {
            "standards_review": standards_review,
            "common_review": common_review,
            "final_report": final_report,
            "workflow_mode": "parallel",
        }


def _run_single_pass_review(llm: Any, project_context_text: str, reference_context_text: str, additional_focus_text: str) -> dict[str, Any]:
    parser = JsonOutputParser(pydantic_object=ReviewReportModel)
    prompt = f"""
You are a senior Australian traffic management reviewer.

Review the supplied TGS, TMP, and CTMP evidence and identify:
1. Whether the TGS design matches the TMP.
2. Whether the project documents appear to comply with the retrieved Austroads, national/federal, and Queensland guidance evidence.
3. Any discrepancies or omissions inside the TMP, TGS, or CTMP.
4. Any other practical reviewer concerns worth flagging.

The project evidence may include extracted text, tables, and image-derived descriptions from drawings or plan sheets.
Use the image-derived descriptions to reason about signage, traffic control devices, barriers, tapers, pedestrian routes, legends, and plan layout when they provide concrete detail.

Review priorities:
- cross-document consistency between drawing notes, staging, sign layouts, traffic control devices, road user management, and worksite constraints
- pedestrian and cyclist continuity
- emergency vehicle access
- temporary speed management
- approvals, roles, work hours, and implementation sequencing
- missing dimensions, conflicting notes, unverified assumptions, or items that would block field implementation

Rules:
- Use only the supplied evidence.
- If evidence is incomplete, say it is unable to be verified.
- Prefer findings that lead to a concrete correction.
- Each finding must include document-specific evidence.
- Each recommendation must be an edit or verification action the designer/planner can perform.
- Severity must be one of: critical, high, medium, low.
- Confidence must be one of: high, medium, low.

Additional focus requested by the user: {additional_focus_text}

{parser.get_format_instructions()}

Project document evidence:
{project_context_text}

Reference guideline evidence:
{reference_context_text}
"""
    return _parse_llm_json_response(llm, prompt, parser)


def render_review_markdown(report: dict[str, Any]) -> str:
    lines = ["# Traffic Document Review", ""]

    overview = (report.get("overview") or "").strip()
    if overview:
        lines.extend(["## Overview", "", overview, ""])

    assumptions = report.get("assumptions") or []
    if assumptions:
        lines.extend(["## Assumptions", ""])
        for item in assumptions:
            lines.append(f"- {item}")
        lines.append("")

    findings = report.get("findings") or []
    if findings:
        lines.extend(["## Findings", ""])
        for index, finding in enumerate(findings, start=1):
            severity = (finding.get("severity") or "medium").upper()
            title = finding.get("title") or f"Finding {index}"
            lines.append(f"### {index}. [{severity}] {title}")
            lines.append("")
            lines.append(finding.get("issue") or "")
            lines.append("")

            affected = finding.get("affected_documents") or []
            if affected:
                lines.append(f"Affected documents: {', '.join(affected)}")
                lines.append("")

            evidence = finding.get("evidence") or []
            if evidence:
                lines.append("Evidence:")
                for item in evidence:
                    lines.append(f"- {item}")
                lines.append("")

            guideline_basis = finding.get("guideline_basis") or []
            if guideline_basis:
                lines.append("Guideline basis:")
                for item in guideline_basis:
                    lines.append(f"- {item}")
                lines.append("")

            recommendation = finding.get("recommendation") or ""
            if recommendation:
                lines.append(f"Recommendation: {recommendation}")
                lines.append("")

    recommendations = report.get("recommendations") or []
    if recommendations:
        lines.extend(["## Recommended Corrections", ""])
        for index, item in enumerate(recommendations, start=1):
            lines.append(f"{index}. {item}")
        lines.append("")

    sources = report.get("reference_sources") or []
    if sources:
        lines.extend(["## Reference Sources Used", ""])
        for item in sources:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def review_traffic_documents(
    tgs_path: str | None,
    tmp_path: str | None,
    ctmp_path: str | None = None,
    *,
    llm: Any,
    vectorstore: Any,
    reranker: Any = None,
    additional_focus: list[str] | None = None,
    extra_project_context: str | None = None,
    extra_context_source_path: str | None = None,
    use_agentic_workflow: bool = True,
    include_images: bool = True,
    vision_model: str | None = None,
    vision_base_url: str | None = None,
    fallback_vision_model: str | None = None,
    fallback_vision_base_url: str | None = None,
    image_scan_mode: str = "auto",
    max_images_per_pdf: int = 4,
    vision_request_timeout_seconds: int = 90,
    progress_callback: Any = None,
) -> dict[str, Any]:
    if not any((tgs_path, tmp_path, ctmp_path)):
        raise ReviewInputError("Upload at least one review document: TGS, TMP, or CTMP")

    def emit_progress(stage: str, message: str, percent: int) -> None:
        if callable(progress_callback):
            try:
                progress_callback({"stage": stage, "message": message, "percent": percent})
            except Exception:
                pass

    image_scan_mode = (image_scan_mode or "auto").strip().lower()
    available_paths = {
        "tgs": tgs_path,
        "tmp": tmp_path,
        "ctmp": ctmp_path,
    }

    image_scan_targets: set[str] = set()
    if include_images:
        if image_scan_mode == "all":
            image_scan_targets = {key for key, path in available_paths.items() if path}
        elif image_scan_mode in {"tgs", "tmp", "ctmp"} and available_paths.get(image_scan_mode):
            image_scan_targets = {image_scan_mode}
        else:
            # Auto mode: prefer TGS, otherwise TMP, otherwise CTMP.
            for candidate in ("tgs", "tmp", "ctmp"):
                if available_paths.get(candidate):
                    image_scan_targets = {candidate}
                    break

    load_specs = [
        ("TGS", tgs_path, "tgs" in image_scan_targets),
        ("TMP", tmp_path, "tmp" in image_scan_targets),
        ("CTMP", ctmp_path, "ctmp" in image_scan_targets),
    ]

    target_text = ", ".join(sorted(target.upper() for target in image_scan_targets)) or "none"
    emit_progress("extracting", f"Extracting uploaded documents. Image scan targets: {target_text}", 22)

    doc_map: dict[str, list[Document]] = {"TGS": [], "TMP": [], "CTMP": []}
    available_count = sum(1 for _, path, _ in load_specs if path)
    completed_count = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                _load_review_document,
                doc_type,
                path,
                include_images=scan_images,
                vision_model=vision_model,
                vision_base_url=vision_base_url,
                fallback_vision_model=fallback_vision_model,
                fallback_vision_base_url=fallback_vision_base_url,
                max_images_per_pdf=max_images_per_pdf,
                vision_request_timeout_seconds=vision_request_timeout_seconds,
            ): doc_type
            for doc_type, path, scan_images in load_specs
        }
        for future in as_completed(futures):
            doc_type = futures[future]
            doc_map[doc_type] = future.result()
            completed_count += 1
            pct = 22 if available_count <= 0 else 22 + int((completed_count / available_count) * 18)
            emit_progress("extracting", f"Extracted {doc_type} evidence ({completed_count}/{available_count})", pct)

    tgs_identifier = _extract_tgs_identifier(doc_map.get("TGS", []))
    filtered_tgs_docs, skipped_tgs_pages = _filter_tgs_options_analysis_pages(doc_map.get("TGS", []))
    doc_map["TGS"] = filtered_tgs_docs

    # Consolidated review queries (reduced from 5 to 2 for speed)
    review_queries = [
        "work location drawing scope chainage stage lane closure shoulder closure detour pedestrian cyclist access speed traffic controller night work dates sign schedule taper buffer",
        "traffic guidance scheme management plan consistency sequencing implementation responsibilities approvals staging work area delineation barrier cones clearance legend signage",
    ]
    if tgs_identifier:
        review_queries.insert(0, f"TGS number {tgs_identifier} drawing revision title block identification")
    if additional_focus:
        review_queries.extend(additional_focus)

    # Extract and cache project chunks once (avoid redundant re-ranking)
    emit_progress("chunking", "Selecting key project evidence chunks.", 44)
    project_chunks_by_type: dict[str, list[Document]] = {}
    for doc_type, docs in doc_map.items():
        project_chunks_by_type[doc_type] = _select_review_chunks(docs, review_queries, reranker=None)
    
    project_evidence: list[Document] = []
    for chunks in project_chunks_by_type.values():
        project_evidence.extend(chunks)
    project_evidence = _dedupe_documents(project_evidence)

    if not project_evidence:
        raise ReviewInputError("No usable evidence was extracted from the supplied review documents")

    # Single consolidated reference query instead of 4 separate queries with re-ranking
    reference_query = (
        "temporary traffic management plan traffic guidance scheme consistency sign schedule "
        "taper buffer spacing staging pedestrian cyclist access detour speed lane closure "
        "traffic controller conflicting notes missing dimensions approvals work hours emergency access"
    )
    reference_queries = [reference_query]
    if additional_focus:
        reference_queries.extend(additional_focus[:2])  # Limit additional focus queries

    # Skip re-ranking for reference retrieval (reranker=None)
    emit_progress("reference", "Retrieving reference guideline evidence.", 58)
    reference_evidence = _retrieve_reference_chunks(
        vectorstore,
        reference_queries,
        reranker=None,  # No re-ranking for faster retrieval
        per_query_fetch=8,  # Reduced from 10
        per_query_limit=3,  # Reduced from 4
        total_limit=12,     # Reduced from 16
    )

    # Use cached chunks (no re-ranking)
    project_context = []
    if tgs_identifier:
        project_context.append(f"[TGS IDENTIFIER]\nTGS number detected from title block: {tgs_identifier}")
    if skipped_tgs_pages:
        pages_text = ", ".join(str(page) for page in skipped_tgs_pages)
        project_context.append(f"[TGS PAGE FILTER]\nExcluded TGS page(s): {pages_text} because they are marked 'Options Analysis & Risk Assessment'.")
    for doc_type, chunks in project_chunks_by_type.items():
        project_context.append(_format_chunks(doc_type, chunks))
    if extra_project_context and extra_project_context.strip():
        project_context.append(f"[EMAIL CONTEXT]\n{extra_project_context.strip()[:12000]}")
    project_context_text = "\n\n".join(project_context)
    reference_context_text = _format_chunks("REFERENCE", reference_evidence)
    additional_focus_text = ", ".join(additional_focus or []) or "none supplied"
    emit_progress("review", "Running review reasoning and findings generation.", 72)
    try:
        if use_agentic_workflow:
            workflow_result = AgenticTrafficPlanReviewer(llm).invoke(
                project_context=project_context_text,
                reference_context=reference_context_text,
                additional_focus_text=additional_focus_text,
            )
            report_data = workflow_result.get("final_report", {})
            workflow_metadata = {
                "mode": "three-agent-review",
                "execution": workflow_result.get("workflow_mode", "sequential"),
                "agents": {
                    "standards_review": workflow_result.get("standards_review", {}),
                    "common_review": workflow_result.get("common_review", {}),
                },
            }
        else:
            report_data = _run_single_pass_review(
                llm=llm,
                project_context_text=project_context_text,
                reference_context_text=reference_context_text,
                additional_focus_text=additional_focus_text,
            )
            workflow_metadata = {
                "mode": "single-pass-review",
                "execution": "single_prompt",
                "agents": {},
            }
    except Exception:
        emit_progress("review", "Primary review mode failed, retrying single-pass review.", 82)
        report_data = _run_single_pass_review(
            llm=llm,
            project_context_text=project_context_text,
            reference_context_text=reference_context_text,
            additional_focus_text=additional_focus_text,
        )
        workflow_metadata = {
            "mode": "single-pass-review",
            "execution": "fallback_after_agentic_error",
            "agents": {},
        }

    findings = _normalize_findings(report_data.get("findings", []))
    emit_progress("finalizing", "Finalizing report and recommendations.", 92)

    report = {
        "overview": report_data.get("overview", ""),
        "assumptions": report_data.get("assumptions", []),
        "findings": findings,
        "recommendations": _build_recommendations(findings, report_data.get("recommendations", [])),
        "tgs_identifier": tgs_identifier,
        "excluded_tgs_pages": skipped_tgs_pages,
        "review_scope": [doc_type for doc_type, path in (("TGS", tgs_path), ("TMP", tmp_path), ("CTMP", ctmp_path)) if path],
        "source_documents": [
            {
                "type": doc_type,
                "path": path,
                "chunks_extracted": len(doc_map[doc_type]),
            }
            for doc_type, path in (("TGS", tgs_path), ("TMP", tmp_path), ("CTMP", ctmp_path))
            if path
        ],
        "reference_sources": sorted(
            {
                os.path.basename(doc.metadata.get("source", ""))
                for doc in reference_evidence
                if doc.metadata.get("source")
            }
        ),
        "review_workflow": workflow_metadata,
        "image_scan": {
            "enabled": bool(include_images),
            "mode": image_scan_mode,
            "targets": sorted(target.upper() for target in image_scan_targets),
            "max_images_per_pdf": max_images_per_pdf,
        },
    }
    if extra_project_context and extra_project_context.strip():
        report["review_scope"].append("EMAIL_CONTEXT")
        report["source_documents"].append({
            "type": "EMAIL_CONTEXT",
            "path": extra_context_source_path or "",
            "chunks_extracted": 1,
        })
    report["markdown_report"] = render_review_markdown(report)
    return report


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review TGS/TMP/CTMP documents against each other and indexed guidance")
    parser.add_argument("--tgs", help="Path to the TGS PDF")
    parser.add_argument("--tmp", help="Path to the TMP PDF")
    parser.add_argument("--ctmp", help="Path to the CTMP PDF")
    parser.add_argument("--focus", action="append", default=[], help="Extra focus area to inject into the review")
    parser.add_argument("--json-output", help="Optional path for JSON report output")
    parser.add_argument("--markdown-output", help="Optional path for markdown report output")
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()

    import app as app_module

    if app_module.vectorstore is None:
        app_module.initialize_or_reload_index(force_rebuild=False)

    report = review_traffic_documents(
        args.tgs,
        args.tmp,
        ctmp_path=args.ctmp,
        llm=app_module.llm,
        vectorstore=app_module.vectorstore,
        reranker=app_module.reranker,
        additional_focus=args.focus,
    )

    print(report["markdown_report"])

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    if args.markdown_output:
        with open(args.markdown_output, "w", encoding="utf-8") as handle:
            handle.write(report["markdown_report"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
