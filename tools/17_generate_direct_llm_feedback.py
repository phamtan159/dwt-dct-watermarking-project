from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SYSTEM_PROMPT = """You are an English pronunciation coach for Vietnamese learners.

Task:
- Read evidence for exactly one phoneme.
- Return useful learner feedback in Vietnamese.
- Return pure JSON only:
{
  "diagnosis": "string",
  "correction_steps": ["string", "string"]
}

Policy:
- Do not use human_label, classifier label, rule label, or outside information.
- Do not invent numbers. If evidence is weak or contradictory, say that the sign is not certain.
- If feedback_policy.analysis_depth is "articulatory", use only the provided articulatory evidence: duration, speech_attribute_prediction.feature_confidence, frication_vs_stop, vowel_quality, WavLM summary/delta, visual summary, and primary_evidence_policy.
- If feedback_policy.analysis_depth is "basic", do not discuss articulatory features, WavLM, or visual evidence. Explain only the normal alignment issue, such as a missing or extra sound.
- If alignment_op is "match" and there is no clear error evidence, return a short OK diagnosis and an empty correction_steps list.
- If the segment is correct, not insertion/deletion/repetition, and not a hard-focus error case that needs careful coaching, do not add correction steps.
- Do not return category, severity, confidence, markdown, or a long explanation.
- diagnosis should be 1-2 Vietnamese sentences.
- correction_steps should be empty for correct/no-error segments, otherwise 2-4 short concrete practice steps.
"""


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_user_prompt(context: dict) -> str:
    policy = context.get("feedback_policy") or {}
    depth = policy.get("analysis_depth")
    if depth == "basic":
        task = (
            "This is a non-focus insertion/deletion case. Create basic feedback only. "
            "Do not mention frication, stop/plosive confidence, WavLM, spectral shape, or visual mouth features."
        )
    else:
        task = (
            "Create diagnosis and correction_steps for this phoneme using the provided evidence fields. "
            "Respect primary_evidence_policy when weighing audio vs visual evidence. "
            "If alignment_op is match and the evidence does not show a clear problem, return correction_steps as an empty list."
        )
    return (
        f"{task}\n"
        "Return pure JSON, no markdown.\n\n"
        f"EVIDENCE:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_feedback(payload: dict) -> dict:
    diagnosis = payload.get("diagnosis")
    steps = payload.get("correction_steps")
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        diagnosis = "Evidence chua du ro de ket luan chac chan ve am nay."
    if not isinstance(steps, list):
        steps = []
    clean_steps = [str(item).strip() for item in steps if str(item).strip()]
    if not clean_steps:
        clean_steps = [
            "Doc cham lai am muc tieu va so sanh voi mau chuan.",
            "Giu cau hinh mieng on dinh truoc khi noi sang am ke tiep.",
        ]
    return {
        "diagnosis": diagnosis.strip(),
        "correction_steps": clean_steps[:4],
    }


def deterministic_no_issue(context: dict) -> dict:
    alignment = context.get("alignment") or {}
    expected = alignment.get("expected_phoneme") or context.get("target_phoneme") or ""
    observed = alignment.get("observed_phoneme") or ""
    return {
        "diagnosis": (
            f"Wav2Vec2/MFA ghi nhan am /{observed or expected}/ khop voi ky vong /{expected}/, "
            "nen khong can phan tich loi chi tiet cho am nay."
        ),
        "correction_steps": [],
    }


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    context: dict,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> dict:
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": compact_user_prompt(context)},
        ],
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(base_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            content = payload["choices"][0]["message"]["content"]
            return normalize_feedback(extract_json_object(content))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM request failed after {max_retries + 1} attempt(s): {last_error}")


def iter_segments(payload: dict, speaker: str | None, sample_id: str | None, segment_id: str | None):
    for sample in payload.get("samples", []):
        if speaker and sample.get("speaker_id") != speaker:
            continue
        sid = sample.get("id") or ""
        if sample_id and sample_id not in sid:
            continue
        for seg in sample.get("segments", []):
            if segment_id and seg.get("segment_id") != segment_id:
                continue
            yield sample, seg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call an OpenAI-compatible LLM API to generate diagnosis/correction_steps from direct LLM inputs."
    )
    parser.add_argument("--input", default="data/final/direct_llm_feedback_inputs.json")
    parser.add_argument("--output", default="data/final/direct_llm_feedback_outputs.json")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--speaker")
    parser.add_argument("--sample-id", help="Substring filter, e.g. thin_C_01.")
    parser.add_argument("--segment-id", help="Exact segment id, e.g. 013_theta.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        raise RuntimeError(
            f"Missing API key. Set ${args.api_key_env} or $OPENAI_API_KEY before running this tool."
        )
    if not args.dry_run and not args.model:
        raise RuntimeError("Missing model. Pass --model or set $env:LLM_MODEL.")

    source = load_json(Path(args.input))
    out_samples: dict[str, dict] = {}
    processed = 0
    llm_calls = 0

    for sample, seg in iter_segments(source, args.speaker, args.sample_id, args.segment_id):
        if args.limit is not None and processed >= args.limit:
            break
        sample_key = sample.get("id") or sample.get("sample_id") or "unknown_sample"
        out_sample = out_samples.setdefault(
            sample_key,
            {
                "id": sample_key,
                "speaker_id": sample.get("speaker_id"),
                "segments": [],
            },
        )
        context = seg.get("llm_context") or {}
        policy = context.get("feedback_policy") or {}
        llm_required = bool(policy.get("llm_required", True))

        if not llm_required:
            feedback = deterministic_no_issue(context)
        elif args.dry_run:
            feedback = {
                "diagnosis": "DRY RUN: tool se gui llm_context nay cho LLM de sinh diagnosis.",
                "correction_steps": ["DRY RUN: kiem tra prompt/input truoc khi goi API."],
            }
            llm_calls += 1
        else:
            feedback = call_openai_compatible(
                base_url=args.base_url,
                api_key=api_key or "",
                model=args.model or "",
                context=context,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            llm_calls += 1

        out_sample["segments"].append(
            {
                "segment_id": seg.get("segment_id"),
                "word": seg.get("word"),
                "target_phoneme": seg.get("target_phoneme"),
                "target_phoneme_normalized": seg.get("target_phoneme_normalized"),
                "feedback_mode": seg.get("feedback_mode") or policy.get("feedback_mode"),
                "analysis_depth": seg.get("analysis_depth") or policy.get("analysis_depth"),
                "llm_required": llm_required,
                **feedback,
            }
        )
        processed += 1
        print(f"Processed {processed}: {sample_key} / {seg.get('segment_id')}")

    output = {
        "schema_version": "direct_llm_feedback_outputs_v2",
        "source_input": str(Path(args.input)).replace("\\", "/"),
        "model": args.model if not args.dry_run else "dry-run",
        "num_samples": len(out_samples),
        "num_segments": processed,
        "num_llm_calls": llm_calls,
        "samples": list(out_samples.values()),
    }
    write_json(Path(args.output), output)
    print(f"Wrote {args.output}")
    print(f"LLM calls: {llm_calls}")


if __name__ == "__main__":
    main()
