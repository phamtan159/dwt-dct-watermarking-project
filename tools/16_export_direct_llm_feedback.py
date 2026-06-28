from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


OUTPUT_SCHEMA = {
    "diagnosis": "1-2 Vietnamese sentences explaining what is happening.",
    "correction_steps": ["2-4 concrete Vietnamese coaching steps."],
}

# Standard target phonemes named in pronunciation_error.md. Unicode escapes keep
# this policy stable even when the Windows console displays IPA as mojibake.
DEFAULT_FOCUS_PHONEMES = {
    "\u03b8",  # theta
    "\u00f0",  # eth / dh
    "v",
    "\u0283",  # sh
    "z",
    "r",
    "\u0279",  # English r in some phoneme inventories
    "p",
    "t",
    "k",
    "d",
    "s",
    "g",
    "f",
    "d\u0292",  # j as in major
    "\u026a",  # short i
    "i",
    "i\u02d0",  # long i
    "\u00e6",
    "e",
    "\u025b",
}

PHONE_ALIASES = {
    "\u0279": "r",
    "\u025b": "e",
    "i\u02d0": "i",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def rounded(value, digits: int = 6):
    value = safe_float(value)
    if value is None:
        return None
    return round(value, digits)


def fix_mojibake_text(value) -> str:
    text = str(value or "")
    if not text:
        return ""
    for encoding in ("cp1252", "latin1"):
        try:
            fixed = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if fixed != text:
            return fixed
    return text


def normalize_phone(value) -> str:
    phone = fix_mojibake_text(value).strip().lower()
    phone = phone.strip("`/[](){} ")
    phone = phone.replace(" ", "")
    return PHONE_ALIASES.get(phone, phone)


def split_phone_group(value) -> list[str]:
    fixed = fix_mojibake_text(value)
    raw_parts = re.split(r"[,/|; ]+", fixed)
    phones = []
    for part in raw_parts:
        phone = normalize_phone(part)
        if phone:
            phones.append(phone)
    return phones


def focus_phonemes_from_file(path: Path) -> set[str]:
    focus = {normalize_phone(item) for item in DEFAULT_FOCUS_PHONEMES}
    if not path.exists():
        return focus

    text = fix_mojibake_text(path.read_text(encoding="utf-8", errors="replace"))
    # Only import slash groups that contain phones already accepted as standard
    # focus targets. This avoids treating common replacement phones such as /j/
    # and /w/ as target phones just because they appear in "v -> j/w".
    for group in re.findall(r"/([^/]+?)/", text):
        for phone in split_phone_group(group):
            canonical = PHONE_ALIASES.get(phone, phone)
            if canonical in focus:
                focus.add(canonical)
    return focus


def target_phoneme(seg: dict) -> str:
    return str(seg.get("target_phoneme") or seg.get("phoneme_standard") or seg.get("standard_phone") or "")


def observed_phoneme(seg: dict) -> str:
    return str(seg.get("observed_phoneme") or seg.get("phoneme_real") or seg.get("phone") or "")


def target_phonemes_from_label_map(label_map_path: Path) -> set[str]:
    if not label_map_path.exists():
        return set()
    label_map = load_json(label_map_path)
    phonemes = label_map.get("phonemes") or {}
    if not isinstance(phonemes, dict):
        return set()
    targets = set()
    for key, item in phonemes.items():
        if isinstance(item, dict):
            targets.add(normalize_phone(item.get("phoneme") or key))
        else:
            targets.add(normalize_phone(key))
    return targets


def target_is_focus(seg: dict, focus_phonemes: set[str]) -> bool:
    target = normalize_phone(target_phoneme(seg))
    if target in focus_phonemes:
        return True
    parts = [normalize_phone(part) for part in re.split(r"[_+\-]", target) if part]
    if any(part in focus_phonemes for part in parts):
        return True
    return any(phone and phone in target for phone in ("\u03b8", "\u00f0", "d\u0292") if phone in focus_phonemes)


def alignment_context(seg: dict) -> dict:
    expected_raw = target_phoneme(seg)
    observed_raw = observed_phoneme(seg)
    return {
        "expected_phoneme": expected_raw,
        "expected_phoneme_normalized": normalize_phone(expected_raw),
        "observed_phoneme": observed_raw,
        "observed_phoneme_normalized": normalize_phone(observed_raw),
        "alignment_op": str(seg.get("alignment_op") or "").lower(),
    }


def primary_evidence_policy(seg: dict) -> str:
    target = normalize_phone(target_phoneme(seg))
    if target:
        return "audio"
    return "uncertain"


def sample_key(sample: dict) -> str:
    return str(sample.get("id") or sample.get("sample_id") or sample.get("audio_id") or "")


def segment_key(sample: dict, seg: dict) -> tuple[str, str]:
    return sample_key(sample), str(seg.get("id") or seg.get("segment_id") or "")


def load_mdd_predictions(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    out = {}
    for sample in payload.get("samples", []):
        sid = str(sample.get("id") or sample.get("sample_id") or sample.get("audio_id") or "")
        for seg in sample.get("segments", []):
            seg_id = str(seg.get("id") or seg.get("segment_id") or "")
            if sid and seg_id:
                out[(sid, seg_id)] = {
                    "mdd_classifier": seg.get("mdd_classifier") or {},
                    "feature_coverage": seg.get("feature_coverage") or {},
                }
    return out


def load_stage2_predictions(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    out = {}
    for sample in payload.get("samples", []):
        sid = str(sample.get("id") or sample.get("sample_id") or sample.get("audio_id") or "")
        for seg in sample.get("segments", []):
            seg_id = str(seg.get("id") or seg.get("segment_id") or "")
            if sid and seg_id:
                out[(sid, seg_id)] = {
                    "stage2_observed_phone": seg.get("stage2_observed_phone") or {},
                    "stage2_feature_coverage": seg.get("feature_coverage") or {},
                    "wav2vec2_observed_phoneme": seg.get("wav2vec2_observed_phoneme"),
                }
    return out


def feedback_policy(seg: dict, focus_phonemes: set[str]) -> dict:
    alignment = alignment_context(seg)
    expected = alignment["expected_phoneme_normalized"]
    observed = alignment["observed_phoneme_normalized"]
    op = alignment["alignment_op"]
    in_focus = target_is_focus(seg, focus_phonemes)
    is_match = op == "match" or (expected and observed and expected == observed and op not in {"delete", "insert"})

    if in_focus:
        return {
            "is_focus_phoneme": True,
            "feedback_mode": "focus_articulatory",
            "analysis_depth": "articulatory",
            "llm_required": True,
            "reason": "Target phoneme is listed in pronunciation_error.md.",
        }
    if is_match:
        return {
            "is_focus_phoneme": False,
            "feedback_mode": "nonfocus_ok",
            "analysis_depth": "none",
            "llm_required": False,
            "reason": "Non-focus phoneme matches Wav2Vec2/MFA expectation.",
        }
    if op == "substitute" or (expected and observed and expected != observed and op not in {"delete", "insert"}):
        return {
            "is_focus_phoneme": False,
            "feedback_mode": "nonfocus_substitution",
            "analysis_depth": "articulatory",
            "llm_required": True,
            "reason": "Non-focus phoneme was substituted, so normal LLM error analysis is allowed.",
        }
    if op in {"delete", "insert", "deletion", "insertion"}:
        return {
            "is_focus_phoneme": False,
            "feedback_mode": "nonfocus_insertion_or_deletion",
            "analysis_depth": "basic",
            "llm_required": True,
            "reason": "Non-focus phoneme is missing or extra; use basic feedback, not articulatory-feature analysis.",
        }
    return {
        "is_focus_phoneme": False,
        "feedback_mode": "nonfocus_uncertain",
        "analysis_depth": "basic",
        "llm_required": True,
        "reason": "Alignment is not a clean match, substitution, insertion, or deletion.",
    }


def speech_context(seg: dict) -> dict:
    prediction = seg.get("speech_attribute_prediction") or {}
    confidence = prediction.get("feature_confidence") or {}
    fricative = safe_float(confidence.get("fricative"))
    plosive = safe_float(confidence.get("plosive"))
    frication_vs_stop = {
        "fricative": rounded(fricative),
        "plosive_or_stop": rounded(plosive),
    }
    if fricative is not None and plosive is not None:
        frication_vs_stop["fricative_minus_stop"] = rounded(fricative - plosive)

    start = safe_float(prediction.get("effective_start"))
    end = safe_float(prediction.get("effective_end"))
    duration = None
    if start is not None and end is not None:
        duration = max(0.0, end - start)

    return {
        "feature_confidence": {
            key: rounded(value)
            for key, value in confidence.items()
            if safe_float(value) is not None
        },
        "frication_vs_stop": frication_vs_stop,
        "vowel_quality": {
            key: rounded(confidence.get(key))
            for key in (
                "vowel",
                "front",
                "back",
                "central",
                "high",
                "mid",
                "low",
                "long",
                "short",
                "round",
                "diphthong",
                "monophthong",
            )
            if safe_float(confidence.get(key)) is not None
        },
        "window": {
            "effective_start": rounded(prediction.get("effective_start")),
            "effective_end": rounded(prediction.get("effective_end")),
            "effective_duration": rounded(duration),
            "context_seconds": rounded(prediction.get("context_seconds")),
            "num_frames": rounded(prediction.get("num_frames")),
        },
    }


def audio_side(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    return {
        "num_frames": rounded(payload.get("num_frames")),
        "embedding_norm": rounded(payload.get("embedding_norm")),
        "embedding_std": rounded(payload.get("embedding_std")),
        "activation_min": rounded(payload.get("activation_min")),
        "activation_max": rounded(payload.get("activation_max")),
    }


def wavlm_context(seg: dict) -> dict:
    audio = seg.get("audio_attributes") or {}
    standard = audio.get("standard") or seg.get("wavlm_standard_attributes") or {}
    real = audio.get("real") or seg.get("wavlm_real_attributes") or {}
    standard_out = audio_side(standard)
    real_out = audio_side(real)
    delta = {}
    for key in ("num_frames", "embedding_norm", "embedding_std", "activation_min", "activation_max"):
        left = safe_float(standard.get(key))
        right = safe_float(real.get(key))
        if left is not None and right is not None:
            delta[key] = rounded(right - left)
            delta[f"abs_{key}"] = rounded(abs(right - left))
    return {
        "standard": standard_out,
        "real": real_out,
        "delta_real_minus_standard": delta,
    }


def duration_context(seg: dict) -> dict:
    start = safe_float(seg.get("start"))
    end = safe_float(seg.get("end"))
    raw_start = safe_float(seg.get("raw_start"))
    raw_end = safe_float(seg.get("raw_end"))
    return {
        "aligned_start": rounded(start),
        "aligned_end": rounded(end),
        "aligned_duration": rounded(max(0.0, end - start) if start is not None and end is not None else None),
        "raw_start": rounded(raw_start),
        "raw_end": rounded(raw_end),
        "raw_duration": rounded(max(0.0, raw_end - raw_start) if raw_start is not None and raw_end is not None else None),
    }


def compact_segment(
    sample: dict,
    seg: dict,
    focus_phonemes: set[str],
    mdd_predictions: dict | None = None,
    stage2_predictions: dict | None = None,
) -> dict:
    policy = feedback_policy(seg, focus_phonemes)
    analysis_depth = policy["analysis_depth"]
    mdd_predictions = mdd_predictions or {}
    stage2_predictions = stage2_predictions or {}
    mdd_prediction = mdd_predictions.get(segment_key(sample, seg))
    stage2_prediction = stage2_predictions.get(segment_key(sample, seg))
    payload = {
        "sample_id": sample.get("id") or sample.get("sample_id") or sample.get("audio_id"),
        "speaker_id": sample.get("speaker_id"),
        "word": seg.get("word"),
        "target_phoneme": target_phoneme(seg),
        "position": {
            "word_index": seg.get("word_index"),
            "phoneme_index_in_word": seg.get("phoneme_index_in_word"),
            "is_word_initial": safe_float(seg.get("phoneme_index_in_word")) == 0,
        },
        "alignment": alignment_context(seg),
        "feedback_policy": policy,
        "primary_evidence_policy": primary_evidence_policy(seg),
    }
    if mdd_prediction:
        payload["mdd_classifier"] = mdd_prediction["mdd_classifier"]
        payload["mdd_feature_coverage"] = mdd_prediction["feature_coverage"]
    if stage2_prediction:
        stage2_observed = stage2_prediction["stage2_observed_phone"]
        payload["stage2_observed_phone"] = stage2_observed
        payload["stage2_feature_coverage"] = stage2_prediction["stage2_feature_coverage"]
        payload["wav2vec2_observed_phoneme_for_comparison"] = stage2_prediction.get("wav2vec2_observed_phoneme")
        stage1_error = bool((payload.get("mdd_classifier") or {}).get("is_error"))
        payload["observed_phone_decision"] = {
            "primary_source": "stage2_observed_phone" if stage1_error and stage2_observed.get("prediction") else "wav2vec2_alignment",
            "primary_observed_phone": (
                stage2_observed.get("prediction")
                if stage1_error and stage2_observed.get("prediction")
                else alignment_context(seg)["observed_phoneme"]
            ),
            "stage2_used_when": "mdd_classifier.is_error is true",
            "wav2vec2_role": "supporting evidence and sanity check",
        }

    if analysis_depth == "none":
        return payload

    payload["duration"] = duration_context(seg)
    if analysis_depth == "basic":
        return payload

    payload["speech_attribute_prediction"] = speech_context(seg)
    payload["wavlm_summary"] = wavlm_context(seg)
    return payload


def prompt_for_context(context: dict) -> str:
    policy = context.get("feedback_policy") or {}
    depth = policy.get("analysis_depth")
    if depth == "basic":
        extra = (
            "This is a non-focus insertion/deletion case. Do not analyze articulatory features, "
            "or WavLM evidence. Explain the missing/extra sound in simple learner language."
        )
    elif depth == "none":
        extra = "No LLM call is needed because the non-focus phoneme matches the expected phoneme."
    else:
        extra = (
            "Use articulatory evidence only from the provided fields: duration, "
            "speech_attribute_prediction.feature_confidence, frication_vs_stop, vowel_quality, "
            "WavLM summary/delta, and primary_evidence_policy. "
            "Use observed_phone_decision.primary_observed_phone as the main observed-phone candidate. "
            "When its primary_source is stage2_observed_phone, treat Wav2Vec2 observed/alignment phone "
            "as supporting evidence and a sanity check. If Stage 2, Wav2Vec2 posterior, and speech "
            "attributes strongly disagree, state uncertainty instead of pretending the diagnosis is certain."
        )
    return (
        "You are an English pronunciation coach for Vietnamese learners. "
        "Use only the JSON evidence below. Do not invent numbers. "
        "Return only diagnosis and correction_steps in Vietnamese.\n\n"
        f"POLICY:\n{extra}\n\n"
        f"OUTPUT_SCHEMA:\n{json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"EVIDENCE:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export compact direct-to-LLM feedback prompts from segment attributes."
    )
    parser.add_argument("--input", default="data/final/segment_attributes.json")
    parser.add_argument("--label-map", default="data/label_map.json")
    parser.add_argument("--focus-phoneme-file", default="pronunciation_error.md")
    parser.add_argument("--output", default="data/final/direct_llm_feedback_inputs.json")
    parser.add_argument("--mdd-predictions", default="data/final/mdd_predictions.json")
    parser.add_argument("--stage2-predictions", default="data/final/stage2_observed_phone_predictions.json")
    parser.add_argument("--all-segments", action="store_true")
    parser.add_argument("--speaker", help="Optional speaker filter, e.g. speaker-01.")
    parser.add_argument("--sample-id", help="Optional sample id substring filter, e.g. thin_C_03.")
    parser.add_argument("--segment-id", help="Optional exact segment id filter, e.g. 013_theta.")
    args = parser.parse_args()

    dataset = load_json(Path(args.input))
    focus_phonemes = focus_phonemes_from_file(Path(args.focus_phoneme_file))
    interested = focus_phonemes | target_phonemes_from_label_map(Path(args.label_map))
    mdd_predictions = load_mdd_predictions(Path(args.mdd_predictions)) if args.mdd_predictions else {}
    stage2_predictions = load_stage2_predictions(Path(args.stage2_predictions)) if args.stage2_predictions else {}

    samples = []
    segment_count = 0
    mode_counts: dict[str, int] = {}
    for sample in dataset.get("samples", []):
        sample_key = sample.get("id") or sample.get("sample_id") or sample.get("audio_id") or ""
        if args.speaker and sample.get("speaker_id") != args.speaker:
            continue
        if args.sample_id and args.sample_id not in sample_key:
            continue

        out_segments = []
        for seg in sample.get("segments", []):
            segment_id = seg.get("id") or seg.get("segment_id")
            if args.segment_id and segment_id != args.segment_id:
                continue
            target = target_phoneme(seg)
            normalized_target = normalize_phone(target)
            if not args.all_segments and interested and normalized_target not in interested:
                continue
            context = compact_segment(sample, seg, focus_phonemes, mdd_predictions, stage2_predictions)
            mode = context["feedback_policy"]["feedback_mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            out_segments.append(
                {
                    "segment_id": segment_id,
                    "word": seg.get("word"),
                    "target_phoneme": target,
                    "target_phoneme_normalized": normalized_target,
                    "feedback_mode": mode,
                    "analysis_depth": context["feedback_policy"]["analysis_depth"],
                    "llm_required": context["feedback_policy"]["llm_required"],
                    "llm_context": context,
                    "llm_prompt": prompt_for_context(context),
                }
            )
        if out_segments:
            segment_count += len(out_segments)
            samples.append(
                {
                    "id": sample.get("id") or sample.get("sample_id") or sample.get("audio_id"),
                    "speaker_id": sample.get("speaker_id"),
                    "segments": out_segments,
                }
            )

    payload = {
        "schema_version": "direct_llm_feedback_inputs_v2",
        "description": (
            "Direct LLM feedback inputs. Focus phonemes from pronunciation_error.md get articulatory "
            "analysis. Non-focus matches are marked OK without an LLM call. Non-focus substitutions "
            "can use normal LLM analysis. Non-focus insertions/deletions use basic feedback only."
        ),
        "output_schema": OUTPUT_SCHEMA,
        "focus_phonemes": sorted(focus_phonemes),
        "target_phonemes": sorted(interested),
        "mdd_predictions": {
            "path": args.mdd_predictions,
            "loaded_segments": len(mdd_predictions),
        },
        "stage2_predictions": {
            "path": args.stage2_predictions,
            "loaded_segments": len(stage2_predictions),
        },
        "feedback_mode_counts": dict(sorted(mode_counts.items())),
        "num_samples": len(samples),
        "num_segments": segment_count,
        "samples": samples,
    }
    write_json(Path(args.output), payload)
    print(f"Wrote {args.output}")
    print(f"Samples: {len(samples)}")
    print(f"Segments: {segment_count}")
    print(f"Feedback modes: {json.dumps(payload['feedback_mode_counts'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
