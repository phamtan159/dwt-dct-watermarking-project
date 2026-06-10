import argparse
import json
import os
import re
import urllib.request
from pathlib import Path


DEFAULT_ATTRIBUTES = Path("data/final/segment_attributes.json")
DEFAULT_LABEL_MAP = Path("data/label_map.json")
DEFAULT_REVIEW_DIR = Path("data/labels/review")

THETA = "\u03b8"
DH = "\u00f0"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_phone(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def sample_rel_path(sample_id):
    return Path(*str(sample_id).replace("\\", "/").split("/")).with_suffix(".json")


def group_for_sample(sample, taxonomy):
    metadata = sample.get("metadata") or {}
    phonemes = taxonomy.get("phonemes") or {}
    candidates = [
        metadata.get("phone_folder"),
        metadata.get("target_group"),
        metadata.get("target_phoneme"),
    ]
    for candidate in candidates:
        if candidate is not None and str(candidate) in phonemes:
            return str(candidate), phonemes[str(candidate)]

    target = metadata.get("target_phoneme")
    for key, group in phonemes.items():
        if isinstance(group, dict) and group.get("phoneme") == target:
            return key, group
    return None, None


def interested_standard_phones(group_key, group):
    key = str(group_key or "")
    phone = str((group or {}).get("phoneme") or "")
    if key == "theta_group":
        return {THETA}
    if key == "dh_group":
        return {DH}
    if key == "theta_medial_group" or key == f"{THETA}_medial" or key.endswith("_medial"):
        return {THETA}
    if key == "dh_er_group" or key == f"{DH}_er" or key.endswith("_er"):
        return {DH, "r", "\u025a", "\u025d", "\u025a\u02de"}
    if key in {"d", "t", "p", THETA, DH}:
        return {key}
    if phone and "_" not in phone:
        return {phone}
    return {key}


def code_to_index(index_map):
    return {str(code): str(index) for index, code in (index_map or {}).items()}


def category_code_for_review(review, code):
    return code_to_index(review.get("category_index_map")).get(code, code)


def label_code_for_review(review, code):
    return code_to_index(review.get("label_index_map")).get(code, code)


def evidence_code_for_review(review, code):
    return code_to_index(review.get("primary_evidence_index_map")).get(code, code)


def severity_code_for_review(review, code_or_index):
    severity_map = review.get("severity_index_map") or {}
    reverse = {str(value): str(key) for key, value in severity_map.items()}
    return reverse.get(str(code_or_index), str(code_or_index))


def feature_confidence(seg):
    prediction = seg.get("speech_attribute_prediction") or {}
    confidence = prediction.get("feature_confidence") or {}
    return confidence if isinstance(confidence, dict) else {}


def segment_neighbors(sample, seg):
    segments = sample.get("segments") or []
    index = None
    for i, item in enumerate(segments):
        if item.get("id") == seg.get("id"):
            index = i
            break
    if index is None:
        return None, None
    previous_seg = segments[index - 1] if index > 0 else None
    next_seg = segments[index + 1] if index + 1 < len(segments) else None
    return previous_seg, next_seg


def phone_summary(seg):
    if not isinstance(seg, dict):
        return None
    return {
        "expected": seg.get("target_phoneme") or seg.get("phoneme_standard") or seg.get("standard_phone"),
        "word": seg.get("word"),
        "start": seg.get("start"),
        "end": seg.get("end"),
    }


def wavlm_compact(seg, side):
    audio = seg.get("audio_attributes") or {}
    payload = audio.get(side) or seg.get(f"wavlm_{side}_attributes") or {}
    if not isinstance(payload, dict):
        return {}
    return {
        "embedding_norm": payload.get("embedding_norm"),
        "embedding_std": payload.get("embedding_std"),
        "activation_min": payload.get("activation_min"),
        "activation_max": payload.get("activation_max"),
        "num_frames": payload.get("num_frames"),
    }


def numeric_delta(left, right):
    try:
        return round(float(right) - float(left), 6)
    except (TypeError, ValueError):
        return None


def relevant_context(sample, seg, review):
    conf = feature_confidence(seg)
    visual = seg.get("visual_attributes") or {}
    mediapipe = visual.get("mediapipe") or {}
    mouth_clip = visual.get("mouth_clip") or {}
    previous_seg, next_seg = segment_neighbors(sample, seg)
    standard_wavlm = wavlm_compact(seg, "standard")
    real_wavlm = wavlm_compact(seg, "real")

    duration = None
    try:
        duration = round(max(0.0, float(seg.get("end")) - float(seg.get("start"))), 6)
    except (TypeError, ValueError):
        pass

    fricative = conf.get("fricative")
    plosive = conf.get("plosive")
    frication_stop_margin = numeric_delta(plosive, fricative)
    return {
        "task": "suggest a pronunciation label using only these acoustic/visual/attribute signals; observed phoneme and auto labels are intentionally hidden",
        "target": {
            "word": seg.get("word"),
            "expected_phoneme": seg.get("target_phoneme") or seg.get("phoneme_standard") or seg.get("standard_phone"),
            "segment_id": seg.get("id"),
        },
        "acoustic_requested_features": {
            "frication_vs_stop": {
                "fricative_confidence": fricative,
                "plosive_or_stop_confidence": plosive,
                "fricative_minus_plosive": frication_stop_margin,
            },
            "aspiration_proxy": {
                "note": "No direct aspiration detector is available; use duration plus WavLM activation/embedding proxies only.",
                "duration_seconds": duration,
                "activation_max_real": real_wavlm.get("activation_max"),
                "embedding_std_real": real_wavlm.get("embedding_std"),
            },
            "duration": {
                "seconds": duration,
                "num_wavlm_frames_real": real_wavlm.get("num_frames"),
            },
            "vowel_quality": {
                key: conf.get(key)
                for key in (
                    "vowel",
                    "front",
                    "back",
                    "central",
                    "high",
                    "mid",
                    "low",
                    "round",
                    "monophthong",
                    "diphthong",
                )
                if key in conf
            },
            "transition_between_phonemes": {
                "previous_expected": phone_summary(previous_seg),
                "current_expected": phone_summary(seg),
                "next_expected": phone_summary(next_seg),
                "word_index": seg.get("word_index"),
                "phoneme_index_in_word": seg.get("phoneme_index_in_word"),
            },
            "energy_spectral_shape_proxy": {
                "wavlm_standard": standard_wavlm,
                "wavlm_real": real_wavlm,
                "delta_embedding_norm_real_minus_standard": numeric_delta(
                    standard_wavlm.get("embedding_norm"), real_wavlm.get("embedding_norm")
                ),
                "delta_embedding_std_real_minus_standard": numeric_delta(
                    standard_wavlm.get("embedding_std"), real_wavlm.get("embedding_std")
                ),
            },
        },
        "speech_attribute_confidence": {
            key: conf.get(key) for key in sorted(conf)
        },
        "visual_summary": {
            "face_detection_rate": mediapipe.get("face_detection_rate"),
            "tongue_landmarks_available": mediapipe.get("tongue_landmarks_available"),
            "mouth_opening": ((mediapipe.get("stats") or {}).get("mouth_opening") or {}).get("mean"),
            "labiodental_contact_proxy": ((mediapipe.get("stats") or {}).get("labiodental_contact_proxy") or {}).get("mean"),
            "motion_mean": mouth_clip.get("motion_mean"),
        },
        "allowed_category_index_map": review.get("category_index_map"),
        "allowed_label_index_map": review.get("label_index_map"),
        "severity_index_map": review.get("severity_index_map"),
        "primary_evidence_index_map": review.get("primary_evidence_index_map"),
    }


def blank_label(label):
    if not isinstance(label, dict):
        return True
    return not str(label.get("category") or "").strip() and not str(label.get("label") or "").strip()


def has_feature_error(seg, attribute):
    for item in seg.get("feature_errors") or []:
        if str(item.get("attribute") or "").lower() == attribute:
            return True
    return False


def observed(seg):
    return normalize_phone(seg.get("observed_phoneme") or seg.get("phoneme_real"))


def expected(seg):
    return normalize_phone(seg.get("target_phoneme") or seg.get("phoneme_standard") or seg.get("standard_phone"))


def category_exists(review, code):
    return code in set(code_to_index(review.get("category_index_map")))


def label_exists(review, code):
    return code in set(code_to_index(review.get("label_index_map")))


def suggestion(review, category, label, severity, primary_evidence, note):
    if not category_exists(review, category):
        category = "OTHER"
    if not label_exists(review, label):
        label = "OTHER"
    return {
        "category": category_code_for_review(review, category),
        "label": label_code_for_review(review, label),
        "severity": severity_code_for_review(review, severity),
        "primary_evidence": evidence_code_for_review(review, primary_evidence),
        "note": note,
    }


def heuristic_suggest(sample, seg, review, group_key):
    std = expected(seg)
    conf = feature_confidence(seg)
    duration = None
    try:
        duration = max(0.0, float(seg.get("end")) - float(seg.get("start")))
    except (TypeError, ValueError):
        pass

    fricative = float(conf.get("fricative") or 0.0)
    plosive = float(conf.get("plosive") or 0.0)
    dental = float(conf.get("dental") or 0.0)
    voiced = float(conf.get("voiced") or 0.0)
    consonant = float(conf.get("consonant") or 0.0)
    vowel = float(conf.get("vowel") or 0.0)
    short_duration = duration is not None and duration <= 0.045
    weak_duration = duration is not None and duration <= 0.065

    if std == THETA:
        if short_duration or fricative < 0.6:
            label = "th_reduced_to_breath" if fricative > 0.25 else "th_omitted_silent"
            severity = "3" if short_duration and fricative < 0.5 else "2"
            return suggestion(review, "th_weak_or_omitted", label, severity, "audio_visual", "suggested_by=heuristic; feature-only theta weak/short/low-frication")
        if plosive >= 0.72 or (weak_duration and plosive >= 0.62):
            label = "th_to_t_short_hard_burst" if weak_duration else "th_to_t_alveolar_stop"
            return suggestion(review, "th_to_t", label, "2", "audio_visual", "suggested_by=heuristic; feature-only plosive/stop evidence for theta")
        if dental >= 0.7 and fricative >= 0.7 and plosive < 0.72:
            return suggestion(review, "OK", "OK", "0", "none", "suggested_by=heuristic; feature-only dental/fricative evidence looks acceptable")
        return suggestion(review, "th_weak_or_omitted", "th_weak_low_frication", "1", "audio_visual", "suggested_by=heuristic; feature-only borderline theta")

    if std == DH:
        if plosive >= 0.72 and voiced >= 0.5:
            return suggestion(review, "dh_to_d", "dh_to_d_alveolar_stop", "2", "audio_visual", "suggested_by=heuristic; feature-only voiced plosive/stop evidence for dh")
        if plosive >= 0.72 and voiced < 0.5:
            return suggestion(review, "dh_to_d", "dh_to_t_devoiced_stop", "2", "audio_visual", "suggested_by=heuristic; feature-only devoiced stop evidence for dh")
        if short_duration or fricative < 0.55:
            return suggestion(review, "dh_weak_or_omitted", "dh_omitted_silent", "3", "audio_visual", "suggested_by=heuristic; feature-only dh weak/omitted")
        if dental >= 0.7 and fricative >= 0.65 and voiced >= 0.5:
            return suggestion(review, "OK", "OK", "0", "none", "suggested_by=heuristic; feature-only dental/fricative/voiced evidence looks acceptable")
        return suggestion(review, "dh_weak_or_omitted", "dh_weak_low_frication", "1", "audio_visual", "suggested_by=heuristic; feature-only borderline dh")

    if std == "d":
        if short_duration or consonant < 0.45:
            return suggestion(review, "final_d_weak_or_omitted", "final_d_omitted", "3", "audio", "suggested_by=heuristic; feature-only final d omitted/weak")
        if voiced < 0.5 and consonant >= 0.45:
            return suggestion(review, "final_d_devoiced_to_t", "final_d_devoiced_clear_t", "2", "audio", "suggested_by=heuristic; feature-only final d devoicing evidence")
    if std == "t":
        if short_duration or consonant < 0.45:
            return suggestion(review, "final_t_weak_or_omitted", "final_t_omitted", "3", "audio", "suggested_by=heuristic; feature-only final t weak/omitted")
        if weak_duration:
            return suggestion(review, "final_t_weak_or_omitted", "final_t_unreleased_weak", "2", "audio", "suggested_by=heuristic; feature-only final t short/unreleased proxy")
    if std == "p":
        if short_duration or consonant < 0.45:
            return suggestion(review, "final_p_weak_or_omitted", "final_p_omitted", "3", "audio", "suggested_by=heuristic; feature-only final p weak/omitted")
        if weak_duration:
            return suggestion(review, "final_p_weak_or_omitted", "final_p_unreleased_weak", "2", "audio", "suggested_by=heuristic; feature-only final p short/unreleased proxy")

    if consonant >= 0.45 or vowel < 0.55:
        return suggestion(review, "OK", "OK", "0", "none", "suggested_by=heuristic; feature-only target evidence looks acceptable")
    return suggestion(review, "OTHER", "OTHER", "2", "audio", "suggested_by=heuristic; feature-only unsupported/borderline target phoneme")


def openai_suggest(context, model):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --provider openai.")
    if not model:
        raise RuntimeError("--model is required for --provider openai.")

    system = (
        "You are a pronunciation-labeling assistant. Return JSON only. "
        "Choose category and label only from the provided allowed maps. "
        "Use numeric indexes if possible. The label is a suggestion, not ground truth."
    )
    user = (
        "Suggest a human_label for this phoneme segment. "
        "Return keys: category, label, severity, primary_evidence, note.\n"
        + json.dumps(context, ensure_ascii=False)
    )
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        raise RuntimeError(f"OpenAI response did not contain JSON: {content}")
    parsed = json.loads(match.group(0))
    return {
        "category": parsed.get("category", ""),
        "label": parsed.get("label", ""),
        "severity": parsed.get("severity", ""),
        "primary_evidence": parsed.get("primary_evidence", ""),
        "note": f"suggested_by=openai; {parsed.get('note', '')}".strip(),
    }


def normalize_llm_suggestion(raw, review):
    category = str(raw.get("category") or "").strip()
    label = str(raw.get("label") or "").strip()
    severity = str(raw.get("severity") or "").strip()
    evidence = str(raw.get("primary_evidence") or "").strip()
    note = str(raw.get("note") or "").strip()

    category_reverse = code_to_index(review.get("category_index_map"))
    label_reverse = code_to_index(review.get("label_index_map"))
    severity_reverse = {str(value): str(key) for key, value in (review.get("severity_index_map") or {}).items()}
    evidence_reverse = code_to_index(review.get("primary_evidence_index_map"))

    return {
        "category": category_reverse.get(category, category),
        "label": label_reverse.get(label, label),
        "severity": severity_reverse.get(severity, severity),
        "primary_evidence": evidence_reverse.get(evidence, evidence),
        "note": note,
    }


def build_attribute_index(attributes):
    samples = {}
    for sample in attributes.get("samples", []):
        sample_id = sample.get("id") or sample.get("video_id") or sample.get("sample_id")
        if not sample_id:
            continue
        segments = {seg.get("id"): seg for seg in sample.get("segments", []) if seg.get("id")}
        samples[sample_id] = (sample, segments)
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Suggest labels for interested phoneme segments and write them into label review JSON files."
    )
    parser.add_argument("--attributes", type=Path, default=DEFAULT_ATTRIBUTES)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--speaker")
    parser.add_argument("--sample-id")
    parser.add_argument("--provider", choices=("heuristic", "openai"), default="heuristic")
    parser.add_argument("--model", help="OpenAI model name for --provider openai.")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--promote-to-human-label",
        action="store_true",
        help="Write suggestions into human_label instead of suggested_label. Use only if you accept pseudo-labels as train labels.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    attributes = load_json(args.attributes)
    taxonomy = load_json(args.label_map)
    attribute_index = build_attribute_index(attributes)

    suggested = 0
    skipped_existing = 0
    skipped_non_target = 0
    touched_files = 0

    for sample_id, (sample, segments_by_id) in sorted(attribute_index.items()):
        if args.speaker and sample.get("speaker_id") != args.speaker:
            continue
        if args.sample_id and sample_id != args.sample_id:
            continue

        group_key, group = group_for_sample(sample, taxonomy)
        if not group:
            continue
        interested_phones = interested_standard_phones(group_key, group)

        review_path = args.review_dir / sample_rel_path(sample_id)
        if not review_path.exists():
            continue
        review = load_json(review_path)
        changed = False

        for review_segment in review.get("segments", []):
            phoneme = review_segment.get("phoneme") or {}
            segment_id = phoneme.get("segment_id")
            seg = segments_by_id.get(segment_id)
            if not seg:
                continue

            std = seg.get("target_phoneme") or seg.get("phoneme_standard") or seg.get("standard_phone") or ""
            if std not in interested_phones:
                skipped_non_target += 1
                continue

            target_key = "human_label" if args.promote_to_human_label else "suggested_label"
            current = review_segment.get(target_key) or {}
            if not args.overwrite_existing and not blank_label(current):
                skipped_existing += 1
                continue

            if args.provider == "openai":
                context = relevant_context(sample, seg, review)
                raw_suggestion = openai_suggest(context, args.model)
                human_label = normalize_llm_suggestion(raw_suggestion, review)
            else:
                human_label = heuristic_suggest(sample, seg, review, group_key)

            if args.dry_run:
                print(json.dumps({
                    "sample_id": sample_id,
                    "segment_id": segment_id,
                    "expected": std,
                    "target_field": target_key,
                    "suggestion": human_label,
                }, ensure_ascii=False))
            else:
                review_segment[target_key] = human_label
                changed = True
            suggested += 1

        if changed:
            write_json(review_path, review)
            touched_files += 1

    print(f"Suggested labels: {suggested}")
    print(f"Skipped existing suggestions/labels: {skipped_existing}")
    print(f"Skipped non-target segments: {skipped_non_target}")
    print(f"Touched files: {touched_files}")
    if args.dry_run:
        print("Dry run only; no files were modified.")


if __name__ == "__main__":
    main()
