import json
import random
import re
from collections import Counter

import numpy as np
import pandas as pd

from config import GROUPS, SEED

AUGMENT_PROTECTED_WORDS = {
    "african", "africa", "black", "asian", "jewish", "jew", "arab", "caucasian",
    "white", "hispanic", "latino", "latina", "indian", "islam", "muslim",
    "women", "woman", "female", "men", "man", "male", "lgbt", "gay", "lesbian",
    "trans", "immigrant", "migrant", "refugee",
}

SYNONYM_MAP = {
    "really": ["very", "truly"],
    "very": ["really", "quite"],
    "so": ["really", "very"],
    "just": ["simply", "only"],
    "maybe": ["perhaps"],
    "perhaps": ["maybe"],
    "think": ["believe"],
    "believe": ["think"],
    "people": ["folks", "persons"],
    "person": ["individual"],
    "everyone": ["everybody"],
    "some": ["a few"],
    "many": ["lots of"],
    "bad": ["awful", "terrible"],
    "awful": ["bad", "terrible"],
    "terrible": ["awful", "bad"],
    "good": ["fine", "great"],
    "great": ["good"],
    "stupid": ["dumb"],
    "dumb": ["stupid"],
    "crazy": ["insane"],
    "insane": ["crazy"],
    "stop": ["quit"],
    "quit": ["stop"],
    "go": ["leave"],
    "leave": ["go"],
    "back": ["away"],
    "always": ["constantly"],
    "never": ["not ever"],
    "today": ["nowadays"],
    "now": ["currently"],
    "because": ["since"],
    "cause": ["because"],
    "also": ["too"],
    "still": ["yet"],
    "cannot": ["can't"],
    "cant": ["can't"],
    "dont": ["don't"],
    "doesnt": ["doesn't"],
    "isnt": ["isn't"],
    "arent": ["aren't"],
    "wont": ["won't"],
}


def _protect_tokens(text):
    return re.findall(r"https?://\S+|@\w+|#\w+|\w+(?:'\w+)?|[^\w\s]", str(text), flags=re.UNICODE)


def _detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    return text.strip()


def _is_protected_token(tok):
    low = tok.lower().strip()
    return (
        low in AUGMENT_PROTECTED_WORDS
        or tok.startswith(("http", "@", "#"))
        or bool(re.fullmatch(r"[^\w\s]", tok))
        or len(low) <= 2
    )


def _match_case(src, replacement):
    if src.isupper():
        return replacement.upper()
    if src[:1].isupper():
        return replacement.capitalize()
    return replacement


def synonym_replacement(tokens, rng, max_replacements=2):
    out = tokens[:]
    candidates = [
        i for i, tok in enumerate(out)
        if not _is_protected_token(tok) and tok.lower() in SYNONYM_MAP
    ]
    rng.shuffle(candidates)
    n = min(max_replacements, len(candidates))
    for i in candidates[:n]:
        repl = rng.choice(SYNONYM_MAP[out[i].lower()])
        out[i] = _match_case(out[i], repl)
    return out


def random_deletion(tokens, rng, p=0.07):
    out = [tok for tok in tokens if _is_protected_token(tok) or rng.random() > p]
    return out if len(out) >= max(3, int(0.65 * len(tokens))) else tokens[:]


def random_swap(tokens, rng, n_swaps=1):
    out = tokens[:]
    candidates = [
        i for i in range(len(out) - 1)
        if not _is_protected_token(out[i]) and not _is_protected_token(out[i + 1])
    ]
    if not candidates:
        return out
    for _ in range(min(n_swaps, len(candidates))):
        i = rng.choice(candidates)
        out[i], out[i + 1] = out[i + 1], out[i]
    return out


def punctuation_noise(tokens, rng):
    out = tokens[:]
    if out and rng.random() < 0.5:
        if out[-1] in [".", "!", "?"]:
            out[-1] = rng.choice([".", "!", "!!", "..."])
        else:
            out.append(rng.choice([".", "!", "..."]))
    return out


def light_char_noise(tokens, rng, p=0.03):
    out = []
    for tok in tokens:
        if _is_protected_token(tok) or len(tok) < 5 or rng.random() > p:
            out.append(tok)
            continue
        chars = list(tok)
        i = rng.randrange(1, len(chars) - 1)
        op = rng.choice(["swap", "drop"])
        if op == "swap" and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop":
            chars.pop(i)
        out.append("".join(chars))
    return out


def augment_text(text, rng, methods=("synonym", "delete", "swap", "punct", "char")):
    tokens = _protect_tokens(text)
    if len(tokens) < 4:
        return str(text), "none_short"

    method = rng.choice(list(methods))
    if method == "synonym":
        aug = synonym_replacement(tokens, rng, max_replacements=2)
    elif method == "delete":
        aug = random_deletion(tokens, rng, p=0.07)
    elif method == "swap":
        aug = random_swap(tokens, rng, n_swaps=1)
    elif method == "punct":
        aug = punctuation_noise(tokens, rng)
    elif method == "char":
        aug = light_char_noise(tokens, rng, p=0.04)
    elif method == "combo":
        aug = synonym_replacement(tokens, rng, max_replacements=1)
        aug = random_deletion(aug, rng, p=0.04)
        aug = punctuation_noise(aug, rng)
    else:
        aug = tokens[:]

    text_aug = _detokenize(aug)
    if text_aug == str(text).strip():
        text_aug = _detokenize(punctuation_noise(tokens, rng))
        method = f"{method}+fallback_punct"
    return text_aug, method


def _do_augment(df_train, targets_to_augment, aug_factor, methods, seed, save_report_path):
    """Shared augmentation logic given a set of groups to augment."""
    rng = random.Random(seed)
    group_counts_before = {
        g: int(((df_train["label"] == 1) & df_train["targets_parsed"].apply(lambda t: g in t)).sum())
        for g in GROUPS
    }

    rows = []
    method_counts = Counter()

    for _, row in df_train.iterrows():
        if row["label"] != 1:
            continue
        group_targets = [t for t in row["targets_parsed"] if t in targets_to_augment]
        if not group_targets:
            continue
        for k in range(aug_factor):
            new_text, method = augment_text(row["text"], rng, methods=methods)
            if new_text == str(row["text"]).strip():
                continue
            new_row = row.copy()
            original_id = row.get("id", f"row_{len(rows)}")
            new_row["id"] = f"{original_id}_aug{k+1}"
            new_row["text"] = new_text
            new_row["augmentation_method"] = method
            new_row["original_id"] = original_id
            rows.append(new_row)
            method_counts[method] += 1

    df_aug = pd.DataFrame(rows)
    df_base = df_train.copy()
    df_base["augmentation_method"] = "original"
    df_base["original_id"] = df_base.get("id", pd.Series(range(len(df_base))))

    if len(df_aug) == 0:
        print("No rows augmented.")
        return df_base

    out = pd.concat([df_base, df_aug], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)

    group_counts_after = {
        g: int(((out["label"] == 1) & out["targets_parsed"].apply(lambda t: g in t)).sum())
        for g in GROUPS
    }

    print(f"\nAugmentation: +{len(df_aug)} samples ({len(df_train)} → {len(out)})")
    print("Methods:", dict(method_counts))

    if save_report_path is not None:
        report = {
            "n_train_original": int(len(df_train)),
            "n_augmented_added": int(len(df_aug)),
            "n_train_after": int(len(out)),
            "aug_factor": int(aug_factor),
            "methods": list(methods),
            "method_counts": dict(method_counts),
            "positive_group_counts_before": group_counts_before,
            "positive_group_counts_after": group_counts_after,
        }
        with open(save_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Augmentation report → {save_report_path}")

    return out


def augment_training_data(
    df_train,
    aug_factor=2,
    min_group_samples=200,
    seed=SEED,
    methods=("synonym", "delete", "swap", "punct", "char", "combo"),
    save_report_path=None,
):
    """Augments hate samples from groups with hate_count < min_group_samples."""
    group_counts = {
        g: int(((df_train["label"] == 1) & df_train["targets_parsed"].apply(lambda t: g in t)).sum())
        for g in GROUPS
    }
    targets_to_augment = {g for g, c in group_counts.items() if c < min_group_samples}

    print("\nGroups targeted for augmentation (count-based):")
    for g in sorted(targets_to_augment):
        print(f"  {g:<12} hate_count={group_counts[g]} < {min_group_samples}")

    return _do_augment(df_train, targets_to_augment, aug_factor, methods, seed, save_report_path)


def augment_training_data_disparity(
    df_train,
    aug_factor=3,
    disparity_threshold=1.5,
    seed=SEED,
    methods=("synonym", "delete", "swap", "punct", "char", "combo"),
    save_report_path=None,
):
    """
    Augments hate samples from groups where normal_count / hate_count > disparity_threshold.
    Targets groups where the model sees many more normal than hate examples, making
    hate detection harder for that group.
    """
    hate_counts = {
        g: max(1, ((df_train["label"] == 1) & df_train["targets_parsed"].apply(lambda t: g in t)).sum())
        for g in GROUPS
    }
    normal_counts = {
        g: max(1, ((df_train["label"] == 0) & df_train["targets_parsed"].apply(lambda t: g in t)).sum())
        for g in GROUPS
    }
    disparity = {g: normal_counts[g] / hate_counts[g] for g in GROUPS}

    targets_to_augment = {g for g, d in disparity.items() if d > disparity_threshold}

    print(f"\nGroups targeted for augmentation (disparity normal/hate > {disparity_threshold}):")
    for g in sorted(GROUPS, key=lambda g: -disparity[g]):
        marker = "← augmented" if g in targets_to_augment else ""
        print(f"  {g:<12} hate={hate_counts[g]:4d}  normal={normal_counts[g]:4d}  ratio={disparity[g]:.2f}  {marker}")

    return _do_augment(df_train, targets_to_augment, aug_factor, methods, seed, save_report_path)
