# evaluate_speech_to_text.py to evaluate STT results against a reference transcript, computing WER, CER, TER, and BERTScore-F1.

import re, csv, os
from typing import Dict

import jiwer
from bert_score import score as bscore


#Text loading and cleaning to handle various formatting issues in the input files
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    def _is_divider(s: str) -> bool:
        if not s:
            return False
        if len(set(s)) == 1 and s[0] in "-=~_":
            return True
        # Unicode box-drawing characters (U+2500–U+257F)
        if all(0x2500 <= ord(c) <= 0x257F for c in s):
            return True
        return False

    cleaned = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("[CHUNK") or _is_divider(s):
            continue
        cleaned.append(s)

    return " ".join(cleaned)

# Text normalization
def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


#Evaluation metrics for WER, CER, TER and bert_score for BERTScore-F1

#Word Error Rate (WER)
def compute_wer(hypothesis: str, reference: str) -> Dict:
    score = jiwer.wer(clean_text(reference), clean_text(hypothesis))
    result = {"wer_pct": round(score * 100, 2)}
    print(f"   WER          : {result['wer_pct']:.2f}%")
    return result

#Character Error Rate (CER)
def compute_cer(hypothesis: str, reference: str) -> Dict:
    score = jiwer.cer(clean_text(reference), clean_text(hypothesis))
    result = {"cer_pct": round(score * 100, 2)}
    print(f"   CER          : {result['cer_pct']:.2f}%")
    return result

# Translation Edit Rate (TER)
def compute_ter(hypothesis: str, reference: str) -> Dict:
    def _norm(text: str) -> str:
        text = re.sub(r"""([.,!?;:'"()])""", r" \1 ", text)
        return " ".join(text.split()).strip()

    score = jiwer.wer(_norm(reference), _norm(hypothesis))
    result = {"ter_pct": round(score * 100, 2)}
    print(f"   TER          : {result['ter_pct']:.2f}%")
    return result

# BERTScore F1
def compute_bertscore(
    hypothesis: str,
    reference: str,
    model_type: str = "distilbert-base-uncased",
) -> Dict:
    print("   BERTScore-F1 : computing")
    _, _, F1 = bscore(
        [hypothesis], [reference],
        lang="en",
        model_type=model_type,
        verbose=False,
        rescale_with_baseline=True,
    )
    result = {"bertscore_f1": round(F1.mean().item(), 4)}
    print(f"   BERTScore-F1 : {result['bertscore_f1']:.4f}")
    return result


# Main evaluation function
def evaluate_stt(
    hyp_path:      str,
    ref_path:      str,
    model_name:    str  = None,
    run_bertscore: bool = True,
) -> Dict:
    #Evaluate one hypothesis file against a reference file
    hypothesis = load_text(hyp_path)
    reference  = load_text(ref_path)

    if not hypothesis.strip():
        raise ValueError(f"Empty hypothesis: {hyp_path}")
    if not reference.strip():
        raise ValueError(f"Empty reference: {ref_path}")

    name = model_name or os.path.splitext(os.path.basename(hyp_path))[0]

    print("\n" + "=" * 52)
    print(f"  MODEL : {name}")
    print(f"  Hyp   : {len(hypothesis.split())} words")
    print(f"  Ref   : {len(reference.split())} words")
    print("=" * 52)

    r = {"model": name}
    r.update(compute_wer(hypothesis, reference))
    r.update(compute_cer(hypothesis, reference))
    r.update(compute_ter(hypothesis, reference))
    if run_bertscore:
        r.update(compute_bertscore(hypothesis, reference))

    _print_summary(r)
    return r


# Save results to CSV and summary of the evaluation results

def _save_csv(result: Dict, path: str) -> None:
    columns = ["model", "wer_pct", "cer_pct", "ter_pct", "bertscore_f1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({col: result.get(col, "") for col in columns})
    print(f"\nCSV   → {path}")


def _print_summary(r: Dict) -> None:
    print("\n" + "=" * 52)
    print("  SUMMARY")
    print("=" * 52)
    for label, key, unit in [
        ("WER",          "wer_pct",      "%"),
        ("CER",          "cer_pct",      "%"),
        ("TER",          "ter_pct",      "%"),
        ("BERTScore-F1", "bertscore_f1",  ""),
    ]:
        val = r.get(key)
        if val is not None:
            print(f"  {label:<16}: {val}{unit}")
    print("=" * 52)

results = evaluate_stt(
    hyp_path   = "knowledge_base.txt",
    ref_path   = "ground_truth.txt",
)

_save_csv(results, "eval_results.csv")