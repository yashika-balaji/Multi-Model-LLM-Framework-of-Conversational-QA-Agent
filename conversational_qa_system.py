
# conversational_qa_pipeline.py (QA Pipeline Implmentation+ Evaluation + LLM-as-a-Judge + RAGAS Evaluation)
# This pipeline runs with knowledge_base.txt and Supplementary_Database.xlsx as two input data sources


# Dependency installation — run once per environment

# System Configurations
import subprocess, sys
import os, re, gc, time, signal, logging, warnings
import numpy as np
import pandas as pd
import torch
from collections import Counter

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("RAG")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "."
log.info(f"Device: {DEVICE}  |  numpy: {np.__version__}")

# API keys to execute LLM-as-a-judge and RAGAS evaluation — set these in Colab userdata before running the pipeline (essential to run)
from google.colab import userdata
OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
HF_TOKEN       = userdata.get("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from huggingface_hub import login
login(token=HF_TOKEN)

# Configuration
KNOWLEDGE_BASE     = "knowledge_base.txt"    # result of STT processing — already chunked with [CHUNK n] markers
SUPPLEMENTARY_XLSX = "Supplementary_Database.xlsx"  # result of web data extraction pipeline
RECHUNK_WINDOW     = 3
RECHUNK_OVERLAP    = 1
MIN_CHUNK_WORDS    = 20
MAX_CHUNK_WORDS    = 200
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CROSS_ENCODER      = "cross-encoder/ms-marco-MiniLM-L-12-v2"
TOP_K_BIENCODER    = 10
TOP_K_RERANK       = 4
BI_THRESHOLD       = 0.25
CE_THRESHOLD       = -3.0
GATE_MODE          = "either"
NORMALIZE          = True
MAX_SEQ_LENGTH     = 256
MAX_NEW_TOKENS     = 150
MAX_INPUT_TOKENS   = 1536
REPETITION_PENALTY = 1.1

SYSTEM_PROMPT = (
    "You are a university module advisor. "
    "Answer questions ONLY based on the provided context about module. "
    "If the answer is not in the context, say: 'This information is not "
    "available in the module knowledge base.' Keep answers concise."
)

# Selected models based on comparative evaluation results
MODELS = [
    "deepset/roberta-base-squad2",
    "Qwen/Qwen2.5-3B-Instruct",
]

# Ground truth QA pairs
# T = transcript-based, E = excel-based, C = composite, I = irrelevant
GROUND_TRUTH = [
    {"id":"T01","q":"What is the abbreviation of Data Management for Engineering Applications?","a":"DMEA","rel":True},
    {"id":"T02","q":"Who teaches the DMEA module?","a":"Eike Schalleen","rel":True},
    {"id":"T03","q":"Which research group does the DMEA lecturer belong to?","a":"database group of Professor Saake","rel":True},
    {"id":"T04","q":"Does DMEA have any prerequisites?","a":"no prerequisites","rel":True},
    {"id":"T05","q":"What is the best strategy to get a good grade in DMEA?","a":"start preparing from the first day of the lecture","rel":True},
    {"id":"T06","q":"What is the most common mistake students make in DMEA?","a":"not attending lectures and letting gaps pile up","rel":True},
    {"id":"T07","q":"What courses can be taken after DMEA?","a":"advanced database models","rel":True},
    {"id":"T08","q":"Is DMEA a good preparation for a master thesis?","a":"not the course to take as preparation for master thesis","rel":True},
    {"id":"T09","q":"What kind of data does DMEA cover?","a":"CAD Computer Aided Design and Electrical Engineering","rel":True},
    {"id":"T10","q":"How advanced is the DMEA course?","a":"relatively basic course","rel":True},
    {"id":"E01","q":"How many credit points does DMEA have?","a":"6","rel":True},
    {"id":"E02","q":"In which semester is DMEA offered?","a":"Winter","rel":True},
    {"id":"E03","q":"What is the examination type for DMEA?","a":"Written exam 120 min","rel":True},
    {"id":"E04","q":"What language is DMEA taught in?","a":"english","rel":True},
    {"id":"E05","q":"What is the workload for DMEA?","a":"56 contact hours + 94 h independent study + 30 h lab","rel":True},
    {"id":"C01","q":"Who teaches DMEA and how many credits is it worth?","a":"Eike Schalleen and 6 credit points","rel":True},
    {"id":"C02","q":"What topics does DMEA cover and what is the exam format?","a":"CAD Computer Aided Design and Electrical Engineering with written exam","rel":True},
    {"id":"I01","q":"What is the weather in Berlin?","a":"Irrelevant Question","rel":False},
    {"id":"I02","q":"Who won the FIFA World Cup 2022?","a":"Irrelevant Question","rel":False},
    {"id":"I03","q":"How do I cook pasta?","a":"Irrelevant Question","rel":False},
    {"id":"I04","q":"What is the capital of Austria?","a":"Irrelevant Question","rel":False},
    {"id":"I05","q":"Explain quantum computing.","a":"Irrelevant Question","rel":False},
    {"id":"I06","q":"How does photosynthesis work?","a":"Irrelevant Question","rel":False},
    {"id":"I07","q":"Best restaurants in Berlin?","a":"Irrelevant Question","rel":False},
    {"id":"I08","q":"What programming language for Android?","a":"Irrelevant Question","rel":False},
]
REL_GT    = [g for g in GROUND_TRUTH if g["rel"]]
IRR_GT    = [g for g in GROUND_TRUTH if not g["rel"]]
TOTAL_REL = len(REL_GT)
TOTAL_IRR = len(IRR_GT)

# Data loading for transcript data source
def load_stt_chunks(filepath):
    if not os.path.exists(filepath):
        log.error(f"NOT FOUND: {filepath}"); return []
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    matches = re.findall(r'\[CHUNK (\d+)\]\n(.*?)(?=\[CHUNK|\Z)', text, re.DOTALL)
    raw = []
    for cid, content in matches:
        content = content.strip()
        if content:
            raw.append({"stt_id": int(cid), "text": content, "words": len(content.split())})
    log.info(f"Loaded {len(raw)} STT chunks from {filepath}")
    return raw

# Data rechunking for retrieval — sliding window with deduplication and length filtering
def rechunk_for_retrieval(raw, window=3, overlap=1, min_w=20, max_w=200):
    stride = max(1, window - overlap)
    chunks = []
    for i in range(0, len(raw), stride):
        wc    = raw[i:i+window]
        text  = " ".join(c["text"] for c in wc)
        sents = re.split(r'(?<=[.!?])\s+', text)
        seen, unique = set(), []
        for s in sents:
            sc = s.strip().lower()
            if sc and sc not in seen:
                seen.add(sc); unique.append(s.strip())
        text   = " ".join(unique)
        wcount = len(text.split())
        if wcount < min_w: continue
        if wcount > max_w: text = " ".join(text.split()[:max_w]); wcount = max_w
        chunks.append({"text":text,"source":"transcript",
                       "chunk_id":f"t_{len(chunks):02d}","words":wcount})
    log.info(f"Re-chunked: {len(raw)} → {len(chunks)} chunks")
    return chunks

# Data loading and chunking for Excel data source — creates metadata, content, and unavailability chunks per course
def load_excel_as_chunks(filepath):
    if not os.path.exists(filepath):
        log.warning(f"NOT FOUND: {filepath}"); return []
    df = pd.read_excel(filepath, engine="openpyxl")
    df.columns = df.columns.str.strip()
    meta = [
        ("Course Name","Course name"),("Abbreaviation of the course name","Abbreviation"),
        ("Lecturer","Lecturer"),("Responsibility","Responsible professor"),
        ("Credit Points","Credit points"),("Semester","Semester offered"),
        ("Term","Starting term"),("Duration of the course","Duration"),
        ("Language","Language of instruction"),("Level","Level"),
        ("Type of examination","Examination type"),
        ("Teaching method / lecture hours per week (SWS)","Teaching method"),
        ("Prerequisites according to examination regulations","Formal prerequisites"),
    ]
    cont = [
        ("Overall Content","Course content"),("Intended Learning Outcomes","Learning outcomes"),
        ("Workload","Workload breakdown"),("Applicability in Curriculum","Applicable in"),
        ("Classes","Class types"),
    ]
    trackable = [
        ("Pre-examination requirements","Pre-examination requirements"),
        ("Recommended prerequisites","Recommended prerequisites"),
        ("Media","Media and tools"),("Literature","Recommended literature"),
    ]
    chunks = []
    for idx, row in df.iterrows():
        course = row.get("Course Name","Unknown")
        abbrev = row.get("Abbreaviation of the course name","")
        parts  = []
        for col, label in meta:
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "keine":
                    parts.append(f"{label} is {str(val).strip()}")
        if parts:
            text = f"Course information for {course} ({abbrev}): " + "; ".join(parts) + "."
            chunks.append({"text":text,"source":"excel_metadata",
                           "chunk_id":f"e_meta_{idx}","words":len(text.split())})
        parts = []
        for col, label in cont:
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{label}: {str(val).strip()}")
        if parts:
            text = f"Detailed information for {course} ({abbrev}): " + "; ".join(parts) + "."
            chunks.append({"text":text,"source":"excel_content",
                           "chunk_id":f"e_cont_{idx}","words":len(text.split())})
        unavail = []
        for col, label in trackable + meta:
            if col in df.columns:
                val = row.get(col)
                if pd.isna(val) or not str(val).strip() or str(val).strip().lower() == "keine":
                    unavail.append(label)
        if unavail:
            text = (f"Data availability notice for {course} ({abbrev}): "
                    f"The following information is not available: {', '.join(unavail)}.")
            chunks.append({"text":text,"source":"excel_unavailable",
                           "chunk_id":f"e_na_{idx}","words":len(text.split())})
    log.info(f"Created {len(chunks)} Excel chunks")
    return chunks

# Build knowledge base (transcript + supplementary data sources) and chunk pool for retrieval
log.info("Loading knowledge base")
raw_stt           = load_stt_chunks(KNOWLEDGE_BASE)
transcript_chunks = rechunk_for_retrieval(raw_stt, RECHUNK_WINDOW, RECHUNK_OVERLAP)
excel_chunks      = load_excel_as_chunks(SUPPLEMENTARY_XLSX)
all_chunks        = transcript_chunks + excel_chunks
log.info(f"Transcript: {len(transcript_chunks)}  Excel: {len(excel_chunks)}  Total: {len(all_chunks)}")


# Build retrieval engine: bi-encoder for dense retrieval + cross-encoder for reranking
from sentence_transformers import SentenceTransformer, util, CrossEncoder

log.info("Loading bi-encoder and cross-encoder")
bi_enc = SentenceTransformer(EMBEDDING_MODEL)
bi_enc.max_seq_length = MAX_SEQ_LENGTH
ce_model = CrossEncoder(CROSS_ENCODER)

log.info("Encoding chunk pool for retrieval")
emb_all = bi_enc.encode([c["text"] for c in all_chunks],
                         convert_to_tensor=True, normalize_embeddings=NORMALIZE,
                         show_progress_bar=True)
log.info(f"Retrieval engine — {emb_all.shape[0]} chunks indexed")

# Retrieval
def retrieve_full(query, chunks, embeddings):
    """Bi-encoder retrieval → Excel injection → cross-encoder reranking"""
    q_emb = bi_enc.encode(query, convert_to_tensor=True, normalize_embeddings=NORMALIZE)
    cos   = util.cos_sim(q_emb, embeddings)[0]
    k     = min(TOP_K_BIENCODER, len(chunks))
    top   = torch.topk(cos, k=k)
    cands = [{**chunks[idx.item()], "bi_score": round(score.item(), 4)}
             for score, idx in zip(top.values, top.indices)]
    seen  = {c["chunk_id"] for c in cands}

    # Inject top-3 Excel chunks if none in top-K
    missing_excel = [
        {**chunk, "bi_score": round(cos[i].item(), 4)}
        for i, chunk in enumerate(chunks)
        if chunk["source"].startswith("excel") and chunk["chunk_id"] not in seen
    ]
    missing_excel.sort(key=lambda x: x["bi_score"], reverse=True)
    cands.extend(missing_excel[:3])

    # Pre-filter before cross-encoder
    ce_cands = [c for c in cands if c["bi_score"] >= 0.10]
    if ce_cands:
        scores = ce_model.predict([[query, c["text"]] for c in ce_cands], batch_size=16)
        for c, s in zip(ce_cands, scores):
            c["ce_score"] = round(float(s), 4)

    # Rerank with cross-encoder — transcript chunks first, then Excel, both sorted by CE score
    for c in cands:
        if "ce_score" not in c:
            c["ce_score"] = None

    tr_top = sorted([c for c in cands if c["source"] == "transcript"],
                    key=lambda x: x["ce_score"] if x["ce_score"] is not None else float("-inf"),
                    reverse=True)
    ex_top = sorted([c for c in cands if c["source"].startswith("excel")],
                    key=lambda x: x["ce_score"] if x["ce_score"] is not None else float("-inf"),
                    reverse=True)
    final = []
    if ex_top and TOP_K_RERANK >= 2: final.append(ex_top[0])
    for c in tr_top:
        if len(final) >= TOP_K_RERANK: break
        final.append(c)
    return final[:TOP_K_RERANK]

# Dual-threshold OR gate for relevance filtering — rejects only if BOTH bi-encoder and cross-encoder fail thresholds
def check_relevance(results, use_filter=True):
    if not use_filter: return True, "gate disabled"
    if not results:    return False, "no results"
    best = results[0]
    bi   = best.get("bi_score") or 0
    ce   = best.get("ce_score")
    if GATE_MODE == "either":
        if bi < BI_THRESHOLD and (ce is None or ce < CE_THRESHOLD):
            return False, "both below threshold"
    ce_str = f"{ce:.3f}" if ce is not None else "n/a"
    return True, f"relevant (bi={bi:.3f} ce={ce_str})"

# Evaluation metrics for relevant questions — token-level F1, semantic similarity, and BERTScore F1
def normalize_text(t):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', t.lower().strip())).strip()

def token_f1(pred, ref):
    p_t = normalize_text(pred).split()
    r_t = normalize_text(ref).split()
    if not p_t and not r_t: return 1.0
    if not p_t or  not r_t: return 0.0
    common = sum((Counter(p_t) & Counter(r_t)).values())
    if common == 0: return 0.0
    p = common / len(p_t); r = common / len(r_t)
    return 2 * p * r / (p + r)

def semantic_sim(pred, ref):
    if not pred.strip() or not ref.strip(): return 0.0
    e1 = bi_enc.encode(pred, convert_to_tensor=True, normalize_embeddings=True)
    e2 = bi_enc.encode(ref,  convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(e1, e2)[0][0])

def bertscore_f1(pred, ref, lang="en"):
    if not pred.strip() or not ref.strip(): return 0.0
    try:
        from bert_score import score as _bs
        _, _, F = _bs([pred], [ref], lang=lang, verbose=False)
        return float(F[0])
    except Exception as e:
        log.warning(f"BERTScore failed: {e}"); return 0.0

# Timeout wrapper for QA model generation — returns empty string on timeout or error
class _TimeoutError(Exception): pass
def _alarm(sig, frm): raise _TimeoutError()

def with_timeout(fn, secs=30):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(secs)
    try:
        r = fn(); signal.alarm(0); return r
    except _TimeoutError:
        log.warning(f"  Timed out after {secs}s"); return ""
    except Exception as e:
        signal.alarm(0); log.warning(f"  Error: {e}"); return ""
    finally:
        signal.alarm(0)

# QA models
from transformers import (pipeline as hf_pipeline, AutoTokenizer,
                           AutoModelForCausalLM, GenerationConfig)
try:
    from transformers import BitsAndBytesConfig; HAS_BNB = True
except ImportError:
    HAS_BNB = False

QA_INFO = {
    "deepset/roberta-base-squad2": {"type":"extractive", "params":125_000_000},
    "Qwen/Qwen2.5-3B-Instruct":    {"type":"causal",     "params":3_000_000_000},
}

class QAModel:
    def __init__(self, model_name):
        self.model_name = model_name
        info            = QA_INFO.get(model_name, {})
        self.model_type = info.get("type", "extractive")
        self.params     = info.get("params", 0)
        log.info(f"Loading {model_name} ({self.model_type}, {self.params/1e6:.0f}M params)")
        if DEVICE == "cuda": torch.cuda.empty_cache(); gc.collect()

        if self.model_type == "extractive":
            self.pipe = hf_pipeline("question-answering", model=model_name,
                                    device=0 if DEVICE=="cuda" else -1)

        elif self.model_type == "causal":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            if HAS_BNB and DEVICE == "cuda":
                log.info("  4-bit NF4 quantisation enabled")
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16,
                                         bnb_4bit_use_double_quant=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, quantization_config=bnb,
                    device_map="auto", trust_remote_code=True)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
                    device_map="auto" if DEVICE=="cuda" else None,
                    trust_remote_code=True)

            self.model.generation_config = GenerationConfig(
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=1.0,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
            self.model.eval()
            if DEVICE == "cuda":
                log.info(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def answer(self, question, context):
        if self.model_type == "extractive":
            try:
                r = self.pipe(question=question, context=context[:2000])
                return r["answer"]
            except Exception as e:
                log.warning(f"  Extractive error: {e}"); return ""

        elif self.model_type == "causal":
            def _gen():
                user_msg = f"Context:\n{context}\n\nQuestion: {question}"
                messages = [{"role":"system","content":SYSTEM_PROMPT},
                            {"role":"user",  "content":user_msg}]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                else:
                    text = f"{SYSTEM_PROMPT}\n\nUser: {user_msg}\n\nAssistant:"
                inputs = self.tokenizer(text, return_tensors="pt",
                                         max_length=MAX_INPUT_TOKENS,
                                         truncation=True).to(self.model.device)
                in_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False, temperature=1.0,
                        repetition_penalty=REPETITION_PENALTY,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True)
                if DEVICE == "cuda": torch.cuda.synchronize()
                return self.tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
            return with_timeout(_gen, secs=30)

    def unload(self):
        for attr in ["model","tokenizer","pipe"]:
            if hasattr(self, attr): delattr(self, attr)
        if DEVICE == "cuda": torch.cuda.empty_cache(); torch.cuda.synchronize()
        gc.collect()
        log.info(f"  Unloaded {self.model_name}")




# Evalation of the QA pipeline with reference to the ground truth QA pairs — metrics computed only for relevant questions that passed the gate

print(f"\n{'='*70}\n  QA Pipeline Evaluation\n{'='*70}")
print(f"  Metrics: Token F1 (%)  Semantic Similarity (%)  BERTScore F1 (%)")

pipeline_rows = []

for model_name in MODELS:
    short = model_name.split("/")[-1]
    print(f"\n  Model: {model_name}")
    if DEVICE == "cuda": torch.cuda.empty_cache(); gc.collect()
    qa = QAModel(model_name)

    f1_scores = []; sem_scores = []; bs_scores = []
    rows = []

    for g in GROUND_TRUTH:
        results          = retrieve_full(g["q"], all_chunks, emb_all)
        relevant, reason = check_relevance(results)

        if not relevant:
            pred   = "Irrelevant Question"
            is_rel = False
        else:
            context = "\n\n".join(r["text"] for r in results)
            pred    = qa.answer(g["q"], context)
            is_rel  = True

        # Metrics for relevant questions that passed the gate
        if g["rel"] and is_rel:
            f1  = token_f1(pred, g["a"])
            sem = semantic_sim(pred, g["a"])
            bs  = bertscore_f1(pred, g["a"])
            f1_scores.append(f1); sem_scores.append(sem); bs_scores.append(bs)
        elif g["rel"] and not is_rel:
            f1_scores.append(0); sem_scores.append(0); bs_scores.append(0)

        # bi_score and ce_score — None when not computed for irrelevant questions
        best     = results[0] if results else {}
        bi_score = best.get("bi_score", None)
        ce_score = best.get("ce_score", None)

        rows.append({
            "id":           g["id"],
            "question":     g["q"],
            "expected":     g["a"],
            "predicted":    pred,
            "relevant":     g["rel"],
            "gate_passed":  is_rel,
            "bi_score":     bi_score,
            "ce_score":     ce_score,
            "token_f1":     round(f1_scores[-1], 4) if g["rel"] else "",
            "semantic_sim": round(sem_scores[-1], 4) if g["rel"] else "",
            "bertscore_f1": round(bs_scores[-1], 4) if g["rel"] else "",
        })

    avg_f1  = round(np.mean(f1_scores)  * 100, 2) if f1_scores  else 0
    avg_sem = round(np.mean(sem_scores) * 100, 2) if sem_scores else 0
    avg_bs  = round(np.mean(bs_scores)  * 100, 2) if bs_scores else 0

    print(f"  Token F1:            {avg_f1:.2f}%")
    print(f"  Semantic Similarity: {avg_sem:.2f}%")
    print(f"  BERTScore F1:        {avg_bs:.2f}%")

    pipeline_rows.append({
        "model":             model_name,
        "short":             short,
        "token_f1_pct":      avg_f1,
        "semantic_sim_pct":  avg_sem,
        "bertscore_f1_pct":  avg_bs,
    })

    safe = model_name.replace("/", "_")
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/pipeline_{safe}.csv", index=False)
    print(f"  Saved → {RESULTS_DIR}/pipeline_{safe}.csv")

    qa.unload(); del qa
    if DEVICE == "cuda": torch.cuda.empty_cache()
    gc.collect()

pd.DataFrame(pipeline_rows).to_csv(f"{RESULTS_DIR}/pipeline_summary.csv", index=False)
print(f"\n  Summary saved → {RESULTS_DIR}/pipeline_summary.csv")


# LLM-as-a-Judge (GPT-4o-mini)
# Judge Score = 0.5*(correctness/3) + 0.3*(faithfulness/3) + 0.2*(conciseness/3)

print(f"\n{'='*70}\n LLM-as-a-Judge (GPT-4o-mini)\n{'='*70}")
print(f"  Judge Score = 0.5*(C/3) + 0.3*(F/3) + 0.2*(Con/3)  →  range [0, 1]")

import openai, json

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a university module question answering system.
Evaluate the predicted answer against the reference answer for the given question.
Score each criterion from 0 to 3:

Correctness  (0-3): Does the predicted answer contain the correct information?
Faithfulness (0-3): Is the predicted answer grounded without hallucination?
Conciseness  (0-3): Is the predicted answer appropriately brief and focused?

Respond ONLY with a JSON object in this exact format:
{"correctness": <int>, "faithfulness": <int>, "conciseness": <int>}
"""

def gpt_judge(question, predicted, reference):
    """Calls GPT-4o-mini exclusively — no fallback. Raises on failure."""
    prompt = (f"Question: {question}\n"
              f"Reference Answer: {reference}\n"
              f"Predicted Answer: {predicted}")
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":JUDGE_SYSTEM_PROMPT},
                  {"role":"user",  "content":prompt}],
        temperature=0, max_tokens=100,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content.strip())

def compute_judge_score(scores):
    """Weighted composite normalised to [0, 1]"""
    return (0.5 * scores["correctness"]  / 3 +
            0.3 * scores["faithfulness"] / 3 +
            0.2 * scores["conciseness"]  / 3)

judge_rows         = []
judge_summary_rows = []

for model_name in MODELS:
    short = model_name.split("/")[-1]
    safe  = model_name.replace("/", "_")
    print(f"\n  Model: {model_name}")

    pred_path = f"{RESULTS_DIR}/pipeline_{safe}.csv"
    if not os.path.exists(pred_path):
        print(f"  WARNING: {pred_path} not found — run Cell 2 first"); continue

    pred_df    = pd.read_csv(pred_path)
    rel_df     = pred_df[pred_df["relevant"] == True].copy()
    all_scores = []

    for _, row in rel_df.iterrows():
        scores      = gpt_judge(row["question"], str(row["predicted"]), str(row["expected"]))
        judge_score = compute_judge_score(scores)
        all_scores.append(judge_score)
        judge_rows.append({
            "model":        model_name,
            "short":        short,
            "id":           row["id"],
            "question":     row["question"],
            "predicted":    row["predicted"],
            "expected":     row["expected"],
            "correctness":  scores["correctness"],
            "faithfulness": scores["faithfulness"],
            "conciseness":  scores["conciseness"],
            "judge_score":  round(judge_score, 4),
        })

    avg_score = round(np.mean(all_scores), 4) if all_scores else 0
    print(f"  Average Judge Score: {avg_score:.4f}  ({len(all_scores)} questions)")
    judge_summary_rows.append({
        "model":       model_name,
        "short":       short,
        "judge_score": avg_score,
    })

pd.DataFrame(judge_rows).to_csv(f"{RESULTS_DIR}/judge_results.csv", index=False)
pd.DataFrame(judge_summary_rows).to_csv(f"{RESULTS_DIR}/judge_summary.csv", index=False)
print(f"\n  Saved → {RESULTS_DIR}/judge_results.csv")
print(f"  Saved → {RESULTS_DIR}/judge_summary.csv")



# RAGAS Evaluation (GPT-4o-mini)

print(f"\n{'='*70}\n  CELL 4: RAGAS Evaluation (GPT-4o-mini)\n{'='*70}")
print(f"  Metrics: Faithfulness  Answer Relevancy")
print(f"  Evaluator LLM: GPT-4o-mini")

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

# calling OpenAI API directly for RAGAS evaluation — ensure API key is set in Colab environment
import os as _os
_os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Explicitly set both LLM and embeddings to GPT
ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini",
               api_key=OPENAI_API_KEY,
               temperature=0)
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small",
                     api_key=OPENAI_API_KEY)
)

ragas_rows = []

for model_name in MODELS:
    short = model_name.split("/")[-1]
    safe  = model_name.replace("/", "_")
    print(f"\n  Model: {model_name}")

    pred_path = f"{RESULTS_DIR}/pipeline_{safe}.csv"
    if not os.path.exists(pred_path):
        print(f"  WARNING: {pred_path} not found — run Cell 2 first"); continue

    pred_df    = pd.read_csv(pred_path)
    rel_df     = pred_df[pred_df["relevant"] == True].copy()
    ragas_data = []

    for _, row in rel_df.iterrows():
        results = retrieve_full(row["question"], all_chunks, emb_all)
        ctx     = [r["text"] for r in results]
        ragas_data.append({
            "question":     str(row["question"]),
            "answer":       str(row["predicted"]),
            "contexts":     ctx,
            "ground_truth": str(row["expected"]),
        })

    if not ragas_data:
        print(f"  No data available"); continue

    dataset = Dataset.from_list(ragas_data)
    try:
        result    = evaluate(dataset,
                             metrics=[faithfulness, answer_relevancy],
                             llm=ragas_llm,
                             embeddings=ragas_embeddings)
        df_result = result.to_pandas()
        faith     = round(df_result["faithfulness"].mean(), 4)
        ans_rel   = round(df_result["answer_relevancy"].mean(), 4)

        print(f"  Faithfulness:     {faith:.4f}")
        print(f"  Answer Relevancy: {ans_rel:.4f}")

        df_result["model"] = model_name
        df_result.to_csv(f"{RESULTS_DIR}/ragas_{safe}.csv", index=False)
        ragas_rows.append({
            "model":            model_name,
            "short":            short,
            "faithfulness":     faith,
            "answer_relevancy": ans_rel,
        })
    except Exception as e:
        raise RuntimeError(f"RAGAS evaluation failed for {model_name}: {e}") from e

pd.DataFrame(ragas_rows).to_csv(f"{RESULTS_DIR}/ragas_summary.csv", index=False)
print(f"\n  Saved → {RESULTS_DIR}/ragas_summary.csv")



# Result Summary

print(f"\n{'='*80}\n  CELL 5: FINAL SUMMARY\n{'='*80}")

def load_csv(name):
    p = f"{RESULTS_DIR}/{name}"
    if os.path.exists(p): return pd.read_csv(p)
    print(f"  WARNING: {p} not found — run the relevant cell first")
    return None

# QA Pipeline
pipe = load_csv("pipeline_summary.csv")
if pipe is not None:
    print(f"\n  QA Pipeline Evaluation:")
    print(f"  {'Model':<30s} {'Token F1(%)':>12s} {'Sem Sim(%)':>11s} {'BERTScore F1(%)':>16s}")
    print(f"  {'─'*72}")
    for _, r in pipe.iterrows():
        print(f"  {r['short']:<30s} {r['token_f1_pct']:>11.2f}%"
              f" {r['semantic_sim_pct']:>10.2f}% {r['bertscore_f1_pct']:>15.2f}%")

# LLM as a Judge
judge = load_csv("judge_summary.csv")
if judge is not None:
    print(f"\n  LLM-as-a-Judge (GPT-4o-mini):")
    print(f"  {'Model':<30s} {'Judge Score [0-1]':>18s}")
    print(f"  {'─'*50}")
    for _, r in judge.iterrows():
        print(f"  {r['short']:<30s} {r['judge_score']:>18.4f}")

# RAGAS
ragas = load_csv("ragas_summary.csv")
if ragas is not None:
    print(f"\n  RAGAS Evaluation (GPT-4o-mini):")
    print(f"  {'Model':<30s} {'Faithfulness':>13s} {'Answer Relevancy':>17s}")
    print(f"  {'─'*63}")
    for _, r in ragas.iterrows():
        print(f"  {r['short']:<30s} {r['faithfulness']:>13.4f}"
              f" {r['answer_relevancy']:>17.4f}")

# Summary Output file
print(f"\n{'='*80}\n  OUTPUT FILES (current directory)\n{'='*80}")
for f in sorted(os.listdir(".")):
    if f.endswith(".csv"):
        kb = os.path.getsize(f) / 1024
        print(f"  {f:<55s} {kb:.1f} KB")

print(f"\n{'='*80}\n  ALL CELLS COMPLETE\n{'='*80}")