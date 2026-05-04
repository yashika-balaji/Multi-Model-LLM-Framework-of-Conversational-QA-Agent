# speech_to_text.py to transcribe audio to text and create RAG chunks

import os
import re
import sys
import whisper
import numpy as np
import soundfile as sf
import librosa
from sentence_transformers import SentenceTransformer
import torch

# Configuration of all parameters
# STT Model Parameters
MODEL_SIZE = "large-v2"      # Best model selected for STT after evaluation of WER/CER/TER/BERTScore-F1
TEMPERATURE = 0.0                     # Deterministic (no sampling randomness)
BEAM_SIZE = 8                         # Beam search width
NO_SPEECH_THRESHOLD = 0.45            # Silence detection sensitivity
COMPRESSION_RATIO_THRESHOLD = 2.4     # Reject repetitive hallucinations
LOGPROB_THRESHOLD = -2.0              # Reject low-confidence segments

# Audio Pre-processing parameters
VAD_THRES = 25                        # dB threshold for silence trimming
PRE_EMPHASIS_COEFF = 0.97             # High-pass filter coefficient

#Chunking Parameters
SIM_THRES = 0.50           # similarity threshold for conversational speech
CHUNK_MAX_CHARS = 512       # Max characters per chunk
CHUNK_MIN_WORDS = 30        # Minimum words per chunk (kills fragments)
MERGE_SHORT_WORDS = 10    # Utterances shorter than this get merged
WINDOW_SIZE = 6            # Look-ahead window for semantic grouping
OVERLAP_SENTENCES = 0       # Overlap between consecutive chunks

# Output paths
_m = MODEL_SIZE.replace(".", "_")
OUTPUT_TXT = f"transcript.txt"
OUTPUT_KB  = f"knowledge_base.txt"


# Audio pre-processing
def audio_preprocessing(audio_path):
    audio, sr = sf.read(audio_path)
    audio = np.append(audio[0], audio[1:] - PRE_EMPHASIS_COEFF * audio[:-1]) # Pre-emphasis

    audio_trimmed, _ = librosa.effects.trim(audio, top_db=VAD_THRES) # VAD trim: remove leading/trailing silence

    return audio_trimmed.astype(np.float32)

# STT model transcription
def transcribe_speech(audio_data):

    model = whisper.load_model(MODEL_SIZE)
    audio_tensor = torch.from_numpy(audio_data).to(torch.float32)

    result = model.transcribe(
        audio_tensor,
        temperature=TEMPERATURE,
        beam_size=BEAM_SIZE,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        logprob_threshold=LOGPROB_THRESHOLD,
    )
    return result["text"]


# Cleaning text
def post_process_text(text):
 
    # Removing fillers
    fillers = (
        r"\b(uhm|umm|um|uh|ah|oh|hmm|"
        r"like|you know|i mean|basically|actually|sort of|kind of|"
        r"let's see|well|so yeah|yeah|right|okay so)\b"
    )
    clean = re.sub(fillers, "", text, flags=re.IGNORECASE)

    # Removing standalone words
    standalone = r'(?<=[.!?])\s+(Okay|Yes|No|Right|Sure|Fine|Great|Alright|Indeed|Definitely)[.!?]\s+'
    clean = re.sub(standalone, ". ", clean)

    #Normalizing whitespace and punctuation
    clean = re.sub(r'\s+', ' ', clean).strip()
    clean = re.sub(r'\.\s*\.', '.', clean)       
    clean = re.sub(r',\s*,', ',', clean)          
    clean = re.sub(r'\s+([.!?,])', r'\1', clean)

    #Capitalizing first letter
    return clean[0].upper() + clean[1:] if clean else ""


# Text Chunking
def split_into_sentences(text):

    # Protect abbreviations from being split
    text = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms|Jr|Sr|e\.g|i\.e|etc|vs)\.',
                  r'\1<ABBR>', text)
    # Protect decimal numbers
    text = re.sub(r'(\d)\.(\d)', r'\1<DEC>\2', text)

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Restore abbreviations and decimals
    sentences = [s.replace('<ABBR>', '.').replace('<DEC>', '.').strip()
                 for s in sentences if s.strip()]

    return sentences

# Merge short sentences to eliminate fragments
def merge_short_sentences(sentences, min_words=MERGE_SHORT_WORDS):
    if not sentences:
        return sentences

    merged = []
    buffer = ""

    for sent in sentences:
        buffer = (buffer + " " + sent).strip() if buffer else sent

        # Flush buffer when it's long enough AND ends with proper punctuation
        if len(buffer.split()) >= min_words:
            merged.append(buffer)
            buffer = ""

    # Attach any remaining buffer to the last chunk
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged

# Window-based semantic chunking with overlap
def semantic_chunk_with_window(sentences, sim_threshold=SIM_THRES,
                                max_chars=CHUNK_MAX_CHARS,
                                min_words=CHUNK_MIN_WORDS,
                                window=WINDOW_SIZE,
                                overlap=OVERLAP_SENTENCES):

    if len(sentences) < 2:
        return [" ".join(sentences)] if sentences else []

    # Encode all sentences at once
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences, normalize_embeddings=True)

    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_text = sentences[0]

    for i in range(1, len(sentences)):
        #Window-based similarity
        window_start = max(0, i - window)
        window_indices = list(range(window_start, i))

        if window_indices:
            # Average similarity with recent context
            similarities = [
                np.dot(embeddings[j], embeddings[i])
                for j in window_indices
            ]
            avg_similarity = np.mean(similarities)
        else:
            avg_similarity = 0.0

        # continue chunk or start new one based on similarity and length
        would_exceed_max = len(current_chunk_text + " " + sentences[i]) > max_chars

        if avg_similarity >= sim_threshold and not would_exceed_max:
            # Continue current chunk
            current_chunk_sentences.append(sentences[i])
            current_chunk_text += " " + sentences[i]
        else:
            if len(current_chunk_text.split()) >= min_words:
                chunks.append(current_chunk_text.strip())
            elif chunks:
                chunks[-1] = chunks[-1] + " " + current_chunk_text.strip()
            else:
                chunks.append(current_chunk_text.strip())

            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk_sentences) >= overlap:
                overlap_sentences = current_chunk_sentences[-overlap:]
                current_chunk_sentences = overlap_sentences + [sentences[i]]
                current_chunk_text = " ".join(s for s in current_chunk_sentences)
            else:
                current_chunk_sentences = [sentences[i]]
                current_chunk_text = sentences[i]

    # Flush last chunk
    if current_chunk_text.strip():
        if len(current_chunk_text.split()) >= min_words:
            chunks.append(current_chunk_text.strip())
        elif chunks:
            chunks[-1] = chunks[-1] + " " + current_chunk_text.strip()
        else:
            chunks.append(current_chunk_text.strip())

    return chunks

# Creating chunk knowledge base
def create_kb_chunks(text):

    # Step A: Split into sentences
    sentences = split_into_sentences(text)
    print(f"  Sentences after split: {len(sentences)}")

    # Step B: Merge short utterances (FIX #1)
    merged = merge_short_sentences(sentences, min_words=MERGE_SHORT_WORDS)
    print(f"  After merging short (<{MERGE_SHORT_WORDS} words): {len(merged)}")

    # Step C: Semantic chunking with window (FIX #2 & #3)
    chunks = semantic_chunk_with_window(
        merged,
        sim_threshold=SIM_THRES,
        max_chars=CHUNK_MAX_CHARS,
        min_words=CHUNK_MIN_WORDS,
        window=WINDOW_SIZE,
        overlap=OVERLAP_SENTENCES,
    )
    print(f"  Final chunks: {len(chunks)}")

    return chunks

# main pipeline for speech to text and chunking
def main(input_file):
    print(f"\n{'='*60}")
    print(f"  Input: {input_file}")
    print(f"{'='*60}")

    print(f"\n  Parameters:")
    print(f"    Whisper model:     {MODEL_SIZE}")
    print(f"    Temperature:       {TEMPERATURE}")
    print(f"    Beam size:         {BEAM_SIZE}")
    print(f"    Sim threshold:     {SIM_THRES}")
    print(f"    Chunk max chars:   {CHUNK_MAX_CHARS}")
    print(f"    Chunk min words:   {CHUNK_MIN_WORDS}")
    print(f"    Merge short words: {MERGE_SHORT_WORDS}")
    print(f"    Window size:       {WINDOW_SIZE}")
    print(f"    Overlap:           {OVERLAP_SENTENCES}")

    #Pre-process audio
    print(f"\nAudio preprocessing...")
    preprocess_audio = audio_preprocessing(input_file)
    print(f"  Audio length: {len(preprocess_audio)/16000:.1f}s at 16kHz")

    # Transcribe
    print(f"\nWhisper transcription ({MODEL_SIZE})...")
    raw_text = transcribe_speech(preprocess_audio)
    print(f"  Raw transcript: {len(raw_text.split())} words")

    # Cleaning text
    print(f"\nText cleaning")
    final_transcript = post_process_text(raw_text)
    print(f"  Cleaned transcript: {len(final_transcript.split())} words")

    # Semantic chunking
    print(f"\nSemantic chunking with window")
    kb_chunks = create_kb_chunks(final_transcript)

    # Results
    output_file = OUTPUT_KB
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(kb_chunks, 1):
            f.write(f"[CHUNK {i:03d}]\n{chunk}\n\n")

    # full transcript for reference and evaluation against ground truth
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_transcript)

    print(f"\n{'='*60}")
    print(f"  SUCCESS")
    print(f"  Chunks: {len(kb_chunks)} saved to {output_file}")
    print(f"  Full transcript saved to transcript.txt")
    print(f"{'='*60}")

    return kb_chunks


if __name__ == "__main__":
    main("audio_processed.wav")