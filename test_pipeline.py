"""
Khaatha - Week 1 accuracy test
--------------------------------
For every voice clip listed in labels.csv:
  1. Speech-to-text with Sarvam (Saaras v3)
  2. Gemini turns the transcript into a ledger entry: name, amount, credit/payment
  3. Compare with the expected answer you wrote in labels.csv

Run:  python test_pipeline.py
Output: results.csv + an accuracy summary in the terminal.
"""

import csv
import json
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sarvamai import SarvamAI

# ---------- settings ----------
BASE = Path(__file__).parent
CLIPS_DIR = BASE / "clips"
LABELS_FILE = BASE / "labels.csv"
RESULTS_FILE = BASE / "results.csv"
CACHE_DIR = BASE / "cache"  # saves transcripts so re-runs don't spend Sarvam credits

load_dotenv(BASE / ".env")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
STT_MODE = os.getenv("STT_MODE", "codemix")  # transcribe | codemix | translit | translate


# ---------- what we want Gemini to return ----------
class LedgerEntry(BaseModel):
    customer_name: Optional[str] = Field(
        description="Customer name written in English letters, e.g. 'Suresh'. null if not said."
    )
    amount_rupees: Optional[int] = Field(
        description="Amount in rupees as a number, e.g. 250. null if not said."
    )
    type: Literal["credit", "payment", "unknown"] = Field(
        description="credit = customer took goods on udhaar (owes money). "
        "payment = customer paid money back. unknown = cannot tell."
    )
    items: List[str] = Field(description="Items mentioned, in English. Empty list if none.")


SYSTEM_PROMPT = """You read short voice notes from Indian kirana shopkeepers and convert them
into ledger entries. The text may be in Telugu, Kannada, Hindi, Tamil, English, or a mix,
and may be in native script or English letters.

Rules:
- Numbers may be spoken in words in any language. Convert them to digits.
  Examples: "rendu vandala yabhai" = 250, "dhai sau" = 250, "ondu saavira" = 1000,
  "paanch sau" = 500, "veyyi" = 1000.
- "paid", "kattadu", "ichadu", "diya", "kottidda", "jama" usually mean payment.
- "likh do", "raasuko", "baaki", "udhaar", "credit", or just name + amount + item
  usually mean credit.
- If you really cannot tell credit from payment, use "unknown". Do not guess.
- Always write the customer name in English letters.
"""


# ---------- helpers ----------
def check_setup():
    problems = []
    if not SARVAM_API_KEY or "paste_" in SARVAM_API_KEY:
        problems.append("SARVAM_API_KEY is missing in .env")
    if not GEMINI_API_KEY or "paste_" in GEMINI_API_KEY:
        problems.append("GEMINI_API_KEY is missing in .env")
    if not LABELS_FILE.exists():
        problems.append("labels.csv not found")
    if not CLIPS_DIR.exists():
        problems.append("clips/ folder not found")
    if problems:
        print("Setup problems:")
        for p in problems:
            print("  -", p)
        sys.exit(1)


def transcribe(sarvam: SarvamAI, clip: Path) -> dict:
    """Speech-to-text with Sarvam. Uses a cache so each clip is sent only once."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{clip.name}.{STT_MODE}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    start = time.time()
    with open(clip, "rb") as f:
        resp = sarvam.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            mode=STT_MODE,
            language_code="unknown",  # auto-detect language
        )
    result = {
        "transcript": resp.transcript,
        "language": resp.language_code,
        "stt_seconds": round(time.time() - start, 2),
    }
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def extract(gemini: genai.Client, transcript: str) -> tuple[LedgerEntry, float]:
    """Transcript -> structured ledger entry with Gemini."""
    start = time.time()
    resp = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Voice note: {transcript}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=LedgerEntry,
            temperature=0,
        ),
    )
    entry = resp.parsed if resp.parsed else LedgerEntry.model_validate_json(resp.text)
    return entry, round(time.time() - start, 2)


def names_match(expected: str, got: Optional[str]) -> bool:
    if not got:
        return False
    a, b = expected.strip().lower(), got.strip().lower()
    return a == b or SequenceMatcher(None, a, b).ratio() >= 0.8


def pct(n: int, total: int) -> str:
    return f"{n}/{total} ({100 * n / total:.0f}%)" if total else "0/0"


# ---------- main ----------
def main():
    check_setup()
    sarvam = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    gemini = genai.Client(api_key=GEMINI_API_KEY)

    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    rows = []
    for i, lab in enumerate(labels, 1):
        clip = CLIPS_DIR / lab["file"].strip()
        print(f"[{i}/{len(labels)}] {clip.name} ... ", end="", flush=True)
        row = {
            "file": clip.name,
            "expected_name": lab["expected_name"],
            "expected_amount": lab["expected_amount"],
            "expected_type": lab["expected_type"],
            "note": lab.get("note", ""),
        }
        if not clip.exists():
            print("FILE NOT FOUND, skipped")
            row["error"] = "file not found"
            rows.append(row)
            continue
        try:
            stt = transcribe(sarvam, clip)
            entry, llm_s = extract(gemini, stt["transcript"])
        except Exception as e:  # keep going even if one clip fails
            print(f"ERROR: {e}")
            row["error"] = str(e)
            rows.append(row)
            continue

        name_ok = names_match(lab["expected_name"], entry.customer_name)
        amount_ok = str(entry.amount_rupees) == lab["expected_amount"].strip()
        type_ok = entry.type == lab["expected_type"].strip().lower()
        all_ok = name_ok and amount_ok and type_ok

        row.update(
            transcript=stt["transcript"],
            language=stt["language"],
            got_name=entry.customer_name,
            got_amount=entry.amount_rupees,
            got_type=entry.type,
            got_items=", ".join(entry.items),
            name_ok=name_ok,
            amount_ok=amount_ok,
            type_ok=type_ok,
            all_ok=all_ok,
            stt_seconds=stt["stt_seconds"],
            llm_seconds=llm_s,
            error="",
        )
        rows.append(row)
        print("OK" if all_ok else "MISMATCH", f'-> "{stt["transcript"]}"')

    # save results
    fields = [
        "file", "note", "transcript", "language",
        "expected_name", "got_name", "name_ok",
        "expected_amount", "got_amount", "amount_ok",
        "expected_type", "got_type", "type_ok",
        "got_items", "all_ok", "stt_seconds", "llm_seconds", "error",
    ]
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary
    done = [r for r in rows if not r.get("error")]
    n = len(done)
    print("\n========== SUMMARY ==========")
    print(f"Clips tested     : {n} (failed/skipped: {len(rows) - n})")
    print(f"Name correct     : {pct(sum(r['name_ok'] for r in done), n)}")
    print(f"Amount correct   : {pct(sum(r['amount_ok'] for r in done), n)}")
    print(f"Type correct     : {pct(sum(r['type_ok'] for r in done), n)}")
    print(f"ALL 3 correct    : {pct(sum(r['all_ok'] for r in done), n)}")
    if n:
        print(f"Avg Gemini time  : {sum(r['llm_seconds'] for r in done) / n:.2f}s")
    print(f"\nDetails saved to {RESULTS_FILE.name} - open it in Excel to see every mistake.")


if __name__ == "__main__":
    main()