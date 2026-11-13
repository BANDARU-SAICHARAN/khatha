"""
Khaatha - make test clips with computer voices (no need to record your own voice)
-------------------------------------------------------------------------------
Uses Sarvam Text-to-Speech to create voice clips in Telugu, Kannada, Hindi and English,
saves them in clips/, and writes labels.csv with the correct answers automatically.

Run:  python make_clips.py
Then: python test_pipeline.py
"""

import base64
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI

BASE = Path(__file__).parent
CLIPS_DIR = BASE / "clips"
LABELS_FILE = BASE / "labels.csv"

load_dotenv(BASE / ".env")
client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

# (text to speak, language, expected_name, expected_amount, expected_type)
SENTENCES = [
    # Telugu
    ("సురేష్, రెండు వందల యాభై, బియ్యం", "te-IN", "Suresh", 250, "credit"),
    ("రమేష్ వంద రూపాయలు కట్టాడు", "te-IN", "Ramesh", 100, "payment"),
    ("లక్ష్మి కి ఐదు వందలు రాసుకో, నూనె", "te-IN", "Lakshmi", 500, "credit"),
    ("వెంకట్ వెయ్యి రూపాయలు ఇచ్చాడు", "te-IN", "Venkat", 1000, "payment"),
    ("శ్రీను, మూడు వందల ఇరవై, పాలు బ్రెడ్", "te-IN", "Srinu", 320, "credit"),
    ("సురేష్ రెండు వందలు", "te-IN", "Suresh", 200, "unknown"),  # ambiguous on purpose
    # Kannada
    ("ಮಂಜುನಾಥ್, ಇನ್ನೂರ ಐವತ್ತು, ಅಕ್ಕಿ", "kn-IN", "Manjunath", 250, "credit"),
    ("ರವಿ ಸಾವಿರ ರೂಪಾಯಿ ಕೊಟ್ಟಿದ್ದಾರೆ", "kn-IN", "Ravi", 1000, "payment"),
    ("ಗೀತಾ ಅವರಿಗೆ ನಾನೂರು ರೂಪಾಯಿ ಬಾಕಿ ಬರೆದುಕೊಳ್ಳಿ, ಎಣ್ಣೆ", "kn-IN", "Geetha", 400, "credit"),
    ("ಪ್ರಕಾಶ್ ಐನೂರು ರೂಪಾಯಿ ಕೊಟ್ಟರು", "kn-IN", "Prakash", 500, "payment"),
    # Hindi
    ("सुनील के नाम पांच सौ लिख दो, चावल", "hi-IN", "Sunil", 500, "credit"),
    ("राजेश ने दो सौ रुपये दिए", "hi-IN", "Rajesh", 200, "payment"),
    ("अनीता, ढाई सौ, दूध और अंडे", "hi-IN", "Anita", 250, "credit"),
    ("मोहन ने हज़ार रुपये जमा किए", "hi-IN", "Mohan", 1000, "payment"),
    ("विक्रम का छह सौ उधार लिख दो", "hi-IN", "Vikram", 600, "credit"),
    # English
    ("Ramesh paid one hundred and fifty rupees", "en-IN", "Ramesh", 150, "payment"),
    ("Kiran, three hundred, rice and dal", "en-IN", "Kiran", 300, "credit"),
    ("Write four hundred and fifty for Priya, soap and oil", "en-IN", "Priya", 450, "credit"),
    ("Arjun gave back two thousand rupees", "en-IN", "Arjun", 2000, "payment"),
    ("Deepa took bread and eggs for eighty rupees", "en-IN", "Deepa", 80, "credit"),
]

# Different voices so the test isn't just one speaker.
# If a voice isn't available on your account, the script falls back to the default voice.
SPEAKERS = ["anushka", "abhilash", "manisha", "vidya", "arya", "karun", "hitesh"]


def speak(text: str, lang: str, speaker: str) -> tuple[bytes, str]:
    try:
        resp = client.text_to_speech.convert(
            text=text, language_code=lang, speaker=speaker, output_audio_codec="wav"
        )
        used = speaker
    except Exception:
        resp = client.text_to_speech.convert(
            text=text, language_code=lang, output_audio_codec="wav"
        )
        used = "default"
    return base64.b64decode(resp.audios[0]), used


def main():
    if not os.getenv("SARVAM_API_KEY") or "paste_" in os.getenv("SARVAM_API_KEY", ""):
        print("SARVAM_API_KEY is missing in .env")
        return
    CLIPS_DIR.mkdir(exist_ok=True)

    rows = []
    for i, (text, lang, name, amount, typ) in enumerate(SENTENCES, 1):
        file_name = f"tts{i:02d}.wav"
        speaker = SPEAKERS[i % len(SPEAKERS)]
        print(f"[{i}/{len(SENTENCES)}] {file_name} ({lang}) ... ", end="", flush=True)
        try:
            audio, used = speak(text, lang, speaker)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        (CLIPS_DIR / file_name).write_bytes(audio)
        rows.append([file_name, name, amount, typ, f"{lang} voice={used}: {text}"])
        print(f"saved (voice: {used})")

    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "expected_name", "expected_amount", "expected_type", "note"])
        w.writerows(rows)

    print(f"\nDone: {len(rows)} clips in clips/ and answers written to labels.csv")
    print("Next: python test_pipeline.py")


if __name__ == "__main__":
    main()