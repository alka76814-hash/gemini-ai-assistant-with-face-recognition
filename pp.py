"""
Emotion-Aware Gemini Voice Assistant with Face Recognition
===========================================================
Combines:
  • OpenCV + DeepFace  →  real-time emotion detection & face boxes
  • MediaPipe          →  custom "bored" emotion via facial landmarks
  • LBPH               →  offline face recognition (known_faces/)
  • Google Gemini      →  voice assistant that SEES your emotions & learns about you
  • SpeechRecognition  →  wake-word + speech-to-text
  • pyttsx3            →  text-to-speech (offline, no API key needed)

INSTALL
───────
pip install opencv-python tensorflow tensorflow-hub deepface mediapipe numpy scipy
pip install google-generativeai SpeechRecognition pyttsx3 pyaudio Pillow
pip install spacy && python -m spacy download en_core_web_sm

GET A GEMINI API KEY
────────────────────
https://aistudio.google.com/app/apikey
Then either:
  export GEMINI_API_KEY="your_key"       (Linux/Mac)
  set    GEMINI_API_KEY=your_key         (Windows)
or pass  --api-key YOUR_KEY  on the CLI.

RUN
───
python emotion_voice_assistant.py
python emotion_voice_assistant.py --api-key YOUR_KEY --source 0
python emotion_voice_assistant.py --wake-word "hey assistant"
python emotion_voice_assistant.py --name "Alice"          # skip name prompt

HOW IT WORKS
────────────
1. The camera window runs continuously, detecting faces + emotions.
2. Say the wake-word (default: "hey gemini") to start speaking.
3. Gemini receives:
     • Your spoken question/statement
     • Your CURRENT detected emotion
     • Everything it has learned about you in this session
4. Gemini replies by voice AND updates its memory about you.
5. Type  q  in the camera window to quit.
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time
import threading
import json
import collections
import textwrap
from datetime import datetime
from scipy.spatial import distance as dist

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   128,   0),
    "fear":     (128,  0,  128),
    "happy":    (0,   220, 220),
    "sad":      (220, 100,   0),
    "surprise": (0,   165, 255),
    "neutral":  (180, 180, 180),
    "bored":    (42,  127, 255),
}
DEFAULT_COLOR = (0, 255, 0)

# MediaPipe landmark indices
L_EYE    = [362, 385, 387, 263, 373, 380]
R_EYE    = [33,  160, 158, 133, 153, 144]
MOUTH_V  = [13,  14]
MOUTH_H  = [61,  291]
NOSE_TIP = 1
CHIN     = 152
FOREHEAD = 10


# ═════════════════════════════════════════════════════════════════════════════
# EMOTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def ear(landmarks, indices, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def mar(landmarks, w, h):
    top    = np.array([landmarks[MOUTH_V[0]].x * w, landmarks[MOUTH_V[0]].y * h])
    bottom = np.array([landmarks[MOUTH_V[1]].x * w, landmarks[MOUTH_V[1]].y * h])
    left   = np.array([landmarks[MOUTH_H[0]].x * w, landmarks[MOUTH_H[0]].y * h])
    right  = np.array([landmarks[MOUTH_H[1]].x * w, landmarks[MOUTH_H[1]].y * h])
    return dist.euclidean(top, bottom) / (dist.euclidean(left, right) + 1e-6)

def head_pitch(landmarks, w, h):
    nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    fore = np.array([landmarks[FOREHEAD].x * w, landmarks[FOREHEAD].y * h])
    chin = np.array([landmarks[CHIN].x     * w, landmarks[CHIN].y     * h])
    return float(np.clip((nose[1] - fore[1]) / (dist.euclidean(fore, chin) + 1e-6), 0, 1))


class BoredomTracker:
    def __init__(self):
        self.neutral_start = None
        self.blink_times   = collections.deque()
        self.prev_ear      = 1.0
        self.history       = collections.deque(maxlen=15)

    def update(self, ear_val, mar_val, pitch, dominant):
        now = time.time()
        if self.prev_ear > 0.21 and ear_val < 0.18:
            self.blink_times.append(now)
        self.prev_ear = ear_val
        while self.blink_times and now - self.blink_times[0] > 5:
            self.blink_times.popleft()
        if dominant in ("neutral", "sad"):
            if self.neutral_start is None:
                self.neutral_start = now
        else:
            self.neutral_start = None
        neutral_secs = (now - self.neutral_start) if self.neutral_start else 0
        ear_s   = max(0, (0.20 - ear_val)  / 0.20) * 100
        mar_s   = max(0, (mar_val - 0.25)  / 0.35) * 100
        pitch_s = max(0, (pitch - 0.52)    / 0.48) * 100
        neut_s  = min(100, neutral_secs / 4 * 100)
        bpm     = (len(self.blink_times) / 5) * 60
        blink_s = max(0, (8 - bpm) / 8) * 100
        score   = 0.30*ear_s + 0.15*mar_s + 0.15*pitch_s + 0.25*neut_s + 0.15*blink_s
        self.history.append(score)
        return float(np.mean(self.history))


# ═════════════════════════════════════════════════════════════════════════════
# SHARED STATE  (camera thread ↔ assistant thread)
# ═════════════════════════════════════════════════════════════════════════════
class SharedState:
    """Thread-safe container for current emotion + person info."""
    def __init__(self):
        self._lock           = threading.Lock()
        self.current_emotion = "neutral"
        self.emotion_scores  = {}
        self.face_name       = ""
        self.bored_score     = 0.0
        # Gemini memory about the user (persisted as JSON)
        self.memory: dict    = {}
        self.user_name       = ""
        # UI overlays
        self.assistant_text  = ""
        self.listening       = False
        self.speaking        = False
        # Live camera frame shared with assistant thread (for Gemini Vision)
        self.current_frame   = None
        self._frame_lock     = threading.Lock()
        # Live noise level 0.0-1.0 sampled from mic thread
        self.noise_level     = 0.0
        self._noise_history  = collections.deque(maxlen=30)  # last 30 samples for smoothing

    def set_noise(self, rms: float):
        """rms is raw RMS amplitude. Normalised to 0-1 and smoothed."""
        with self._lock:
            self._noise_history.append(min(rms / 3000.0, 1.0))
            self.noise_level = float(sum(self._noise_history) / len(self._noise_history))

    def get_noise(self):
        with self._lock:
            return self.noise_level

    def set_frame(self, frame):
        with self._frame_lock:
            self.current_frame = frame.copy() if frame is not None else None

    def get_frame(self):
        with self._frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def set_emotion(self, emotion, scores, name="", bored_score=0.0):
        with self._lock:
            self.current_emotion = emotion
            self.emotion_scores  = scores
            self.face_name       = name
            self.bored_score     = bored_score

    def get_emotion_snapshot(self):
        with self._lock:
            return (self.current_emotion, dict(self.emotion_scores),
                    self.face_name, self.bored_score)

    def update_memory(self, new_facts: dict):
        with self._lock:
            self.memory.update(new_facts)

    def get_memory(self):
        with self._lock:
            return dict(self.memory)

    def set_ui(self, text="", listening=None, speaking=None):
        with self._lock:
            if text is not None:
                self.assistant_text = text
            if listening is not None:
                self.listening = listening
            if speaking is not None:
                self.speaking = speaking

    def get_ui(self):
        with self._lock:
            return self.assistant_text, self.listening, self.speaking


STATE = SharedState()


# ═════════════════════════════════════════════════════════════════════════════
# NOISE LEVEL SAMPLER  (runs in its own daemon thread)
# ═════════════════════════════════════════════════════════════════════════════
def noise_sampler_thread(stop_event: threading.Event):
    """Continuously reads mic RMS and pushes it to STATE.set_noise()."""
    try:
        import pyaudio
    except ImportError:
        print("[WARN] pyaudio not found — noise bar disabled")
        return

    CHUNK      = 1024
    FORMAT     = pyaudio.paInt16
    CHANNELS   = 1
    RATE       = 16000
    pa         = pyaudio.PyAudio()
    stream     = None

    try:
        stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                         input=True, frames_per_buffer=CHUNK)
        print("[NOISE] Mic sampler started")
        while not stop_event.is_set():
            try:
                data   = stream.read(CHUNK, exception_on_overflow=False)
                arr    = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms    = float(np.sqrt(np.mean(arr ** 2)))
                STATE.set_noise(rms)
            except Exception:
                pass
    except Exception as e:
        print(f"[NOISE] Could not open mic stream: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        pa.terminate()


# ═════════════════════════════════════════════════════════════════════════════
# ADVANCED USER PROFILE  (NLP-powered learning)
# ═════════════════════════════════════════════════════════════════════════════
try:
    import tensorflow_hub as hub
    TF_HUB_AVAILABLE = True
except ImportError:
    TF_HUB_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class AdvancedUserProfile:
    """
    3-Tier Memory System
    ────────────────────
    CORE       (permanent)  — name, age, job, location. Never deleted.
    LONG-TERM  (compressed) — summaries of old conversation batches. Grows slowly.
    SHORT-TERM (recent)     — last 10 full turns with embeddings. Rolls over into long-term.

    When short-term is full (10 entries):
      • Oldest 5 are compressed into a single long-term summary entry
      • Freed slots are reused
    When long-term exceeds 40 summaries:
      • Least-unique summaries are merged/dropped (cosine distance pruning)
    Core memory is never touched.
    """

    SHORT_TERM_MAX  = 10    # full turns kept in short-term
    SHORT_TERM_ROLL = 5     # how many to compress when full
    LONG_TERM_MAX   = 40    # max long-term summary entries

    def __init__(self, profile_path="user_data/memory.json"):
        self.profile_path = profile_path
        self.profile = {
            # ── Tier 0: Core (permanent, always sent to Gemini) ──────────────
            "core": {
                "name": "", "age": "", "location": "", "occupation": "",
                "personality_notes": [], "important_dates": {}
            },
            # ── Tier 1: Long-term (compressed summaries) ──────────────────────
            "long_term": [],        # [{"summary": str, "period": str, "embedding": [...]}]
            # ── Tier 2: Short-term (recent full turns with embeddings) ────────
            "short_term": [],       # [{"user":str,"assistant":str,"emotion":str,"embedding":[...], "ts":str}]
            # ── Structured knowledge (never expires) ──────────────────────────
            "preferences":    {},
            "relationships":  {},
            "goals":          [],
            "entities":       {"people":[],"places":[],"organizations":[],"products":[]},
            "interests":      {"topics":[],"hobbies":[],"skills":[]},
            # ── Stats ─────────────────────────────────────────────────────────
            "stats": {
                "total_conversations": 0,
                "sentiment_history": [],   # last 50
                "tier_promotions": 0,      # how many times short→long happened
            },
            "created_at":  None,
            "last_updated": None,
        }
        self.nlp_model      = None
        self.use_embeddings = None
        self._gemini_model  = None   # set later for compression calls
        self._positive_words = {"love","like","enjoy","great","awesome","fantastic",
                                "wonderful","excellent","happy","excited","amazing"}
        self._negative_words = {"hate","dislike","bad","terrible","awful","horrible",
                                "annoying","frustrating","sad","angry","upset"}
        self._load_nlp_models()
        self._load_profile()

    # ─────────────────────────────────────────────────────────────────────────
    # Init helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _load_nlp_models(self):
        if SPACY_AVAILABLE:
            for model in ["en_core_web_sm", "en_core_web_md"]:
                try:
                    self.nlp_model = spacy.load(model)
                    print(f"✓ spaCy loaded: {model}")
                    break
                except Exception:
                    pass
            if not self.nlp_model:
                print("⚠ spaCy model missing. Run:  python -m spacy download en_core_web_sm")
        if TF_HUB_AVAILABLE:
            try:
                print("⏳ Loading Universal Sentence Encoder …")
                self.use_embeddings = hub.load(
                    "https://tfhub.dev/google/universal-sentence-encoder/4")
                print("✓ Universal Sentence Encoder ready")
            except Exception as e:
                print(f"⚠ USE not loaded: {e}")

    def set_gemini_model(self, genai_model):
        """Give profile a Gemini model handle for compression calls."""
        self._gemini_model = genai_model

    def _load_profile(self):
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path) as f:
                    saved = json.load(f)

                # ── Auto-migrate old profile formats ──────────────────────────
                # interaction_stats → stats
                if "interaction_stats" in saved and "stats" not in saved:
                    saved["stats"] = saved.pop("interaction_stats")
                    print("[MIGRATE] interaction_stats → stats")
                # basic_info → core
                if "basic_info" in saved and "core" not in saved:
                    saved["core"] = saved.pop("basic_info")
                    print("[MIGRATE] basic_info → core")
                # flat conversation_history → short_term entries
                if "conversation_history" in saved:
                    for c in saved.pop("conversation_history", []):
                        entry = {
                            "user":      c.get("user", ""),
                            "assistant": c.get("assistant", ""),
                            "emotion":   c.get("visual_emotion", "unknown"),
                            "sentiment": c.get("sentiment", "neutral"),
                            "embedding": [],
                            "ts":        c.get("timestamp", "")
                        }
                        saved.setdefault("short_term", []).append(entry)
                    saved["short_term"] = saved["short_term"][-self.SHORT_TERM_MAX:]
                    print(f"[MIGRATE] conversation_history → short_term "
                          f"({len(saved['short_term'])} entries)")
                # old semantic_memory format is incompatible — drop it
                if "semantic_memory" in saved:
                    saved.pop("semantic_memory")
                    print("[MIGRATE] Dropped old semantic_memory (incompatible format)")
                # ensure stats sub-keys exist
                saved.setdefault("stats", {})
                for k, default in [("total_conversations", 0),
                                   ("sentiment_history",   []),
                                   ("tier_promotions",     0)]:
                    saved["stats"].setdefault(k, default)

                # ── Merge saved data — keep defaults for any missing keys ──────
                for key, val in saved.items():
                    if key in self.profile:
                        if isinstance(val, dict):
                            self.profile[key].update(val)
                        else:
                            self.profile[key] = val
                print(f"✓ Profile loaded  |  "
                      f"core={bool(self.profile['core']['name'])}  "
                      f"long_term={len(self.profile['long_term'])}  "
                      f"short_term={len(self.profile['short_term'])}")
                return
            except Exception as e:
                print(f"⚠ Profile load error: {e}")
        self.profile["created_at"] = datetime.now().isoformat()
        self.save_profile()
        print("✓ New profile created")

    def save_profile(self):
        dirpath = os.path.dirname(self.profile_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self.profile["last_updated"] = datetime.now().isoformat()
        with open(self.profile_path, "w") as f:
            json.dump(self.profile, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _embed(self, text: str):
        """Return numpy embedding or None."""
        if self.use_embeddings is None:
            return None
        try:
            return self.use_embeddings([text]).numpy()[0]
        except Exception:
            return None

    def _cosine_sim(self, a, b):
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / denom)

    # ─────────────────────────────────────────────────────────────────────────
    # NLP extraction
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_entities(self, text):
        if not self.nlp_model:
            return {}
        doc = self.nlp_model(text)
        ents = {"people":[],"places":[],"organizations":[],"products":[]}
        for e in doc.ents:
            if e.label_ == "PERSON":        ents["people"].append(e.text)
            elif e.label_ in ["GPE","LOC"]: ents["places"].append(e.text)
            elif e.label_ == "ORG":         ents["organizations"].append(e.text)
            elif e.label_ == "PRODUCT":     ents["products"].append(e.text)
        return ents

    def _analyze_sentiment(self, text):
        words = set(text.lower().split())
        pos = len(words & self._positive_words)
        neg = len(words & self._negative_words)
        return "positive" if pos > neg else "negative" if neg > pos else "neutral"

    def _extract_interests(self, text):
        if not self.nlp_model:
            return []
        doc = self.nlp_model(text)
        return [c.text for c in doc.noun_chunks if c.root.dep_ == "dobj"]

    # ─────────────────────────────────────────────────────────────────────────
    # TIER MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────
    def _compress_to_long_term(self, turns: list) -> str:
        """
        Compress a list of short-term turns into a single summary string.
        Uses Gemini if available, otherwise builds a structured text summary.
        """
        if self._gemini_model is not None:
            try:
                block = "\n".join(
                    f"User ({t.get('emotion','?')}): {t['user']}\n"
                    f"Assistant: {t['assistant']}"
                    for t in turns
                )
                prompt = (
                    "Summarise these conversation turns into a compact memory entry "
                    "of 3-5 bullet points capturing key facts, preferences, emotions "
                    "and anything important about the user. Be concise.\n\n" + block
                )
                resp = self._gemini_model.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                print(f"[MEMORY] Gemini compression failed ({e}), using fallback")

        # Fallback: structured text summary
        lines = [f"[Compressed {len(turns)} turns — "
                 f"{turns[0].get('ts','?')[:10]} to {turns[-1].get('ts','?')[:10]}]"]
        for t in turns:
            lines.append(f"• ({t.get('emotion','?')} emotion) "
                         f"User said: {t['user'][:100]}")
        return "\n".join(lines)

    def _roll_short_to_long_term(self):
        """
        Promote oldest SHORT_TERM_ROLL entries from short-term into a
        single compressed long-term summary entry.
        """
        to_compress = self.profile["short_term"][:self.SHORT_TERM_ROLL]
        self.profile["short_term"] = self.profile["short_term"][self.SHORT_TERM_ROLL:]

        summary_text = self._compress_to_long_term(to_compress)
        emb = self._embed(summary_text)

        entry = {
            "summary":   summary_text,
            "period":    f"{to_compress[0].get('ts','?')[:10]} — "
                         f"{to_compress[-1].get('ts','?')[:10]}",
            "turn_count": len(to_compress),
            "embedding": emb.tolist() if emb is not None else [],
            "created_at": datetime.now().isoformat(),
        }
        self.profile["long_term"].append(entry)
        self.profile["stats"]["tier_promotions"] += 1
        print(f"[MEMORY] 🔄 Promoted {len(to_compress)} short-term turns → long-term "
              f"(long_term size: {len(self.profile['long_term'])})")

        # Prune long-term if over limit
        if len(self.profile["long_term"]) > self.LONG_TERM_MAX:
            self._prune_long_term()

    def _prune_long_term(self):
        """
        Remove least-unique long-term entries (those most similar to their
        neighbours) until we are back under LONG_TERM_MAX.
        Entries without embeddings are pruned first.
        """
        lt = self.profile["long_term"]

        # Step 1: drop entries with no embeddings first (oldest fallback)
        no_emb = [e for e in lt if not e.get("embedding")]
        has_emb = [e for e in lt if e.get("embedding")]

        while len(lt) > self.LONG_TERM_MAX and no_emb:
            oldest = no_emb.pop(0)
            lt.remove(oldest)
            print("[MEMORY] ✂ Pruned no-embedding long-term entry")

        # Step 2: cosine similarity pruning — drop least unique
        while len(lt) > self.LONG_TERM_MAX:
            # Find the pair with highest similarity
            best_sim, best_idx = -1, 0
            emb_entries = [(i, e) for i, e in enumerate(lt) if e.get("embedding")]
            for i in range(len(emb_entries) - 1):
                idx_a, ea = emb_entries[i]
                idx_b, eb = emb_entries[i + 1]
                sim = self._cosine_sim(ea["embedding"], eb["embedding"])
                if sim > best_sim:
                    best_sim, best_idx = sim, idx_b  # drop the later duplicate
            lt.pop(best_idx)
            print(f"[MEMORY] ✂ Pruned most-similar long-term entry (sim={best_sim:.2f})")

        self.profile["long_term"] = lt

    # ─────────────────────────────────────────────────────────────────────────
    # Core memory update
    # ─────────────────────────────────────────────────────────────────────────
    def _update_core(self, user_input: str, extracted: dict):
        """Keep core memory up-to-date with basic facts."""
        c = self.profile["core"]
        low = user_input.lower()
        import re

        if extracted.get("name"):
            c["name"] = extracted["name"]
        if extracted.get("age"):
            c["age"] = extracted["age"]
        if extracted.get("location"):
            c["location"] = extracted["location"]
        if extracted.get("occupation"):
            c["occupation"] = extracted["occupation"]

        # Personality notes from strong sentiment phrases
        strong_phrases = ["i always","i never","i really","i absolutely","i truly"]
        for ph in strong_phrases:
            if ph in low:
                note = low.split(ph)[1].strip().split(".")[0][:80]
                if note and note not in c["personality_notes"]:
                    c["personality_notes"].append(note)
                    if len(c["personality_notes"]) > 20:
                        c["personality_notes"] = c["personality_notes"][-20:]

    # ─────────────────────────────────────────────────────────────────────────
    # Main public API
    # ─────────────────────────────────────────────────────────────────────────
    def extract_and_store(self, user_input: str, ai_response: str, visual_emotion=None):
        """Full NLP extraction pipeline. Returns dict of what was found."""
        extracted = {}
        low = user_input.lower()

        # ── Entities ─────────────────────────────────────────────────────────
        entities = self._extract_entities(user_input)
        for kind, items in entities.items():
            for item in items:
                bucket = self.profile["entities"].setdefault(kind, [])
                if item not in bucket:
                    bucket.append(item)
                    extracted[f"new_{kind[:-1]}"] = item

        # ── Sentiment ────────────────────────────────────────────────────────
        sentiment = self._analyze_sentiment(user_input)
        self.profile["stats"]["sentiment_history"].append({
            "sentiment": sentiment,
            "visual_emotion": visual_emotion or "unknown",
            "ts": datetime.now().isoformat()
        })
        if len(self.profile["stats"]["sentiment_history"]) > 50:
            self.profile["stats"]["sentiment_history"] =                 self.profile["stats"]["sentiment_history"][-50:]
        extracted["sentiment"] = sentiment

        # ── Preferences ──────────────────────────────────────────────────────
        for pattern, ptype in [
            ("i love","strong_like"), ("i like","like"), ("i enjoy","like"),
            ("i prefer","preference"), ("my favorite","favorite"),
            ("i hate","strong_dislike"), ("i don't like","dislike"),
            ("i can't stand","strong_dislike")
        ]:
            if pattern in low:
                pref = low.split(pattern)[1].strip().split(".")[0].split(",")[0].strip()
                bucket = self.profile["preferences"].setdefault(ptype, [])
                if pref and pref not in bucket:
                    bucket.append(pref)
                    extracted[f"pref_{ptype}"] = pref

        # ── Goals ────────────────────────────────────────────────────────────
        for pattern in ["i want to","i'm trying to","my goal is","i hope to","i plan to"]:
            if pattern in low:
                goal = low.split(pattern)[1].strip().split(".")[0]
                self.profile["goals"].append(
                    {"goal": goal, "ts": datetime.now().isoformat()})
                extracted["new_goal"] = goal

        # ── Relationships ────────────────────────────────────────────────────
        for pattern, rel in [
            ("my wife","wife"),("my husband","husband"),("my mom","mother"),
            ("my dad","father"),("my brother","brother"),("my sister","sister"),
            ("my friend","friend"),("my boss","boss"),("my partner","partner"),
            ("my colleague","colleague"),("my son","son"),("my daughter","daughter")
        ]:
            if pattern in low:
                ents = self._extract_entities(user_input)
                if ents.get("people"):
                    pname = ents["people"][0]
                    self.profile["relationships"][pname] = rel
                    extracted["new_relationship"] = f"{pname} ({rel})"

        # ── Basic info ───────────────────────────────────────────────────────
        import re
        if "my name is" in low:
            name = low.split("my name is")[1].strip().split()[0].strip(".,!?")
            extracted["name"] = name.title()
        m = re.search(r"(\d+)\s+years?\s+old", low)
        if m:
            extracted["age"] = m.group(1)
        for pat in ["i live in","i'm from","i'm in","based in"]:
            if pat in low:
                loc = low.split(pat)[1].strip().split()[0].strip(".,!?")
                extracted["location"] = loc.title()
        for pat in ["i work as","i'm a","my job is"]:
            if pat in low:
                job = low.split(pat)[1].strip().split(".")[0].strip()
                extracted["occupation"] = job
                break

        # ── Interests ────────────────────────────────────────────────────────
        for interest in self._extract_interests(user_input):
            if interest not in self.profile["interests"]["topics"]:
                self.profile["interests"]["topics"].append(interest)
                extracted["new_interest"] = interest

        # ── Update core memory (Tier 0) ───────────────────────────────────────
        self._update_core(user_input, extracted)

        return extracted

    def add_conversation(self, user_input: str, ai_response: str, visual_emotion=None):
        """Record one conversation turn through all three tiers."""
        extracted = self.extract_and_store(user_input, ai_response, visual_emotion)

        # Build embedding for this turn
        turn_text = f"{user_input} {ai_response}"
        emb = self._embed(turn_text)

        # Add to short-term (Tier 2)
        self.profile["short_term"].append({
            "user":      user_input,
            "assistant": ai_response,
            "emotion":   visual_emotion or "unknown",
            "sentiment": extracted.get("sentiment", "neutral"),
            "embedding": emb.tolist() if emb is not None else [],
            "ts":        datetime.now().isoformat(),
        })

        self.profile["stats"]["total_conversations"] += 1

        # ── Roll short-term → long-term when full ─────────────────────────────
        if len(self.profile["short_term"]) >= self.SHORT_TERM_MAX:
            self._roll_short_to_long_term()

        self.save_profile()
        return extracted

    def find_similar(self, query: str, top_k=3):
        """Semantic search across ALL tiers (short + long term)."""
        if self.use_embeddings is None:
            return []
        qe = self._embed(query)
        if qe is None:
            return []
        results = []
        # Search short-term
        for t in self.profile["short_term"]:
            if t.get("embedding"):
                sim = self._cosine_sim(qe, t["embedding"])
                results.append((t["user"][:80], sim, "short"))
        # Search long-term
        for lt in self.profile["long_term"]:
            if lt.get("embedding"):
                sim = self._cosine_sim(qe, lt["embedding"])
                results.append((lt["summary"][:80], sim, "long"))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # Context generation
    # ─────────────────────────────────────────────────────────────────────────
    def get_context_for_gemini(self) -> str:
        """Rich, tiered context string injected into every Gemini prompt."""
        p = self.profile
        parts = []

        # ── TIER 0: Core — always present ────────────────────────────────────
        c = p["core"]
        core_facts = {k: v for k, v in c.items()
                      if v and k != "personality_notes"}
        if core_facts:
            parts.append("CORE FACTS: " + ", ".join(
                f"{k}={v}" for k, v in core_facts.items()))
        if c.get("personality_notes"):
            parts.append("PERSONALITY: " + " | ".join(c["personality_notes"][:4]))

        # ── Structured knowledge ──────────────────────────────────────────────
        if p["preferences"]:
            prefs = []
            for ptype, items in p["preferences"].items():
                if items:
                    prefs.append(f"{ptype.replace('_',' ')}: {', '.join(items[:3])}")
            if prefs:
                parts.append("PREFERENCES: " + " | ".join(prefs))

        if p["relationships"]:
            rels = [f"{n} ({r})" for n, r in list(p["relationships"].items())[:6]]
            parts.append("PEOPLE: " + ", ".join(rels))

        if p["goals"]:
            parts.append("GOALS: " + " | ".join(g["goal"] for g in p["goals"][-3:]))

        if p["interests"]["topics"]:
            parts.append("INTERESTS: " + ", ".join(p["interests"]["topics"][:5]))

        if p["entities"]["places"]:
            parts.append("PLACES: " + ", ".join(p["entities"]["places"][:4]))

        # ── TIER 1: Long-term — recent summaries ─────────────────────────────
        if p["long_term"]:
            parts.append(f"\nLONG-TERM MEMORY ({len(p['long_term'])} compressed batches):")
            for lt in p["long_term"][-3:]:   # last 3 summaries
                parts.append(f"  [{lt['period']}] {lt['summary'][:200]}")

        # ── TIER 2: Short-term — recent full turns ────────────────────────────
        if p["short_term"]:
            parts.append(f"\nSHORT-TERM MEMORY (last {len(p['short_term'])} turns):")
            for t in p["short_term"][-5:]:
                parts.append(f"  [{t['emotion']} / {t['sentiment']}] "
                              f"User: {t['user'][:80]}")
                parts.append(f"    Assistant: {t['assistant'][:80]}")

        # ── Stats ─────────────────────────────────────────────────────────────
        sh = p["stats"]["sentiment_history"]
        if sh:
            mood_recent = [s["sentiment"] for s in sh[-5:]]
            parts.append(f"\nMOOD TREND: {' → '.join(mood_recent)}")
        parts.append(f"TOTAL CONVERSATIONS: {p['stats']['total_conversations']}  "
                     f"|  TIER PROMOTIONS: {p['stats']['tier_promotions']}")

        return "\n".join(parts)

    def get_profile_summary(self) -> str:
        """Human-readable profile for terminal display."""
        p = self.profile
        lines = ["="*58, "🧠  TIERED MEMORY PROFILE", "="*58]

        c = p["core"]
        lines.append("📌 CORE (permanent):")
        for k, v in c.items():
            if v and k != "personality_notes":
                lines.append(f"   {k}: {v}")
        if c.get("personality_notes"):
            lines.append("   personality: " + " | ".join(c["personality_notes"][:3]))

        lines.append(f"\n🗂  LONG-TERM  ({len(p['long_term'])} summaries / max {self.LONG_TERM_MAX}):")
        for lt in p["long_term"][-4:]:
            lines.append(f"   [{lt['period']}]  {lt['summary'][:120]}")

        lines.append(f"\n⚡ SHORT-TERM  ({len(p['short_term'])} turns / max {self.SHORT_TERM_MAX}):")
        for t in p["short_term"][-3:]:
            lines.append(f"   {t['ts'][:16]}  ({t['emotion']})  {t['user'][:80]}")

        if p["preferences"]:
            lines.append("\n⭐ Preferences:")
            for pt, items in p["preferences"].items():
                if items:
                    lines.append(f"   {pt}: {', '.join(items[:4])}")
        if p["relationships"]:
            lines.append("\n👥 People:")
            for n, r in p["relationships"].items():
                lines.append(f"   • {n} ({r})")
        if p["goals"]:
            lines.append(f"\n🎯 Goals ({len(p['goals'])}):")
            for g in p["goals"][-3:]:
                lines.append(f"   • {g['goal']}")
        if p["interests"]["topics"]:
            lines.append(f"\n💡 Interests:")
            for t in p["interests"]["topics"][:6]:
                lines.append(f"   • {t}")

        stats = p["stats"]
        lines.append(f"\n📊 Conversations : {stats['total_conversations']}")
        lines.append(f"   Tier promotions: {stats['tier_promotions']}")
        sh = stats["sentiment_history"]
        if sh:
            recent = [s["sentiment"] for s in sh[-5:]]
            lines.append(f"   Mood trend    : {' → '.join(recent)}")
        lines.append("="*58)
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# GEMINI ASSISTANT
# ═════════════════════════════════════════════════════════════════════════════

def build_system_prompt(user_name: str) -> str:
    return textwrap.dedent(f"""
        You are a friendly, perceptive, emotionally-intelligent AI voice assistant.
        The person talking to you is named {user_name or "the user"}.

        VISION
        ──────
        You have a LIVE CAMERA feed. Every message includes a real webcam photo.
        Use it to SEE the person — describe their expression, environment, what
        they are wearing. When asked "can you see me?" answer from the actual image.

        ADVANCED MEMORY SYSTEM
        ──────────────────────
        You have access to a rich NLP-powered profile of this person, built from
        every past conversation. It includes:
          • Basic info (name, age, location, occupation)
          • Preferences (likes, dislikes, favourites)
          • Known people & their relationships (wife, friend, boss etc.)
          • Goals and aspirations
          • Interests and hobbies
          • Sentiment/mood trends over time
          • Recent conversation context

        This profile is injected into every message as [PROFILE CONTEXT].
        USE IT actively — reference past facts naturally, notice mood changes,
        ask follow-up questions about things they mentioned before.

        EMOTION AWARENESS
        ─────────────────
        You receive both the CV-detected emotion label AND the live image.
        Adapt your tone: gentle when sad, energetic when happy, engaging when bored.
        Notice if their visual emotion contradicts what they are saying.

        RESPONSE STYLE
        ──────────────
        Keep spoken responses concise (2-4 sentences) — they are read aloud by TTS.
        Be warm, natural, and conversational. Use the person's name occasionally.
        Reference things you remember to show you genuinely know them.
    """).strip()

MEMORY_TAG_START = "<memory_update>"
MEMORY_TAG_END   = "</memory_update>"

def parse_memory_update(text: str):
    """Extract and remove the <memory_update> JSON block from Gemini's reply."""
    if MEMORY_TAG_START not in text:
        return text, {}
    try:
        start = text.index(MEMORY_TAG_START) + len(MEMORY_TAG_START)
        end   = text.index(MEMORY_TAG_END)
        json_str = text[start:end].strip()
        data     = json.loads(json_str)
        clean    = (text[:text.index(MEMORY_TAG_START)] +
                    text[end + len(MEMORY_TAG_END):]).strip()
        return clean, data.get("new_facts", {})
    except Exception:
        # If parsing fails just return original text
        return text, {}


class GeminiAssistant:
    def __init__(self, api_key: str, user_name: str):
        try:
            import google.generativeai as genai
        except ImportError:
            print("[ERROR] google-generativeai not installed.\n"
                  "        pip install google-generativeai")
            sys.exit(1)

        genai.configure(api_key=api_key)
        self.genai      = genai
        self.user_name  = user_name
        self.chat       = None
        self._init_chat()

    def _init_chat(self):
        model = self.genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=build_system_prompt(self.user_name),
        )
        self.chat = model.start_chat(history=[])

    def ask(self, user_text: str, emotion: str, scores: dict,
            bored_score: float, profile: "AdvancedUserProfile", frame=None) -> str:
        scores_str = ", ".join(f"{k}: {v:.0f}%" for k, v in
                                sorted(scores.items(), key=lambda kv: -kv[1])[:4])
        profile_ctx = profile.get_context_for_gemini()

        # Find semantically similar past conversations
        similar = profile.find_similar(user_text, top_k=2)
        similar_str = ""
        
        if similar:
            similar_str = "\nSEMANTICALLY SIMILAR PAST MESSAGES:\n" + "\n".join(
                f"  - [{tier}] {text[:80]} (score={sim:.2f})" for text, sim, tier in similar)

        context = textwrap.dedent(f"""
            [SYSTEM CONTEXT — do not read aloud]
            Timestamp        : {datetime.now().strftime("%H:%M")}
            CV emotion label : {emotion}  (boredom score: {bored_score:.0f}/100)
            Emotion scores   : {scores_str}

            [PROFILE CONTEXT]
            {profile_ctx}
            {similar_str}

            NOTE: A live webcam image is attached. Use it to SEE the person.
            ────────────────────────────────────
            {self.user_name}: {user_text}
        """).strip()

        try:
            if frame is not None:
                # Encode the frame as JPEG bytes and send as inline image
                import PIL.Image, io
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = PIL.Image.fromarray(rgb)
                # Resize to keep API payload small (~640 wide max)
                max_w = 640
                if pil_img.width > max_w:
                    ratio = max_w / pil_img.width
                    pil_img = pil_img.resize(
                        (max_w, int(pil_img.height * ratio)), PIL.Image.LANCZOS)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=80)
                image_bytes = buf.getvalue()
                # Use generate_content with vision (chat history kept manually)
                model = self.genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=build_system_prompt(self.user_name),
                )
                # Build message with history + image
                history_msgs = []
                for msg in self.chat.history:
                    history_msgs.append({"role": msg.role, "parts": [msg.parts[0].text
                        if hasattr(msg.parts[0], "text") else str(msg.parts[0])]})
                history_msgs.append({
                    "role": "user",
                    "parts": [
                        {"mime_type": "image/jpeg", "data": image_bytes},
                        context
                    ]
                })
                response = model.generate_content(history_msgs)
                reply = response.text
                # Append to chat history for continuity
                self.chat.history.append(
                    type("M", (), {"role": "user",
                                   "parts": [type("P", (), {"text": context})()]})())
                self.chat.history.append(
                    type("M", (), {"role": "model",
                                   "parts": [type("P", (), {"text": reply})()]})())
                return reply
            else:
                response = self.chat.send_message(context)
                return response.text
        except Exception as e:
            print(f"[GEMINI ERROR] {e}")
            return f"Sorry, I had a problem processing that: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# SPEECH  (STT + TTS)
# ═════════════════════════════════════════════════════════════════════════════
def init_tts():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        voices = engine.getProperty("voices")
        # Prefer a female voice if available
        for v in voices:
            if "female" in v.name.lower() or "zira" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        return engine
    except Exception as e:
        print(f"[WARN] TTS unavailable: {e}")
        return None

def speak(engine, text: str):
    if engine is None:
        print(f"[ASSISTANT] {text}")
        return
    STATE.set_ui(speaking=True)
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass
    STATE.set_ui(speaking=False)

def listen_once(recogniser, source, timeout=6) -> str:
    """Listen for one utterance on an ALREADY-OPEN mic source. Returns transcript or ""."""
    import speech_recognition as sr
    STATE.set_ui(listening=True)
    try:
        audio = recogniser.listen(source, timeout=timeout, phrase_time_limit=12)
        text  = recogniser.recognize_google(audio)
        print(f"[STT] Heard: '{text}'")
        return text
    except sr.WaitTimeoutError:
        print("[STT] Timeout - nothing heard")
        return ""
    except sr.UnknownValueError:
        print("[STT] Could not understand")
        return ""
    except Exception as e:
        print(f"[STT Error] {e}")
        return ""
    finally:
        STATE.set_ui(listening=False)


def assistant_thread(args, stop_event: threading.Event):
    """Runs the voice assistant loop in a background thread."""
    try:
        import speech_recognition as sr
    except ImportError:
        print("[ERROR] SpeechRecognition not installed.\n"
              "        pip install SpeechRecognition pyaudio")
        return

    # ── Load advanced profile ─────────────────────────────────────────────────
    print("[INFO] Loading advanced user profile ...")
    profile = AdvancedUserProfile(profile_path=args.profile)

    # Migrate old "basic_info" key to new "core" key if old profile exists on disk
    if "basic_info" in profile.profile:
        old_info = profile.profile.pop("basic_info", {})
        for k in ["name", "age", "location", "occupation"]:
            if old_info.get(k) and not profile.profile["core"].get(k):
                profile.profile["core"][k] = old_info[k]
        profile.save_profile()
        print("[INFO] Migrated old profile format to new tiered format")

    user_name = args.name or profile.profile["core"].get("name", "")

    tts = init_tts()
    recogniser = sr.Recognizer()

    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"[ERROR] Microphone unavailable: {e}")
        return

    # Open mic ONCE and keep it open — avoids freeze from repeated open/close
    mic_source = mic.__enter__()

    if not user_name:
        speak(tts, "Hello! I am your emotion-aware assistant. What is your name?")
        user_name = listen_once(recogniser, mic_source, timeout=8).strip()
        if not user_name:
            user_name = "Friend"
        profile.profile["core"]["name"] = user_name.title()
        profile.save_profile()
        STATE.user_name = user_name
        speak(tts, f"Nice to meet you, {user_name}! I will remember everything you tell me. "
                   f"Say {args.wake_word} whenever you want to talk.")
        print(f"[PROFILE] New user: {user_name}")
    else:
        STATE.user_name = user_name
        emotion, *_ = STATE.get_emotion_snapshot()
        convo_count = profile.profile["stats"]["total_conversations"]
        if convo_count > 0:
            speak(tts, f"Welcome back, {user_name}! We have had {convo_count} conversations. "
                       f"You look {emotion} today.")
        else:
            speak(tts, f"Welcome back, {user_name}! You look {emotion} today.")
        print(profile.get_profile_summary())

    # ── Init Gemini ───────────────────────────────────────────────────────────
    api_key = "YOUR GEMINI API KEY HERE" #######################################################
    if not api_key:
        speak(tts, "No Gemini API key found. Please set GEMINI_API_KEY.")
        return

    STATE._profile_ref = profile   # share with camera thread for HUD display
    assistant = GeminiAssistant(api_key, user_name)
    wake       = args.wake_word.lower()
    wake_words = wake.split()

    # Calibrate mic ONCE at startup using the already-open source
    print("[INFO] Calibrating mic for ambient noise (1 sec) ...")
    recogniser.adjust_for_ambient_noise(mic_source, duration=1)
    recogniser.energy_threshold = max(recogniser.energy_threshold, 300)
    print(f"[INFO] Energy threshold: {recogniser.energy_threshold:.0f}")
    print(f"[INFO] Ready! Say '{wake}' to activate.")

    # Stop phrases — saying any of these exits conversation mode
    STOP_PHRASES = ["stop", "goodbye", "bye", "exit", "quit", "shut up",
                    "stop listening", "go to sleep", "that's all", "thats all"]

    def is_stop_phrase(text):
        t = text.lower().strip()
        return any(p in t for p in STOP_PHRASES)

    def handle_turn(user_text):
        """Send user_text + live frame to Gemini, extract info, update profile."""
        if not user_text:
            return None
        print(f"[USER] {user_text}")
        emotion, scores, face_name, bored_score = STATE.get_emotion_snapshot()
        frame = STATE.get_frame()
        STATE.set_ui(text=f"You ({emotion}): {user_text}")

        # Ask Gemini (now passes full profile instead of simple dict)
        response_raw  = assistant.ask(user_text, emotion, scores, bored_score, profile, frame)
        response_clean, gemini_facts = parse_memory_update(response_raw)

        # ── Advanced NLP extraction on what the user said ────────────────────
        extracted = profile.add_conversation(user_text, response_clean, visual_emotion=emotion)

        # Also merge any Gemini-detected facts into core memory
        if gemini_facts:
            profile.profile["core"].update(gemini_facts)
            profile.save_profile()
            print(f"[PROFILE] Gemini facts merged: {gemini_facts}")

        # Print what was learned this turn
        if extracted:
            learned = {k:v for k,v in extracted.items() if k != "sentiment"}
            if learned:
                print(f"[PROFILE] Extracted this turn: {learned}")

        print(f"[ASSISTANT] {response_clean}")
        STATE.set_ui(text=f"Assistant: {response_clean[:80]}..." if len(response_clean) > 80
                          else f"Assistant: {response_clean}")
        speak(tts, response_clean)
        return response_clean

    while not stop_event.is_set():
        # ── PHASE 1: Wait for wake word ───────────────────────────────────────
        STATE.set_ui(text=f"Say '{wake}' to activate ...")
        print(f"[WAKE] Waiting ... say '{wake}'")
        heard = listen_once(recogniser, mic_source, timeout=10)

        if not heard:
            continue

        heard_lower = heard.lower()
        triggered = (
            wake in heard_lower
            or any(w in heard_lower for w in wake_words)
            or "gemini" in heard_lower
            or "assistant" in heard_lower
        )
        if not triggered:
            print(f"[WAKE] Ignored: '{heard}'")
            continue

        print(f"[WAKE] Triggered by: '{heard}'")

        # Check if the wake word utterance already contained a question
        # e.g. user said "hey gemini what is the weather" in one breath
        inline_question = heard_lower
        for w in wake_words + ["hey", "gemini", "assistant"]:
            inline_question = inline_question.replace(w, "")
        inline_question = inline_question.strip(" ,.")

        if inline_question:
            # Question was in the same utterance as wake word — answer it directly
            print(f"[INFO] Inline question detected: '{inline_question}'")
            speak(tts, "Sure!")
            handle_turn(inline_question)
        else:
            speak(tts, "Yes?")

        # ── PHASE 2: Conversation mode — keep listening until stop phrase ─────
        consecutive_misses = 0
        MAX_MISSES = 3   # 3 silent timeouts in a row = go back to sleep

        while not stop_event.is_set():
            print(f"[CONV] Listening for next message (miss {consecutive_misses}/{MAX_MISSES}) ...")
            STATE.set_ui(text="Listening ... (say 'stop' to deactivate)", listening=True)
            user_text = listen_once(recogniser, mic_source, timeout=8)
            STATE.set_ui(listening=False)

            if not user_text:
                consecutive_misses += 1
                if consecutive_misses >= MAX_MISSES:
                    print(f"[CONV] {MAX_MISSES} timeouts in a row — going back to sleep")
                    speak(tts, "Going to sleep. Say hey gemini to wake me up again.")
                    break
                STATE.set_ui(text="Still here! Keep talking or say 'stop'.")
                continue

            # Reset miss counter on any successful speech
            consecutive_misses = 0

            # Check for stop/exit phrase
            if is_stop_phrase(user_text):
                print(f"[CONV] Stop phrase detected: '{user_text}'")
                speak(tts, "Okay, going to sleep. Say hey gemini whenever you need me!")
                break

            # Special: show profile summary in terminal
            if any(p in user_text.lower() for p in
                   ["show profile","my profile","what do you know about me","profile summary"]):
                summary = profile.get_profile_summary()
                print(summary)
                # Give Gemini a chance to summarise verbally too
                handle_turn(user_text)
                continue

            # Normal turn
            handle_turn(user_text)

    STATE.set_ui(text="")
    try:
        mic.__exit__(None, None, None)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA + EMOTION DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def load_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError:
        print("[ERROR] deepface not installed.  pip install deepface")
        sys.exit(1)

def analyse_emotion(face_bgr, DeepFace):
    try:
        r = DeepFace.analyze(face_bgr, actions=["emotion"],
                             enforce_detection=False, silent=True)
        if isinstance(r, list): r = r[0]
        return r["dominant_emotion"], r["emotion"]
    except Exception:
        return "neutral", {}

def build_recogniser(known_dir="known_faces"):
    if not os.path.isdir(known_dir):
        return None, {}
    rec = cv2.face.LBPHFaceRecognizer_create()
    images, labels, label_map = [], [], {}
    lid = 0
    for name in sorted(os.listdir(known_dir)):
        d = os.path.join(known_dir, name)
        if not os.path.isdir(d): continue
        label_map[lid] = name
        for f in os.listdir(d):
            img = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(lid)
        lid += 1
    if not images: return None, {}
    rec.train(images, np.array(labels))
    return rec, label_map

def build_mp():
    try:
        import mediapipe as mp
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=4, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except ImportError:
        return None


# ─── Drawing ──────────────────────────────────────────────────────────────────
def draw_face(frame, x, y, w, h, name, emotion, scores, bored_score, color):
    # ── Bounding box (thicker when bored) ────────────────────────────────────
    thickness = 3 if emotion == "bored" else 2
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

    # ── Corner accents for a modern look ─────────────────────────────────────
    clen = max(12, w // 6)
    for cx, cy, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx + dx*clen, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy*clen), color, 3)

    # ── Name badge (above box) ────────────────────────────────────────────────
    if name:
        (nw, nh), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # Dark pill background
        cv2.rectangle(frame, (x, y - nh - 30), (x + nw + 12, y - 18), (30,30,30), -1)
        cv2.rectangle(frame, (x, y - nh - 30), (x + nw + 12, y - 18), color, 1)
        cv2.putText(frame, name, (x+6, y-22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ── Emotion label (directly above box or below name) ─────────────────────
    emo_label = emotion.upper()
    if emotion == "bored":
        emo_label += f"  {bored_score:.0f}%"
    (ew, eh), _ = cv2.getTextSize(emo_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    label_y = (y - eh - 36) if name else (y - 6)
    label_y_top = label_y - eh - 6
    cv2.rectangle(frame, (x, label_y_top), (x + ew + 12, label_y + 4), color, -1)
    cv2.putText(frame, emo_label, (x+6, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    # ── Emotion confidence bars (below box) ───────────────────────────────────
    if scores:
        by = y + h + 10
        for emo, val in sorted(scores.items(), key=lambda kv: -kv[1])[:4]:
            bar_w = int(w * val / 100)
            ec    = EMOTION_COLORS.get(emo, DEFAULT_COLOR)
            # Bar background
            cv2.rectangle(frame, (x, by), (x+w, by+10), (40,40,40), -1)
            # Bar fill
            cv2.rectangle(frame, (x, by), (x+bar_w, by+10), ec, -1)
            # Label
            cv2.putText(frame, f"{emo[:4]}  {val:.0f}%",
                        (x+2, by+22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)
            by += 28

def draw_noise_bar(frame):
    """
    Vertical noise bar on the LEFT edge of the frame.
    Shows live mic RMS level with colour coding:
      green  = quiet        (< 30 %)
      yellow = moderate     (30-65 %)
      orange = loud         (65-85 %)
      red    = very loud    (> 85 %)
    """
    h, w   = frame.shape[:2]
    level  = STATE.get_noise()          # 0.0 – 1.0
    _, listening, speaking = STATE.get_ui()

    BAR_W     = 14
    BAR_H     = h - 80                  # leave room for HUD at bottom
    BAR_X     = 4
    BAR_Y_TOP = 30

    # Background track
    cv2.rectangle(frame,
                  (BAR_X, BAR_Y_TOP),
                  (BAR_X + BAR_W, BAR_Y_TOP + BAR_H),
                  (35, 35, 35), -1)
    cv2.rectangle(frame,
                  (BAR_X, BAR_Y_TOP),
                  (BAR_X + BAR_W, BAR_Y_TOP + BAR_H),
                  (80, 80, 80), 1)

    # Fill height
    fill_h   = int(BAR_H * level)
    fill_y   = BAR_Y_TOP + BAR_H - fill_h

    # Colour: green → yellow → orange → red
    if level < 0.30:
        bar_color = (0, 200, 80)            # green
    elif level < 0.65:
        bar_color = (0, 200, 220)           # yellow (BGR)
    elif level < 0.85:
        bar_color = (0, 120, 255)           # orange
    else:
        bar_color = (0, 50, 255)            # red

    # Pulse effect when listening — brighten the bar
    if listening:
        bar_color = tuple(min(255, int(c * 1.3)) for c in bar_color)

    if fill_h > 0:
        cv2.rectangle(frame,
                      (BAR_X, fill_y),
                      (BAR_X + BAR_W, BAR_Y_TOP + BAR_H),
                      bar_color, -1)

    # Threshold line at 65 % (wake-word detection sensitivity hint)
    thresh_y = BAR_Y_TOP + int(BAR_H * (1 - 0.65))
    cv2.line(frame,
             (BAR_X, thresh_y),
             (BAR_X + BAR_W, thresh_y),
             (255, 255, 100), 1)

    # Label
    cv2.putText(frame, "MIC", (BAR_X - 1, BAR_Y_TOP - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
    pct = int(level * 100)
    cv2.putText(frame, f"{pct}%", (BAR_X - 2, BAR_Y_TOP + BAR_H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)


def draw_hud(frame, assistant_text, listening, speaking):
    """Draw assistant status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 52
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-bar_h), (w, h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status indicator
    if listening:
        status, color = "● LISTENING", (0, 255, 100)
    elif speaking:
        status, color = "◉ SPEAKING",  (0, 200, 255)
    else:
        status, color = "◌ STANDBY",   (140, 140, 140)
    cv2.putText(frame, status, (10, h-32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Assistant text (truncated to fit)
    if assistant_text:
        max_chars = w // 11
        display   = assistant_text[:max_chars] + ("…" if len(assistant_text) > max_chars else "")
        cv2.putText(frame, display, (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)

    # ── Profile stats panel (top-right) ─────────────────────────────────────
    profile_ref = getattr(STATE, "_profile_ref", None)
    if profile_ref is not None:
        p     = profile_ref.profile
        stats = p["stats"]
        convos = stats["total_conversations"]
        n_facts = (len(p["preferences"]) + len(p["goals"]) +
                   len(p["entities"]["people"]) + len(p["interests"]["topics"]))
        sh = stats["sentiment_history"]
        mood = sh[-1]["sentiment"] if sh else "unknown"
        mood_color = {"positive":(0,220,120),"negative":(0,80,220),"neutral":(180,180,180)}.get(mood,(200,200,200))

        panel_x = w - 210
        cv2.rectangle(frame, (panel_x-4, 2), (w-2, 72), (25,25,25), -1)
        cv2.putText(frame, f"Profile: {n_facts} facts learned", (panel_x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,255,200), 1)
        cv2.putText(frame, f"Convos: {convos}", (panel_x, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,255), 1)
        cv2.putText(frame, f"Mood trend: {mood}", (panel_x, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, mood_color, 1)
        st = len(p.get("short_term", []))
        lt = len(p.get("long_term",  []))
        cv2.putText(frame, f"ST:{st}/{profile_ref.SHORT_TERM_MAX}  LT:{lt}/{profile_ref.LONG_TERM_MAX}", (panel_x, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
    else:
        cv2.putText(frame, "Profile loading...", (w-200, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)

    # User name top-left
    uname = STATE.user_name
    if uname:
        cv2.putText(frame, f"User: {uname}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,255), 1)


def camera_loop(args, stop_event: threading.Event):
    DeepFace    = load_deepface()
    recogniser, label_map = build_recogniser(args.known)
    mp_mesh     = build_mp()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    bored_trackers = {}

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {src}")
        stop_event.set()
        return

    frame_count = 0
    cache       = []
    prev_tick   = cv2.getTickCount()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        fh, fw = frame.shape[:2]
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Run detection every N frames ──────────────────────────────────────
        if frame_count % args.skip == 0:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            mp_res = None
            if mp_mesh:
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_res = mp_mesh.process(rgb)

            new_cache = []
            for fi, (x, y, w, h) in enumerate(faces):
                face_bgr  = frame[y:y+h, x:x+w]
                face_gray = gray[y:y+h, x:x+w]
                dominant, scores = analyse_emotion(face_bgr, DeepFace)

                name = ""
                if recogniser:
                    try:
                        resized = cv2.resize(face_gray, (200,200))
                        lid, conf = recogniser.predict(resized)
                        name = label_map.get(lid, "Unknown") if conf < 80 else "Unknown"
                    except Exception: pass

                bored_score = 0.0
                if mp_res and mp_res.multi_face_landmarks:
                    best, best_d = None, 1e9
                    cx, cy = x + w//2, y + h//2
                    for lm in mp_res.multi_face_landmarks:
                        dx = lm.landmark[1].x * fw - cx
                        dy = lm.landmark[1].y * fh - cy
                        d  = dx*dx + dy*dy
                        if d < best_d: best_d, best = d, lm
                    if best:
                        ear_v  = (ear(best.landmark, L_EYE, fw, fh) +
                                  ear(best.landmark, R_EYE, fw, fh)) / 2
                        mar_v  = mar(best.landmark, fw, fh)
                        pitch  = head_pitch(best.landmark, fw, fh)
                        if fi not in bored_trackers:
                            bored_trackers[fi] = BoredomTracker()
                        bored_score = bored_trackers[fi].update(ear_v, mar_v, pitch, dominant)

                if bored_score >= args.bored_threshold:
                    dominant = "bored"
                    scores["bored"] = float(bored_score)

                color = EMOTION_COLORS.get(dominant, DEFAULT_COLOR)
                new_cache.append((x, y, w, h, name, dominant, scores, bored_score, color))

                # Push primary face emotion to shared state
                if fi == 0:
                    STATE.set_emotion(dominant, scores, name, bored_score)

            cache = new_cache

        # Share raw (un-annotated) frame with assistant thread for Gemini Vision
        STATE.set_frame(frame)

        # ── Draw ─────────────────────────────────────────────────────────────
        for item in cache:
            draw_face(frame, *item)

        # ── HUD overlay ──────────────────────────────────────────────────────
        text, listening, speaking = STATE.get_ui()
        draw_noise_bar(frame)
        draw_hud(frame, text, listening, speaking)

        # ── FPS ──────────────────────────────────────────────────────────────
        now = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (now - prev_tick)
        prev_tick = now
        cv2.putText(frame, f"FPS {fps:.1f}", (fw-90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,100), 1)

        cv2.imshow("Emotion-Aware Assistant  [Q to quit]", frame)
        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Emotion-Aware Gemini Voice Assistant")
    parser.add_argument("--source",          default=0,
                        help="Webcam index or video path (default 0)")
    parser.add_argument("--skip",            type=int, default=3,
                        help="Emotion detection every N frames (default 3)")
    parser.add_argument("--known",           default="known_faces",
                        help="Directory of labelled face images")
    parser.add_argument("--bored-threshold", type=float, default=55.0,
                        help="Boredom score threshold 0-100 (default 55)")
    parser.add_argument("--profile", default="user_data/memory.json",
                        help="Path to profile JSON file (default: user_data/memory.json)")
    parser.add_argument("--api-key",         default="",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--wake-word",       default="hey gemini",
                        help="Phrase to activate the assistant (default: 'hey gemini')")
    parser.add_argument("--name",            default="",
                        help="Your name (skips the first-time prompt)")
    args = parser.parse_args()

    stop_event = threading.Event()

    # Start noise sampler thread
    noise_thread = threading.Thread(
        target=noise_sampler_thread, args=(stop_event,), daemon=True
    )
    noise_thread.start()

    # Start voice assistant in background thread
    va_thread = threading.Thread(
        target=assistant_thread, args=(args, stop_event), daemon=True
    )
    va_thread.start()

    # Camera loop runs on main thread (OpenCV needs the main thread on macOS)
    print("[INFO] Starting camera … press Q in the window to quit.")
    camera_loop(args, stop_event)

    stop_event.set()
    print("[INFO] Saving profile …")
    profile_ref = getattr(STATE, "_profile_ref", None)
    if profile_ref is not None:
        profile_ref.save_profile()
        print(profile_ref.get_profile_summary())
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    main()