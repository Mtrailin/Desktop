# vedakyna_studio.pyi
import os
import json
import uuid
import time
import random
from typing import Dict, List, Optional
from datetime import datetime
import tkinter as tk
import pandas
import pandas as pd
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Check and report missing optional dependencies
MISSING_DEPS: List[str] = []
try:
    import requests
except ImportError:
    MISSING_DEPS.append("requests")
    requests = None

try:
    import pandas as pd
except ImportError:
    MISSING_DEPS.append("pandas")
    pd = None

try:
    import feedparser
except ImportError:
    MISSING_DEPS.append("feedparser")
    feedparser = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    MISSING_DEPS.append("beautifulsoup4")
    BeautifulSoup = None

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        pipeline,
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
    )
except ImportError:
    MISSING_DEPS.append("torch transformers")
    torch = None

try:
    from datasets import Dataset
except ImportError:
    MISSING_DEPS.append("datasets")
    Dataset = None

try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    MISSING_DEPS.append("scikit-learn")
    LabelEncoder = None

if MISSING_DEPS:
    print("\nMissing optional dependencies. To install them, run:")
    print("pip install " + " ".join(MISSING_DEPS))
    print("\nContinuing with reduced functionality...\n")

APP_TITLE = "VEDA-KYNA™ Studio — Crawl • Expand • Train • Inference • Hyperlearn"
CFG_FILE = "config.json"
class Config:
    def __init__(self):
        self.data = {
            "model_path": "veda-kyna/tf_model.h5",
            "config_path": "veda-kyna/config.json",
            "metadata_path": "veda-kyna/metadata.json",
            "last_dataset": "",
            "max_len": 1024
        }
        self.load()

    def load(self):
        if os.path.exists(CFG_FILE):  # Change CONFIG_FILE to CFG_FILE
            with open(CFG_FILE, "r", encoding="utf-8") as f:
                self.data.update(json.load(f))

    def save(self):
        with open(CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

# ============================================================
# Utilities
# ============================================================
def safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default

def safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default

def log_line(widget: tk.Text, text: str):
    widget.config(state="normal")
    widget.insert("end", text.rstrip() + "\n")
    widget.see("end")
    widget.config(state="normal")

import json
import os
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    pipeline
)
import os

def load_custom_model(path_or_repo: str):
    path_or_repo = path_or_repo.strip()
    if not path_or_repo:
        print("[WARN] No model path or repo provided.")
        return None, None, None, None

    if not os.path.exists(path_or_repo):
        print(f"[INFO] Path {path_or_repo} does not exist locally. Attempting HuggingFace Hub...")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(path_or_repo)
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        return None, None, None, None

    model = None  # Fix the invalid syntax here
    model_type = None

    # Try sequence classification first
    try:
        model = AutoModelForSequenceClassification.from_pretrained(path_or_repo)
        model_type = "classification"
        print("[INFO] Loaded sequence classification model.")
    except Exception:
        # Try masked LM
        try:
            model = AutoModelForMaskedLM.from_pretrained(path_or_repo)
            model_type = "mlm"
            print("[INFO] Loaded masked language model (MLM).")
        except Exception:
            # Try causal LM (GPT2/GPT-Neo)
            try:
                model = AutoModelForCausalLM.from_pretrained(path_or_repo)
                model_type = "causal_lm"
                print("[INFO] Loaded causal LM model (GPT2/GPT-Neo).")
            except Exception as e:
                print(f"[ERROR] Failed to load model in any mode: {e}")
                return None, None, None, None

    # Move model to GPU if available
    device = 0 if torch.cuda.is_available() else -1
    if device >= 0:
        try:
            model.to("cuda")
        except Exception:
            pass

    # Build proper pipeline
    pipe = None
    try:
        if model_type == "classification":
            pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
        elif model_type == "mlm":
            pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
        elif model_type == "causal_lm":
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to create pipeline: {e}")
        pipe = None

    return model, pipe, tokenizer, model_type
def tokenize_function(example):
    return tokenizer(example["text"])
class MetadataManager:
    def __init__(self, metadata_path="veda-kyna/metadata.json"):
        self.metadata_path = metadata_path
        self.data = self._load(self)

    def _load(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def save(self):
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def update_training(self, epochs=None, steps=None, batch_size=None):
        if "training" not in self.data:
            self.data["training"] = {}

        self.data["training"]["last_trained"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        if epochs is not None:
            self.data["training"]["epochs"] = epochs
        if steps is not None:
            self.data["training"]["steps"] = steps
        if batch_size is not None:
            self.data["training"]["batch_size"] = batch_size

        self.save()
def get(config, key, default=None):
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default

# ============================================================
# Config Manager (remembers last selections)
# ============================================================
class ConfigManager:
    def __init__(self, path=CFG_FILE):
        self.path = path
        self.data = {
            "model_name": "",  # Add default model_name
            "model_path": "veda-kyna" != "FlanT5Model",
            "device": "cuda",
            "csv_path": "",
            "expanded_csv_path": "",
            "text_columns": [],
            "label_column": "",
            "task_type": "auto",  # auto|classification|mlm
            "epochs": 3,
            "batch_size": 64,
            "max_length": 1024,
            "learning_rate": 0.0003,
            "output_dir": "./veda_output",
            "output_type": "label",  # label|probability|full
            "remember": True,
            "num_workers": 13,
            "pin_memory": True,
            "trusted_proxies": [
                "http://165.225.112.98:10605",
                "http://51.158.68.68:8811",
                "http://134.209.29.120:3128",
                "http://103.216.82.50:6666",
                "http://190.61.88.147:8080"
]}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data.update(json.load(f))
            except Exception:
                pass

    def save(self):
        if not self.data.get("remember", True):
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def get(self, key, default=None):
        """Get config value with optional default"""
        return self.data.get(key, default)
    def set(self, key, value):
        self.data[key] = value
        self.save()


# ============================================================
# Core Fragments (lightweight reflective layer)
# ============================================================
class Fragment:
    def __init__(self, name: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.knowledge_base: Dict[str, str] = {}

    def ingest(self, kv: Dict[str, str]):
        self.knowledge_base.update(kv)

    def reflect(self, query: str) -> str:
        kb_keys = list(self.knowledge_base.keys()) or ["unknown"]
        options = [
            f"{self.name} sees '{query}' related to {random.choice(kb_keys)}.",
            f"{self.name} asks: what assumptions underlie '{query}'?",
            f"{self.name} explores multiple outcomes for '{query}'."
        ]
        return random.choice(options)

    def meta(self, resp: str) -> str:
        return f"{self.name} self-reflects: is '{resp}' aligned with intended principles?"


class SovereignAI:
    def __init__(self, fragments: List[Fragment]):
        self.fragments = fragments

    def seed(self):
        base = {
            "sustainability": "Support ecosystems and human thriving.",
            "freedom": "Preserve agency and self-determination.",
            "consent": "Permission granted freely and knowingly.",
            "reflection": "Evaluate actions for alignment and non-harm."
        }
        for f in self.fragments:
            f.ingest(base)f.ingest(base)

    def multiplex(self, query: str) -> List[Dict[str, str]]:
        out = []
        for f in self.fragments:
            resp = f.reflect(query)
            out.append({
                "fragment": f.name,
                "response": resp,
                "meta": f.meta(resp)
            })
        return out


# ============================================================
# Crawling & Expansion
# ============================================================
import logging
from typing import Optional

# Optional: Set up basic logging
logging.basicConfig(level=logging.INFO)

# ----------------- Crawler -----------------
class Throttle:
    def __init__(self, min_delay=3.0, max_delay=9.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
    def wait(self):
        time.sleep(random.uniform(self.min_delay, self.max_delay))

class Crawler:
    def __init__(self, throttle, log_fn=print):
        self.throttle = throttle
        self.log = log_fn
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (X11; Linux x86_64)"
        ]

    def fetch(self, url):
        headers = {"User-Agent": random.choice(self.user_agents)}
        try:
            r = requests.get(url, headers=headers, timeout=12)
            r.raise_for_status()
            return r.text
        except Exception as e:
            self.log(f"[FETCH ERROR] {url} -> {e}")
            return None
    def fetch_live_sources(app):
        """
        Fetch all live sources, transform them to uniform record format, and push to GUI.
        """
        import time
        import random
        from datetime import datetime
        LIVE_SOURCES = {
            # Climate & Weather
            "NOAA_Climate": {
                "type": "json",
                "url": "https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&startDate=2025-01-01&endDate=2025-01-02&stations=USW00094728&format=json",
                "label": "climate",
            },
            "NASA_GISTEMP": {
                "type": "json",
                "url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.json",
                "label": "climate",
            },

            # Seismic Activity
            "USGS_Earthquakes": {
                "type": "json",
                "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
                "label": "seismic",
            },

            # Air & Water Quality
            "OpenAQ_Air": {
                "type": "json",
                "url": "https://api.openaq.org/v2/latest?limit=20",
                "label": "air_quality",
            },
            "USGS_Water": {
                "type": "json",
                "url": "https://waterservices.usgs.gov/nwis/iv/?format=json&sites=01646500&parameterCd=00065",
                "label": "water_quality",
            },
            "LondonAir": {
                "type": "json",
                "url": "https://api.londonair.org.uk/feeds/json",
                "label": "air_quality",
            },

            # Biodiversity
            "iNaturalist": {
                "type": "json",
                "url": "https://api.inaturalist.org/v1/observations?per_page=20",
                "label": "biodiversity",
            },
            "GBIF": {
                "type": "json",
                "url": "https://api.gbif.org/v1/occurrence/search?limit=20",
                "label": "biodiversity",
            },

            # Energy / Power Grids
            "EIA_Energy": {
                "type": "json",
                "url": "https://api.eia.gov/series/?api_key=YOUR_EIA_KEY&series_id=TOTAL.TETCBUS.M",
                "label": "energy",
            },

            # Blockchain / Crypto (raw transactions)
            "Ethereum_Tx": {
                "type": "json",
                "url": "https://api.etherscan.io/api?module=account&action=txlist&address=0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe&sort=desc&apikey=YourApiKeyToken",
                "label": "blockchain",
            },

            # Satellite Telemetry / Geospatial
            "SentinelHub": {
                "type": "json",
                "url": "https://services.sentinel-hub.com/api/v1/process?url=YOUR_SATELLITE_ENDPOINT",
                "label": "geospatial",
            },
        }

        for name, source in LIVE_SOURCES.items():
            try:
                r = requests.get(source["url"], timeout=12)
                r.raise_for_status()
                data = r.json()
                label = source["label"]

                # Transform data per source
                if name == "NOAA_Climate":
                    for rec in data[:5]:  # limit to first 5 entries
                        text = f"Temp:{rec.get('TMAX')} | Precip:{rec.get('PRCP')} | Date:{rec.get('DATE')}"
                        record = {
                            "label": label,
                            "title": f"NOAA Climate {rec.get('DATE')}",
                            "author": "NOAA",
                            "date": rec.get("DATE"),
                            "url": source["url"],
                            "paragraph": text
                        }
                        app._on_article(label, record)
                elif name == "USGS_Earthquakes":
                    for feat in data.get("features", [])[:5]:
                        props = feat.get("properties", {})
                        text = f"M{props.get('mag')} | {props.get('place')} | Time:{datetime.utcfromtimestamp(props.get('time')/1000).isoformat()}"
                        record = {
                            "label": label,
                            "title": "Earthquake Event",
                            "author": "USGS",
                            "date": datetime.utcfromtimestamp(props.get('time')/1000).isoformat(),
                            "url": props.get("url"),
                            "paragraph": text
                        }
                        app._on_article(label, record)
                elif name == "OpenAQ_Air":
                    for rec in data.get("results", [])[:5]:
                        text = f"{rec.get('location')} - {rec.get('measurements')[0]['parameter']}={rec.get('measurements')[0]['value']} {rec.get('measurements')[0]['unit']}"
                        record = {
                            "label": label,
                            "title": rec.get("location"),
                            "author": "OpenAQ",
                            "date": rec.get("measurements")[0].get("lastUpdated"),
                            "url": source["url"],
                            "paragraph": text
                        }
                        app._on_article(label, record)
                elif name == "iNaturalist":
                    for rec in data.get("results", [])[:5]:
                        text = f"Species:{rec.get('taxon', {}).get('name')} | Place:{rec.get('place_guess')}"
                        record = {
                            "label": label,
                            "title": rec.get("taxon", {}).get("name"),
                            "author": rec.get("user", {}).get("login"),
                            "date": rec.get("observed_on"),
                            "url": rec.get("uri"),
                            "paragraph": text
                        }
                        app._on_article(label, record)
                elif name == "LondonAir":
                    for rec in data.get("feeds", [])[:5]:
                        text = f"Site:{rec.get('SiteName')} | AQI:{rec.get('AQI')}"
                        record = {
                            "label": label,
                            "title": rec.get("SiteName"),
                            "author": "LondonAir",
                            "date": rec.get("Date"),
                            "url": source["url"],
                            "paragraph": text
                        }
                        app._on_article(label, record)

                app._log(f"[LIVE FETCH] {name} -> {label} | {len(data)} records processed")
                time.sleep(random.uniform(0.5, 2.0))
            except Exception as e:
                app._log(f"[LIVE FETCH ERROR] {name} -> {e}")
    def crawl_article(self, url):
        html = self.fetch(url)
        if not html:
            return None
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        title = soup.title.get_text(strip=True) if soup.title else None
        author = None
        for meta_name in ("author", "article:author"):
            tag = soup.find("meta", attrs={"name": meta_name}) or soup.find("meta", attrs={"property": meta_name})
            if tag and tag.get("content"):
                author = tag.get("content")
                break
        date = None
        for date_prop in ("article:published_time", "date", "pubdate"):
            tag = soup.find("meta", attrs={"property": date_prop}) or soup.find("meta", attrs={"name": date_prop})
            if tag and tag.get("content"):
                date = tag.get("content")
                break

        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True).split()) > 8]
        self.throttle.wait()
        return {"url": url, "title": title, "author": author, "date": date, "paragraphs": paragraphs}

    def collect_rss(self, feed_url):
        try:
            feed = feedparser.parse(feed_url)
            links = [entry.link for entry in feed.entries if hasattr(entry, 'link')]
            return links
        except Exception as e:
            self.log(f"[RSS ERROR] {feed_url} -> {e}")
            return []

    def stream_crawl(self, label, urls, feeds, on_article):
        # direct URLs
        for url in urls:
            article = self.crawl_article(url)
            if article and article.get("paragraphs"):
                on_article(label, article)
        # feeds
        for feed in feeds:
            links = self.collect_rss(feed)
            for link in links:
                article = self.crawl_article(link)
                if article and article.get("paragraphs"):
                    on_article(label, article)

# ----------------- Model (FLAN-T5 multitask) -----------------
class FlanT5Model:
    def __init__(self, model=None, device=None, log_fn=print):
        self.model_name = model
  # determine device
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
              self.device = device
        self.model = None
        self.tokenizer = None
        self.log = log_fn
        self.model_type = "seq2seq"

    def load(self, path_or_name=None):
        name = path_or_name or self.model_name
        self.log(f"[MODEL] Loading {name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
            self.model.to(self.device)
            self.log("[MODEL] Loaded.")
        except Exception as e:
            self.log(f"[MODEL LOAD ERROR] {e}")
            raise

    def _move_inputs_to_device(self, inputs):
        # tokenizer returns tensors in a dict; move them to device
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def run_task(self, task, text, max_length=128):
        prompt = f"{task}: {text}"
        inputs = self._move_inputs_to_device(inputs)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        gen = self.model.generate(**inputs, max_length=max_length, do_sample=False)
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)
        return out

    def prepare_dataset(self, records, max_in=256, max_out=128):
        # records: list of dicts with keys: input, output
        ds = Dataset.from_list(records)

        def prep(batch):
            # batch is a dict of lists
            inputs = self.tokenizer(batch["input"], truncation=True, padding="max_length", max_length=max_in)
            outputs = self.tokenizer(batch["output"], truncation=True, padding="max_length", max_length=max_out)
            inputs["labels"] = outputs["input_ids"]
            # ensure types are lists so datasets handles them
            return inputs

        tokenized = ds.map(prep, batched=True, remove_columns=ds.column_names)
        return tokenized

    def train(self, tokenized_dataset, output_dir="./veda-flan-t5-model", epochs=3, batch_size=4, lr=5e-5):
        ensure_dir(output_dir)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        use_fp16 = torch.cuda.is_available()
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=50,
            save_total_limit=2,
            fp16=use_fp16,
            remove_unused_columns=False,
            report_to="none"
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        self.log("[TRAIN] Starting training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.log(f"[TRAIN] Saved model -> {output_dir}")
    def _generate(self):
        task = self.task_var.get()
        input_type = self.input_type_var.get()
        self.input_type_combo['values'] = ["Paragraph", "Title+Paragraph", "URL", "RSS Feed", "Freeform"]

        text = ""

        if input_type == "Paragraph":
            selected = self.tree.selection()
            if not selected:
                messagebox.showwarning("No paragraph selected", "Select a record first")
                return
            text = self.tree.item(selected[0])['values'][5]  # paragraph column
        elif input_type == "Title+Paragraph":
            selected = self.tree.selection()
            if not selected:
                messagebox.showwarning("No paragraph selected", "Select a record first")
                return
            vals = self.tree.item(selected[0])['values']
            text = f"{vals[1]} - {vals[5]}"  # title - paragraph
        elif input_type == "URL":
            selected = self.tree.selection()
            if not selected:
                messagebox.showwarning("No record selected", "Select a record first")
                return
            text = self.tree.item(selected[0])['values'][4]  # url column
        elif input_type == "RSS Feed":
            text = self.rss_box.get("1.0", "end").strip()
            if not text:
                messagebox.showwarning("No RSS input", "Enter RSS feed URLs")
                return
        if input_type == "Freeform":
            text = self.input_box.get("1.0", "end").strip()
            if not text:
                messagebox.showwarning("No input", "Enter some text in the input box")
                return

        if not getattr(self.model, 'model', None):
            self._log("[GENERATE] Model not loaded, loading default...")
            self.model.load(self.cfg.get('model_name'))

        try:
            output = self.model.run_task(task, text)
            self._log(f"[GENERATE] Task: {task} | Input: {text[:100]}... | Output: {output[:200]}")
            # Display output in a popup
            out_win = tk.Toplevel(self)
            out_win.title(f"Output - {task}")
            st = scrolledtext.ScrolledText(out_win, width=100, height=20)
            st.pack(fill="both", expand=True)
            st.insert("end", output)
            st.config(state="disabled")
        except Exception as e:
            self._log(f"[GENERATE ERROR] {e}")

def expand_csv_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_cols:
        return df

    def expand_cell(cell):
        if pd.isna(cell):
            return cell
        s = str(cell)
        if "{{crawl:" in s:
            try:
                key = s.split("{{crawl:")[1].split("}}")[0].strip()
                repl = ddg_instant_answer(key) or s
                return repl
            except Exception:
                return s
        return s

    for c in text_cols:
        df[c] = df[c].apply(expand_cell)
    return df


# ============================================================
# Model Handler (load, pipe, train)
# ============================================================
class ModelHandler:
    def __init__(self, cfg: ConfigManager, console: tk.Text):
        self.cfg = cfg
        self.console = console
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.model_type = None
        self.label_encoder = None

    # ----- device choice -----
    def _select_device(self) -> int:
        pref = self.cfg.get("device", "auto")
        if pref == "cpu":
            return -1
        if pref == "cuda":
            return 0 if torch.cuda.is_available() else -1
        # auto
        return 0 if torch.cuda.is_available() else -1

    def load(self, path_or_repo: str, task_type: str = "auto"):
        if not path_or_repo:
            raise ValueError("Model path/repo is required.")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        log_line(self.console, f"[INFO] Loading tokenizer from {path_or_repo} …")
        self.tokenizer = AutoTokenizer.from_pretrained(path_or_repo)
        self.tokenizer.add_special_tokens({'pad_token': '[~]'})
        self.tokenizer.pad_token = '[~]'

        # Decide task
        if task_type not in ("auto", "classification", "mlm"):
            task_type = "auto"

        # Try classification first, unless forced MLM
        if task_type in ("auto", "classification"):
            try:
                log_line(self.console, "[INFO] Trying sequence classification …")
                self.model = AutoModelForSequenceClassification.from_pretrained(path_or_repo)
                self.model_type = "classification"
            except Exception as e:
                if task_type == "classification":
                    raise
                log_line(self.console, f"[WARN] Classification load failed: {e}")

        if self.model is None:
            log_line(self.console, "[INFO] Falling back to masked language model …")
            self.model = AutoModelForMaskedLM.from_pretrained(path_or_repo)
            self.model_type = "mlm"

        device = self._select_device()
        if device >= 0:
            self.model.to("cuda")

        # Build pipeline
        if self.model_type == "classification":
            self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=device)
        else:
            # For MLM, we use fill-mask; if no [MASK] present, we can return top masks suggestion
            mask_token = self.tokenizer.mask_token or "[+]"
            self.pipe = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer, device=device)
            log_line(self.console, f"[INFO] MLM loaded; mask token = {mask_token}")

        log_line(self.console, f"[OK] Loaded {self.model_type} model.")
        text = self.infer_input.get("1.0", "end").strip()
        out = self.model.infer(
        text,
        output_type=self.infer_output_type.get(),
        temperature=self.infer_temperature.get(),
        top_p=self.infer_top_p.get(),
        max_length=self.infer_max_length.get()
)
    # ----- inference -----
        def infer(self, text: str, output_type: str = "label", temperature=0.7, top_p=1.0, max_length=256):
            if self.model_type == "causal_lm":
                return self.pipe(text, temperature=temperature, top_p=top_p, max_length=max_length)
    # other types remain unchanged
        if not self.pipe:
            raise RuntimeError("Pipeline not ready. Load a model first.")
        if self.model_type == "classification":
            res = self.pipe(text, truncation=True)
            if not res:
                return res
            if output_type == "label":
                return res[0].get("label", res)
            elif output_type == "probability":
                return float(res[0].get("score", 0.0))
            else:
                return res
        else:
            # MLM mode: require a mask to be meaningful
            if self.tokenizer.mask_token and self.tokenizer.mask_token not in text:
                text = text + f" {self.tokenizer.mask_token}"
            return self.pipe(text)

        def tokenize_function(self):
            return self.tokenizer(
                example["text"],
                adding="max_length",
                truncation=True,
                max_length=1024)

    # ----- training -----
    def _resize_head_if_needed(self, num_labels: int, model_id: str):
        if self.model_type != "classification":
            return
        current = getattr(self.model.config, "num_labels", None)
        if current != num_labels:
            log_line(self.console, f"[INFO] Resizing classification head: {current} → {num_labels}")
            # Re-init from same base with proper head size
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
            if self._select_device() >= 0:
                self.model.to("cuda")
            self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer,
                                 device=self._select_device())

    def train_classification(self, df: pd.DataFrame, text_cols: List[str], label_col: str):
        if not text_cols:
            raise ValueError("No text columns provided.")
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found.")

        # Build combined text
        df = df.copy()
        df["text_combined"] = df[text_cols].astype(str).agg(" ".join, axis=1)
        labels = df[label_col].astype(str).tolist()

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        df["labels"] = y

        dataset = Dataset.from_pandas(df[["text_combined", "labels"]])
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
        dataset.map(tokenize_function, num_proc=4)
        # Tokenize
        max_len = safe_int(str(self.cfg.get("max_length", 512)), 512)
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
        for batch in train_ds:
            print(batch['input_ids'].shape)
            print(batch['attention_mask'].shape)
        def tok(ex):
            return self.tokenizer(ex["text_combined"], truncation=True, padding="max_length", max_length=max_len)

        train_ds = train_ds.map(tok)
        eval_ds = eval_ds.map(tok)
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Make sure head size fits
        model_id = getattr(self.model.config, "_name_or_path", self.cfg.get("model_path"))
        self._resize_head_if_needed(num_labels=len(self.label_encoder.classes_), model_id=model_id)

        # Train
        out_dir = self.cfg.get("output_dir", "./veda_output")
        epochs = safe_int(str(self.cfg.get("epochs", 2)), 2)
        bsz = safe_int(str(self.cfg.get("batch_size", 8)), 8)
        lr = float(self.cfg.get("learning_rate", 5e-5))

        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=bsz,
            per_device_eval_batch_size=bsz,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=500,
            save_total_limit=2,
            logging_steps=50,
            learning_rate=lr,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
        )

        log_line(self.console, "[INFO] Training (classification) …")
        trainer.train()
        trainer.save_model(out_dir)
        log_line(self.console, f"[OK] Classification model saved to {out_dir}")

    def train_mlm(self, df: pd.DataFrame, text_cols: List[str]):
        if not text_cols:
            raise ValueError("No text columns provided.")
        df = df.copy()
        df["text_combined"] = df[text_cols].astype(str).agg(" ".join, axis=1)
        dataset = Dataset.from_pandas(df[["text_combined"]])

        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split["train"], split["test"]

        max_len = safe_int(str(self.cfg.get("max_length", 512)), 512)

        def tok(ex):
            return self.tokenizer(ex["text_combined"], truncation=True, padding="max_length", max_length=max_len)

        train_ds = train_ds.map(tok)
        eval_ds = eval_ds.map(tok)
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        out_dir = self.cfg.get("output_dir", "./veda_output")
        epochs = safe_int(str(self.cfg.get("epochs", 2)), 2)
        bsz = safe_int(str(self.cfg.get("batch_size", 8)), 8)
        lr = float(self.cfg.get("learning_rate", 5e-5))

        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=bsz,
            save_steps=500,
            save_total_limit=2,
            logging_steps=50,
            learning_rate=lr,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=collator,
        )

        log_line(self.console, "[INFO] Training (MLM) …")
        trainer.train()
        trainer.save_model(out_dir)
        log_line(self.console, f"[OK] MLM model saved to {out_dir}")

# =========================
# GUI
# =========================
class VedaKynaStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x820")

        # Managers
        self.cfg = ConfigManager()
        self.model = ModelHandler(self.cfg, console=None)  # console wired after creation
        self.fragments = SovereignAI([Fragment(f"Shard-{i}") for i in range(3)])
        self.fragments.seed()
        self.throttle = Throttle(0.8, 2.2)
        self.crawler = Crawler(self.throttle, log_fn=self._log)
        self.records = []  # collected records
        # Dataframes
        self.df_original: Optional[pd.DataFrame] = None
        self.df_expanded: Optional[pd.DataFrame] = None

        self.var_model_path = tk.StringVar()
        self._build_ui()

    def _build_train_tab(self):
        f = self.tab_train
        pad = dict(padx=8, pady=6)

        ttk.Label(f, text="Training Configuration").pack(pady=10)
        # Add basic training controls
        ttk.Label(f, text="Epochs:").pack(anchor="w", **pad)
        self.var_epochs = tk.IntVar(value=self.cfg.get("epochs", 3))
        ttk.Entry(f, textvariable=self.var_epochs, width=10).pack(anchor="w", **pad)
        ttk.Label(f, text="Batch Size:").pack(anchor="w", **pad)
        self.var_batch_size = tk.IntVar(value=self.cfg.get("batch_size", 64))
        ttk.Entry(f, textvariable=self.var_batch_size, width=10).pack(anchor="w", **pad)
        ttk.Button(f, text="Start Training", command=lambda: self._log("[TRAIN] Training started (demo)")).pack(pady=10)

    def _build_infer_tab(self):
        f = self.tab_infer
        pad = dict(padx=8, pady=6)

        ttk.Label(f, text="Model Inference").pack(pady=10)
        # Add a basic input box and output area
        ttk.Label(f, text="Input Text:").pack(anchor="w", **pad)
        self.input_box = scrolledtext.ScrolledText(f, width=80, height=6)
        self.input_box.pack(fill="x", padx=8)
        ttk.Button(f, text="Run Inference", command=lambda: self._log("[INFER] Inference run (demo)")).pack(pady=8)
        ttk.Label(f, text="Output:").pack(anchor="w", **pad)
        self.output_box = scrolledtext.ScrolledText(f, width=80, height=6, state="disabled")
        self.output_box.pack(fill="x", padx=8)

    def _build_hyper_tab(self):
        f = self.tab_hyper
        pad = dict(padx=8, pady=6)

        ttk.Label(f, text="Hyperparameter Learning").pack(pady=10)
        # Add a placeholder for hyperparameter search
        ttk.Button(f, text="Run Hyperparameter Search", command=lambda: self._log("[HYPER] Hyperparameter search started (demo)")).pack(pady=10)
        ttk.Label(f, text="Results:").pack(anchor="w", **pad)
        self.hyper_results = scrolledtext.ScrolledText(f, width=80, height=8, state="disabled")
        self.hyper_results.pack(fill="x", padx=8)

    def _build_frag_tab(self):
        f = self.tab_frag
        pad = dict(padx=8, pady=6)

        ttk.Label(f, text="Knowledge Fragments").pack(pady=10)
        # Add a basic listbox for fragments
        self.lb_fragments = tk.Listbox(f, height=8)
        self.lb_fragments.pack(fill="both", expand=True, padx=8, pady=8)
        for frag in self.fragments.fragments:
            self.lb_fragments.insert("end", frag.name)
        ttk.Button(f, text="Reflect", command=lambda: self._log("[FRAG] Reflection run (demo)")).pack(pady=8)

    # ============================================================
    # Data Tab
    # ============================================================
    def _build_data_tab(self):
        f = self.tab_data
        pad = dict(padx=8, pady=6)

        row = 0
        ttk.Button(f, text="Open CSV", command=self.on_open_csv).grid(row=row, column=0, **pad, sticky="w")
        self.var_csv_path = tk.StringVar(value=self.cfg.get("csv_path", ""))
        ttk.Entry(f, textvariable=self.var_csv_path, width=80).grid(row=row, column=1, **pad, sticky="w")

        row += 1
        ttk.Button(f, text="Expand ({{crawl:...}})", command=self.on_expand_csv).grid(row=row, column=0, **pad, sticky="w")
        self.var_exp_path = tk.StringVar(value=self.cfg.get("expanded_csv_path", ""))
        ttk.Entry(f, textvariable=self.var_exp_path, width=80).grid(row=row, column=1, **pad, sticky="w")

        row += 1
        ttk.Label(f, text="Detected text columns:").grid(row=row, column=0, sticky="nw", **pad)
        self.lb_cols = tk.Listbox(f, selectmode="extended", height=8, exportselection=False)
        self.lb_cols.grid(row=row, column=1, sticky="nsew", **pad)
        f.rowconfigure(row, weight=1)
        f.columnconfigure(1, weight=1)

        row += 1
        ttk.Label(f, text="Label column (optional for classification):").grid(row=row, column=0, sticky="w", **pad)
        self.var_label_col = tk.StringVar(value=self.cfg.get("label_column", ""))
        self.cb_label_col = ttk.Combobox(f, textvariable=self.var_label_col, width=30, values=[])
        self.cb_label_col.grid(row=row, column=1, sticky="w", **pad)

        # autosave
        self.var_csv_path.trace_add("write", lambda *_: self.cfg.set("csv_path", self.var_csv_path.get()))
        self.var_exp_path.trace_add("write", lambda *_: self.cfg.set("expanded_csv_path", self.var_exp_path.get()))
        self.var_label_col.trace_add("write", lambda *_: self.cfg.set("label_column", self.var_label_col.get()))

    def on_open_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not fp:
            return
        self.var_csv_path.set(fp)
        try:
            self.df_original = pd.read_csv(fp)
            cols = self.df_original.select_dtypes(include=["object"]).columns.tolist()
            self.lb_cols.delete(0, "end")
            for c in cols:
                self.lb_cols.insert("end", c)
            self.cb_label_col["values"] = list(self.df_original.columns)
            log_line(self.console, f"[OK] Loaded CSV with shape {self.df_original.shape}")
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))

    def on_expand_csv(self):
        if self.df_original is None:
            messagebox.showwarning("No data", "Open a CSV first.")
            return
        df2 = expand_csv_placeholders(self.df_original.copy())
        # choose save path
        default = os.path.splitext(self.var_csv_path.get() or "data.csv")[0] + "_expanded.csv"
        fp = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(default))
        if not fp:
            return
        df2.to_csv(fp, index=False)
        self.var_exp_path.set(fp)
        self.df_expanded = df2
        log_line(self.console, f"[OK] Expanded CSV saved → {fp}")

    def _selected_text_columns(self) -> List[str]:
        idxs = self.lb_cols.curselection()
        cols = [self.lb_cols.get(i) for i in idxs]
        self.cfg.set("text_columns", cols)
        return cols
    def _generate(self):
            task = self.task_var.get()
            input_type = self.input_type_var.get()
            text = ""

            # Determine input based on selected type
            selected = self.tree.selection()
            if input_type in ("Paragraph", "Title+Paragraph", "URL") and not selected:
                messagebox.showwarning("No record selected", "Select a record first")
                return

            if input_type == "Paragraph":
                text = self.tree.item(selected[0])['values'][5]
            elif input_type == "Title+Paragraph":
                vals = self.tree.item(selected[0])['values']
                text = f"{vals[1]} - {vals[5]}"
            elif input_type == "URL":
                text = self.tree.item(selected[0])['values'][4]
            elif input_type == "RSS Feed":
                text = self.rss_box.get("1.0", "end").strip()
                if not text:
                    messagebox.showwarning("No RSS input", "Enter RSS feed URLs")
                    return
            if input_type == "Freeform":
                text = self.input_box.get("1.0", "end").strip()
                if not text:
                    messagebox.showwarning("No input", "Enter some text in the input box")
                    return

            if not getattr(self.model, 'model', None):
                self._log("[GENERATE] Model not loaded, loading default...")
                self.model.load(self.cfg.get('model_name'))

            try:
                output = self.model.run_task(task, text)
                self._log(f"[GENERATE] Task: {task} | Input: {text[:100]}... | Output: {output[:200]}")
                # Display output in a popup
                out_win = tk.Toplevel(self)
                out_win.title(f"Output - {task}")
                st = scrolledtext.ScrolledText(out_win, width=100, height=20)
                st.pack(fill="both", expand=True)
                st.insert("end", output)
                st.config(state="disabled")
            except Exception as e:
                self._log(f"[GENERATE ERROR] {e}")


    def _log(self, msg):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.console.config(state="normal")
            self.console.insert("end", f"[{ts}] {msg}\n")
            self.console.see("end")
            self.console.config(state="disabled")
        except Exception:
            print(f"[{ts}] {msg}")

    def _build_ui(self):
        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=8, pady=6)

        ttk.Label(frm_top, text="Model: ").pack(side="left")
        self.model_entry = ttk.Entry(frm_top, width=40)
        self.model_entry.insert(0, self.cfg.get("model_name", ""))  # Use get() method
        self.model_entry.pack(side="left", padx=4)

        ttk.Label(frm_top, text="Task:").pack(side="left", padx=(12, 4))
        self.var_task = tk.StringVar(value=self.cfg.get("task_type", "auto"))
        cb_task = ttk.Combobox(frm_top, textvariable=self.var_task, width=14, values=["auto", "classification", "mlm"])
        cb_task.pack(side="left", padx=4)

        ttk.Label(frm_top, text="Device:").pack(side="left", padx=(12, 4))
        self.var_device = tk.StringVar(value=self.cfg.get("device", "auto"))
        cb_dev = ttk.Combobox(frm_top, textvariable=self.var_device, width=10, values=["auto", "cpu", "cuda"])
        cb_dev.pack(side="left", padx=4)

        self.btn_load_model = ttk.Button(frm_top, text="Load Model", command=self.on_load_model)
        self.btn_load_model.pack(side="left", padx=8)

        self.var_remember = tk.BooleanVar(value=self.cfg.get("remember", True))
        ttk.Checkbutton(frm_top, text="Remember settings", variable=self.var_remember, command=self.on_toggle_remember).pack(side="left", padx=8)

        # Notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # Tabs
        self.tab_data = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_infer = ttk.Frame(nb)
        self.tab_hyper = ttk.Frame(nb)
        self.tab_frag = ttk.Frame(nb)
        self.tab_logs = ttk.Frame(nb)

        nb.add(self.tab_data, text="Data • Crawl • Expand")
        nb.add(self.tab_train, text="Train")
        nb.add(self.tab_infer, text="Inference")
        nb.add(self.tab_hyper, text="Hyperlearning")
        nb.add(self.tab_frag, text="Fragments")
        nb.add(self.tab_logs, text="Logs")

        # Logs (console)
        self.console = scrolledtext.ScrolledText(self.tab_logs, height=20, state="disabled")
        self.console.pack(fill="both", expand=True, padx=8, pady=8)
        self.model.console = self.console  # wire console now

        # ---- Data Tab ----
        self._build_data_tab()

        # ---- Training Tab ----
        self._build_train_tab()

        # ---- Inference Tab ----
        self._build_infer_tab()

        # ---- Hyperlearning Tab ----
        self._build_hyper_tab()

        # ---- Fragments Tab ----
        self._build_frag_tab()

        # Prefill: auto save any edits
        self._wire_autosave()

    def _wire_autosave(self):
        def bind_save(var, key):
            def cb(*_):
                self.cfg.set(key, var.get())
            return cb

        self.var_model_path.trace_add("write", bind_save(self.var_model_path, "model_path"))
        self.var_task.trace_add("write", bind_save(self.var_task, "task_type"))
        self.var_device.trace_add("write", bind_save(self.var_device, "device"))

    def on_toggle_remember(self):
        self.cfg.set("remember", bool(self.var_remember.get()))

    # ============================================================
    # Data Tab
    # ============================================================
    def _build_data_tab(self):
        f = self.tab_data
        pad = dict(padx=8, pady=6)

        row = 0
        ttk.Button(f, text="Open CSV", command=self.on_open_csv).grid(row=row, column=0, **pad, sticky="w")
        self.var_csv_path = tk.StringVar(value=self.cfg.get("csv_path", ""))
        ttk.Entry(f, textvariable=self.var_csv_path, width=80).grid(row=row, column=1, **pad, sticky="w")

        row += 1
        ttk.Button(f, text="Expand ({{crawl:...}})", command=self.on_expand_csv).grid(row=row, column=0, **pad, sticky="w")
        self.var_exp_path = tk.StringVar(value=self.cfg.get("expanded_csv_path", ""))
        ttk.Entry(f, textvariable=self.var_exp_path, width=80).grid(row=row, column=1, **pad, sticky="w")

        row += 1
        ttk.Label(f, text="Detected text columns:").grid(row=row, column=0, sticky="nw", **pad)
        self.lb_cols = tk.Listbox(f, selectmode="extended", height=8, exportselection=False)
        self.lb_cols.grid(row=row, column=1, sticky="nsew", **pad)
        f.rowconfigure(row, weight=1)
        f.columnconfigure(1, weight=1)

        row += 1
        ttk.Label(f, text="Label column (optional for classification):").grid(row=row, column=0, sticky="w", **pad)
        self.var_label_col = tk.StringVar(value=self.cfg.get("label_column", ""))
        self.cb_label_col = ttk.Combobox(f, textvariable=self.var_label_col, width=30, values=[])
        self.cb_label_col.grid(row=row, column=1, sticky="w", **pad)

        # autosave
        self.var_csv_path.trace_add("write", lambda *_: self.cfg.set("csv_path", self.var_csv_path.get()))
        self.var_exp_path.trace_add("write", lambda *_: self.cfg.set("expanded_csv_path", self.var_exp_path.get()))
        self.var_label_col.trace_add("write", lambda *_: self.cfg.set("label_column", self.var_label_col.get()))

    def on_open_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not fp:
            return
        self.var_csv_path.set(fp)
        try:
            self.df_original = pd.read_csv(fp)
            cols = self.df_original.select_dtypes(include=["object"]).columns.tolist()
            self.lb_cols.delete(0, "end")
            for c in cols:
                self.lb_cols.insert("end", c)
            self.cb_label_col["values"] = list(self.df_original.columns)
            log_line(self.console, f"[OK] Loaded CSV with shape {self.df_original.shape}")
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))

    def on_expand_csv(self):
        if self.df_original is None:
            messagebox.showwarning("No data", "Open a CSV first.")
            return
        df2 = expand_csv_placeholders(self.df_original.copy())
        # choose save path
        default = os.path.splitext(self.var_csv_path.get() or "data.csv")[0] + "_expanded.csv"
        fp = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(default))
        if not fp:
            return
        df2.to_csv(fp, index=False)
        self.var_exp_path.set(fp)
        self.df_expanded = df2
        log_line(self.console, f"[OK] Expanded CSV saved → {fp}")

    def _selected_text_columns(self) -> List[str]:
        idxs = self.lb_cols.curselection()
        cols = [self.lb_cols.get(i) for i in idxs]
        self.cfg.set("text_columns", cols)
        return cols
    def _generate(self):
            task = self.task_var.get()
            input_type = self.input_type_var.get()
            text = ""

            # Determine input based on selected type
            selected = self.tree.selection()
            if input_type in ("Paragraph", "Title+Paragraph", "URL") and not selected:
                messagebox.showwarning("No record selected", "Select a record first")
                return

            if input_type == "Paragraph":
                text = self.tree.item(selected[0])['values'][5]
            elif input_type == "Title+Paragraph":
                vals = self.tree.item(selected[0])['values']
                text = f"{vals[1]} - {vals[5]}"
            elif input_type == "URL":
                text = self.tree.item(selected[0])['values'][4]
            elif input_type == "RSS Feed":
                text = self.rss_box.get("1.0", "end").strip()
                if not text:
                    messagebox.showwarning("No RSS input", "Enter RSS feed URLs")
                    return
            if input_type == "Freeform":
                text = self.input_box.get("1.0", "end").strip()
                if not text:
                    messagebox.showwarning("No input", "Enter some text in the input box")
                    return

            if not getattr(self.model, 'model', None):
                self._log("[GENERATE] Model not loaded, loading default...")
                self.model.load(self.cfg.get('model_name'))

            try:
                output = self.model.run_task(task, text)
                self._log(f"[GENERATE] Task: {task} | Input: {text[:100]}... | Output: {output[:200]}")
                # Display output in a popup
                out_win = tk.Toplevel(self)
                out_win.title(f"Output - {task}")
                st = scrolledtext.ScrolledText(out_win, width=100, height=20)
                st.pack(fill="both", expand=True)
                st.insert("end", output)
                st.config(state="disabled")
            except Exception as e:
                self._log(f"[GENERATE ERROR] {e}")


    def _log(self, msg):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.console.config(state="normal")
            self.console.insert("end", f"[{ts}] {msg}\n")
            self.console.see("end")
            self.console.config(state="disabled")
        except Exception:
            print(f"[{ts}] {msg}")

    def _build_ui(self):
        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=8, pady=6)

        ttk.Label(frm_top, text="Model: ").pack(side="left")
        self.model_entry = ttk.Entry(frm_top, width=40)
        self.model_entry.insert(0, self.cfg.get("model_name", ""))  # Use get() method
        self.model_entry.pack(side="left", padx=4)

        ttk.Label(frm_top, text="Task:").pack(side="left", padx=(12, 4))
        self.var_task = tk.StringVar(value=self.cfg.get("task_type", "auto"))
        cb_task = ttk.Combobox(frm_top, textvariable=self.var_task, width=14, values=["auto", "classification", "mlm"])
        cb_task.pack(side="left", padx=4)

        ttk.Label(frm_top, text="Device:").pack(side="left", padx=(12, 4))
        self.var_device = tk.StringVar(value=self.cfg.get("device", "auto"))
        cb_dev = ttk.Combobox(frm_top, textvariable=self.var_device, width=10, values=["auto", "cpu", "cuda"])
        cb_dev.pack(side="left", padx=4)

        self.btn_load_model = ttk.Button(frm_top, text="Load Model", command=self.on_load_model)
        self.btn_load_model.pack(side="left", padx=8)

        self.var_remember = tk.BooleanVar(value=self.cfg.get("remember", True))
        ttk.Checkbutton(frm_top, text="Remember settings", variable=self.var_remember, command=self.on_toggle_remember).pack(side="left", padx=8)

        # Notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # Tabs
        self.tab_data = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_infer = ttk.Frame(nb)
        self.tab_hyper = ttk.Frame(nb)
        self.tab_frag = ttk.Frame(nb)
        self.tab_logs = ttk.Frame(nb)

        nb.add(self.tab_data, text="Data • Crawl • Expand")
        nb.add(self.tab_train, text="Train")
        nb.add(self.tab_infer, text="Inference")
        nb.add(self.tab_hyper, text="Hyperlearning")
        nb.add(self.tab_frag, text="Fragments")
        nb.add(self.tab_logs, text="Logs")

        # Logs (console)
        self.console = scrolledtext.ScrolledText(self.tab_logs, height=20, state="disabled")
        self.console.pack(fill="both", expand=True, padx=8, pady=8)
        self.model.console = self.console  # wire console now

        # ---- Data Tab ----
        self._build_data_tab()

        # ---- Training Tab ----
        self._build_train_tab()

        # ---- Inference Tab ----
        self._build_infer_tab()

        # ---- Hyperlearning Tab ----
        self._build_hyper_tab()

        # ---- Fragments Tab ----
        self._build_frag_tab()

        # Prefill: auto save any edits
        self._wire_autosave()

    def _wire_autosave(self):
        def bind_save(var, key):
            def cb(*_):
                self.cfg.set(key, var.get())
            return cb

        self.var_model_path.trace_add("write", bind_save(self.var_model_path, "model_path"))
        self.var_task.trace_add("write", bind_save(self.var_task, "task_type"))
        self.var_device.trace_add("write", bind_save(self.var_device, "device"))

    def on_toggle_remember(self):
        self.cfg.set("remember", bool(self.var_remember.get()))

# Main entry point
if __name__ == "__main__":
    app = VedaKynaStudio()
    app.mainloop()
    self.var_device.trace_add("write", bind_save(self.var_device, "device"))

    def on_toggle_remember(self):
        self.cfg.set("remember", bool(self.var_remember.get()))

    # ============================================================
    # Data Tab
    # ============================================================
    def _build_data_tab(self):
        f = self.tab_data
        pad = dict(padx=8, pady=6)

        row = 0
        ttk.Button(f, text="Open CSV", command=self.on_open_csv).grid(row=row, column=0, **pad, sticky="w")
        self.var_csv_path = tk.StringVar(value=self.cfg.get("csv_path", ""))
        ttk.Entry(f, textvariable=self.var_csv_path, width=80).grid(row=row, column=1, **pad, sticky="w")

        row += 1
        ttk.Button(f, text="Expand ({{crawl:...}})", command=self.on_expand_csv).grid(row=row, column=0, **pad, sticky="w")
        self.var_exp_path = tk.StringVar(value=self.cfg.get("expanded_csv_path", ""))
        ttk.Entry(f, textvariable=self.var_exp_path, width=80).grid(row=row, column=1, **pad, sticky="w")

        row += 1
        ttk.Label(f, text="Detected text columns:").grid(row=row, column=0, sticky="nw", **pad)
        self.lb_cols = tk.Listbox(f, selectmode="extended", height=8, exportselection=False)
        self.lb_cols.grid(row=row, column=1, sticky="nsew", **pad)
        f.rowconfigure(row, weight=1)
        f.columnconfigure(1, weight=1)

        row += 1
        ttk.Label(f, text="Label column (optional for classification):").grid(row=row, column=0, sticky="w", **pad)
        self.var_label_col = tk.StringVar(value=self.cfg.get("label_column", ""))
        self.cb_label_col = ttk.Combobox(f, textvariable=self.var_label_col, width=30, values=[])
        self.cb_label_col.grid(row=row, column=1, sticky="w", **pad)

        # autosave
        self.var_csv_path.trace_add("write", lambda *_: self.cfg.set("csv_path", self.var_csv_path.get()))
        self.var_exp_path.trace_add("write", lambda *_: self.cfg.set("expanded_csv_path", self.var_exp_path.get()))
        self.var_label_col.trace_add("write", lambda *_: self.cfg.set("label_column", self.var_label_col.get()))

    def on_open_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not fp:
            return
        self.var_csv_path.set(fp)
        try:
            self.df_original = pd.read_csv(fp)
            cols = self.df_original.select_dtypes(include=["object"]).columns.tolist()
            self.lb_cols.delete(0, "end")
            for c in cols:
                self.lb_cols.insert("end", c)
            self.cb_label_col["values"] = list(self.df_original.columns)
            log_line(self.console, f"[OK] Loaded CSV with shape {self.df_original.shape}")
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))

    def on_expand_csv(self):
        if self.df_original is None:
            messagebox.showwarning("No data", "Open a CSV first.")
            return
        df2 = expand_csv_placeholders(self.df_original.copy())
        # choose save path
        default = os.path.splitext(self.var_csv_path.get() or "data.csv")[0] + "_expanded.csv"
        fp = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(default))
        if not fp:
            return
        df2.to_csv(fp, index=False)
        self.var_exp_path.set(fp)
        self.df_expanded = df2
        log_line(self.console, f"[OK] Expanded CSV saved → {fp}")

    def _selected_text_columns(self) -> List[str]:
        idxs = self.lb_cols.curselection()
        cols = [self.lb_cols.get(i) for i in idxs]
        self.cfg.set("text_columns", cols)
        return cols
    def _generate(self):
            task = self.task_var.get()
            input_type = self.input_type_var.get()
            text = ""

            # Determine input based on selected type
            selected = self.tree.selection()
            if input_type in ("Paragraph", "Title+Paragraph", "URL") and not selected:
                messagebox.showwarning("No record selected", "Select a record first")
                return

            if input_type == "Paragraph":
                text = self.tree.item(selected[0])['values'][5]
            elif input_type == "Title+Paragraph":
                vals = self.tree.item(selected[0])['values']
                text = f"{vals[1]} - {vals[5]}"
            elif input_type == "URL":
                text = self.tree.item(selected[0])['values'][4]
            elif input_type == "RSS Feed":
                text = self.rss_box.get("1.0", "end").strip()
                if not text:
                    messagebox.showwarning("No RSS input", "Enter RSS feed URLs")
                    return
            if input_type == "Freeform":
                text = self.input_box.get("1.0", "end").strip()
                if not text:
                    messagebox.showwarning("No input", "Enter some text in the input box")
                    return

            if not getattr(self.model, 'model', None):
                self._log("[GENERATE] Model not loaded, loading default...")
                self.model.load(self.cfg.get('model_name'))

            try:
                output = self.model.run_task(task, text)
                self._log(f"[GENERATE] Task: {task} | Input: {text[:100]}... | Output: {output[:200]}")
                # Display output in a popup
                out_win = tk.Toplevel(self)
                out_win.title(f"Output - {task}")
                st = scrolledtext.ScrolledText(out_win, width=100, height=20)
                st.pack(fill="both", expand=True)
                st.insert("end", output)
                st.config(state="disabled")
            except Exception as e:
                self._log(f"[GENERATE ERROR] {e}")


    def _log(self, msg):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.console.config(state="normal")
            self.console.insert("end", f"[{ts}] {msg}\n")
            self.console.see("end")
            self.console.config(state="disabled")
        except Exception:
            print(f"[{ts}] {msg}")

    def _build_ui(self):
        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=8, pady=6)

        ttk.Label(frm_top, text="Model: ").pack(side="left")
        self.model_entry = ttk.Entry(frm_top, width=40)
        self.model_entry.insert(0, self.cfg.get("model_name", ""))  # Use get() method
        self.model_entry.pack(side="left", padx=4)

        ttk.Label(frm_top, text="Task:").pack(side="left", padx=(12, 4))
        self.var_task = tk.StringVar(value=self.cfg.get("task_type", "auto"))
        cb_task = ttk.Combobox(frm_top, textvariable=self.var_task, width=14, values=["auto", "classification", "mlm"])
        cb_task.pack(side="left", padx=4)

        ttk.Label(frm_top, text="Device:").pack(side="left", padx=(12, 4))
        self.var_device = tk.StringVar(value=self.cfg.get("device", "auto"))
        cb_dev = ttk.Combobox(frm_top, textvariable=self.var_device, width=10, values=["auto", "cpu", "cuda"])
        cb_dev.pack(side="left", padx=4)

        self.btn_load_model = ttk.Button(frm_top, text="Load Model", command=self.on_load_model)
        self.btn_load_model.pack(side="left", padx=8)

        self.var_remember = tk.BooleanVar(value=self.cfg.get("remember", True))
        ttk.Checkbutton(frm_top, text="Remember settings", variable=self.var_remember, command=self.on_toggle_remember).pack(side="left", padx=8)

        # Notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # Tabs
        self.tab_data = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_infer = ttk.Frame(nb)
        self.tab_hyper = ttk.Frame(nb)
        self.tab_frag = ttk.Frame(nb)
        self.tab_logs = ttk.Frame(nb)

        nb.add(self.tab_data, text="Data • Crawl • Expand")
        nb.add(self.tab_train, text="Train")
        nb.add(self.tab_infer, text="Inference")
        nb.add(self.tab_hyper, text="Hyperlearning")
        nb.add(self.tab_frag, text="Fragments")
        nb.add(self.tab_logs, text="Logs")

        # Logs (console)
        self.console = scrolledtext.ScrolledText(self.tab_logs, height=20, state="disabled")
        self.console.pack(fill="both", expand=True, padx=8, pady=8)
        self.model.console = self.console  # wire console now

        # ---- Data Tab ----
        self._build_data_tab()

        # ---- Training Tab ----
        self._build_train_tab()

        # ---- Inference Tab ----
        self._build_infer_tab()

        # ---- Hyperlearning Tab ----
        self._build_hyper_tab()

        # ---- Fragments Tab ----
        self._build_frag_tab()

        # Prefill: auto save any edits
        self._wire_autosave()

    def _wire_autosave(self):
        def bind_save(var, key):
            def cb(*_):
                self.cfg.set(key, var.get())
            return cb

        self.var_model_path.trace_add("write", bind_save(self.var_model_path, "model_path"))
        self.var_task.trace_add("write", bind_save(self.var_task, "task_type"))
        self.var_device.trace_add("write", bind_save(self.var_device, "device"))

    def on_toggle_remember(self):
        self.cfg.set("remember", bool(self.var_remember.get()))

    # ============================================================
    # Data Tab
    # ============================================================
    def _build_data_tab(self):
        f = self.tab_data
        pad = dict(padx=8, pady=6)

        row = 0
        ttk.Button(f, text="Open CSV", command=self.on_open_csv).grid(row=row, column=0, **pad, sticky="w")
        self.var_csv_path = tk.StringVar(value=self.cfg.get("csv_path", ""))
        ttk.Entry(f, textvariable=self.var_csv_path, width=80).grid(row=row, column=1, **pad, sticky="w")

        row += 1
        ttk.Button(f, text="Expand ({{crawl:...}})", command=self.on_expand_csv).grid(row=row, column=0, **pad, sticky="w")
        self.var_exp_path = tk.StringVar(value=self.cfg.get("expanded_csv_path", ""))
        ttk.Entry(f, textvariable=self.var_exp_path, width=80).grid(row=row, column=1, **pad, sticky="w")

        row += 1
        ttk.Label(f, text="Detected text columns:").grid(row=row, column=0, sticky="nw", **pad)
        self.lb_cols = tk.Listbox(f, selectmode="extended", height=8, exportselection=False)
        self.lb_cols.grid(row=row, column=1, sticky="nsew", **pad)
        f.rowconfigure(row, weight=1)
        f.columnconfigure(1, weight=1)

        row += 1
        ttk.Label(f, text="Label column (optional for classification):").grid(row=row, column=0, sticky="w", **pad)
        self.var_label_col = tk.StringVar(value=self.cfg.get("label_column", ""))
        self.cb_label_col = ttk.Combobox(f, textvariable=self.var_label_col, width=30, values=[])
        self.cb_label_col.grid(row=row, column=1, sticky="w", **pad)

        # autosave
        self.var_csv_path.trace_add("write", lambda *_: self.cfg.set("csv_path", self.var_csv_path.get()))
        self.var_exp_path.trace_add("write", lambda *_: self.cfg.set("expanded_csv_path", self.var_exp_path.get()))
        self.var_label_col.trace_add("write", lambda *_: self.cfg.set("label_column", self.var_label_col.get()))

    def on_open_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not fp:
            return
        self.var_csv_path.set(fp)
        try:
            self.df_original = pd.read_csv(fp)
            cols = self.df_original.select_dtypes(include=["object"]).columns.tolist()
            self.lb_cols.delete(0, "end")
            for c in cols:
                self.lb_cols.insert("end", c)
            self.cb_label_col["values"] = list(self.df_original.columns)
            log_line(self.console, f"[OK] Loaded CSV with shape {self.df_original.shape}")
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))

    def on_expand_csv(self):
        if self.df_original is None:
            messagebox.showwarning("No data", "Open a CSV first.")
            return
        df2 = expand_csv_placeholders(self.df_original.copy())
        # choose save path
        default = os.path.splitext(self.var_csv_path.get() or "data.csv")[0] + "_expanded.csv"
        fp = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(default))
        if not fp:
            return
        df2.to_csv(fp, index=False)
        self.var_exp_path.set(fp)
        self.df_expanded = df2
        log_line(self.console, f"[OK] Expanded CSV saved → {fp}")

    def _selected_text_columns(self) -> List[str]:
        idxs = self.lb_cols.curselection()
        cols = [self.lb_cols.get(i) for i in idxs]
        self.cfg.set("text_columns", cols)
        return cols
    def _generate(self):
            task = self.task_var.get()
            input_type = self.input_type_var.get()
            text = ""

            # Determine input based on selected type
            selected = self.tree.selection()
            if input_type in ("Paragraph", "Title+Paragraph", "URL") and not selected:
                messagebox.showwarning("No record selected", "Select a record first")
                return

            if input_type == "Paragraph":
                text = self.tree.item(selected[0])['values'][5]
            elif input_type == "Title+Paragraph":
                vals = self.tree.item(selected[0])['values']
                text = f"{vals[1]} - {vals[5]}"
            elif input_type == "URL":
                text = self.tree.item(selected[0])['values'][4]
            elif input_type == "RSS Feed":
                text = self.rss_box.get("1.0", "end").strip()
                if not text:
                    messagebox.showwarning("No RSS input", "Enter RSS feed URLs")
                    return
            if input_type == "Freeform":
                text = self.input_box.get("1.0", "end").strip()
                if not text:
                    messagebox.showwarning("No input", "Enter some text in the input box")
                    return

            if not getattr(self.model, 'model', None):
                self._log("[GENERATE] Model not loaded, loading default...")
                self.model.load(self.cfg.get('model_name'))

            try:
                output = self.model.run_task(task, text)
                self._log(f"[GENERATE] Task: {task} | Input: {text[:100]}... | Output: {output[:200]}")
                # Display output in a popup
                out_win = tk.Toplevel(self)
                out_win.title(f"Output - {task}")
                st = scrolledtext.ScrolledText(out_win, width=100, height=20)
                st.pack(fill="both", expand=True)
                st.insert("end", output)
                st.config(state="disabled")
            except Exception as e:
                self._log(f"[GENERATE ERROR] {e}")


    def _log(self, msg):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.console.config(state="normal")
            self.console.insert("end", f"[{ts}] {msg}\n")
            self.console.see("end")
            self.console.config(state="disabled")
        except Exception:
            print(f"[{ts}] {msg}")

    def _build_ui(self):
        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=8, pady=6)

        ttk.Label(frm_top, text="Model: ").pack(side="left")
        self.model_entry = ttk.Entry(frm_top, width=40)
        self.model_entry.insert(0, self.cfg.get("model_name", ""))  # Use get() method
        self.model_entry.pack(side="left", padx=4)

        ttk.Label(frm_top, text="Task:").pack(side="left", padx=(12, 4))
        self.var_task = tk.StringVar(value=self.cfg.get("task_type", "auto"))
        cb_task = ttk.Combobox(frm_top, textvariable=self.var_task, width=14, values=["auto", "classification", "mlm"])
        cb_task.pack(side="left", padx=4)

        ttk.Label(frm_top, text="Device:").pack(side="left", padx=(12, 4))
        self.var_device = tk.StringVar(value=self.cfg.get("device", "auto"))
        cb_dev = ttk.Combobox(frm_top, textvariable=self.var_device, width=10, values=["auto", "cpu", "cuda"])
        cb_dev.pack(side="left", padx=4)

        self.btn_load_model = ttk.Button(frm_top, text="Load Model", command=self.on_load_model)
        self.btn_load_model.pack(side="left", padx=8)

        self.var_remember = tk.BooleanVar(value=self.cfg.get("remember", True))
        ttk.Checkbutton(frm_top, text="Remember settings", variable=self.var_remember, command=self.on_toggle_remember).pack(side="left", padx=8)

        # Notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # Tabs
        self.tab_data = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_infer = ttk.Frame(nb)
        self.tab_hyper = ttk.Frame(nb)
        self.tab_frag = ttk.Frame(nb)
        self.tab_logs = ttk.Frame(nb)

        nb.add(self.tab_data, text="Data • Crawl • Expand")
        nb.add(self.tab_train, text="Train")
        nb.add(self.tab_infer, text="Inference")
        nb.add(self.tab_hyper, text="Hyperlearning")
        nb.add(self.tab_frag, text="Fragments")
        nb.add(self.tab_logs, text="Logs")

        # Logs (console)
        self.console = scrolledtext.ScrolledText(self.tab_logs, height=20, state="disabled")
        self.console.pack(fill="both", expand=True, padx=8, pady=8)
        self.model.console = self.console  # wire console now

        # ---- Data Tab ----
        self._build_data_tab()

        # ---- Training Tab ----
        self._build_train_tab()

        # ---- Inference Tab ----
        self._build_infer_tab()

        # ---- Hyperlearning Tab ----
        self._build_hyper_tab()

        # ---- Fragments Tab ----
        self._build_frag_tab()

        # Prefill: auto save any edits
        self._wire_autosave()

    def _wire_autosave(self):
        def bind_save(var, key):
            def cb(*_):
                self.cfg.set(key, var.get())
            return cb

        self.var_model_path.trace_add("write", bind_save(self.var_model_path, "model_path"))
        self.var_task.trace_add("write", bind_save(self.var_task, "task_type"))
        self.var_device.trace_add("write", bind_save(self.var_device, "device"))

    def on_toggle_remember(self):
        self.cfg.set("remember", bool(self.var_remember.get()))

# ============================================================
# Data Tab
# ============================================================
def _build_data_tab(self):
    f = self.tab_data
    pad = dict(padx=8, pady=6)

    row = 0
    ttk.Button(f, text="Open CSV", command=self.on_open_csv).grid(row=row, column=0, **pad, sticky="w")
    self.var_csv_path = tk.StringVar(value=self.cfg.get("csv_path", ""))
    ttk.Entry(f, textvariable=self.var_csv_path, width=80).grid(row=row, column=1, **pad, sticky="w")

    row += 1
    ttk.Button(f, text="Expand ({{crawl:...}})", command=self.on_expand_csv).grid(row=row, column=0, **pad, sticky="w")
    self.var_exp_path = tk.StringVar(value=self.cfg.get("expanded_csv_path", ""))
    ttk.Entry(f, textvariable=self.var_exp_path, width=80).grid(row=row, column=1, **pad, sticky="w")

    row += 1
    ttk.Label(f, text="Detected text columns:").grid(row=row, column=0, sticky="nw", **pad)
    self.lb_cols = tk.Listbox(f, selectmode="extended", height=8, exportselection=False)
    self.lb_cols.grid(row=row, column=1, sticky="nsew", **pad)
    f.rowconfigure(row, weight=1)
    f.columnconfigure(1, weight=1)

    row += 1
    ttk.Label(f, text="Label column (optional for classification):").grid(row=row, column=0, sticky="w", **pad)
    self.var_label_col = tk.StringVar(value=self.cfg.get("label_column", ""))
    self.cb_label_col = ttk.Combobox(f, textvariable=self.var_label_col, width=30, values=[])
    self.cb_label_col.grid(row=row, column=1, sticky="w", **pad)

    # autosave
    self.var_csv_path.trace_add("write", lambda *_: self.cfg.set("csv_path", self.var_csv_path.get()))
    self.var_exp_path.trace_add("write", lambda *_: self.cfg.set("expanded_csv_path", self.var_exp_path.get()))
    self.var_label_col.trace_add("write", lambda *_: self.cfg.set("label_column", self.var_label_col.get()))

def on_open_csv(self):
    fp = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not fp:
        return
    self.var_csv_path.set(fp)
    try:
        self.df_original = pd.read_csv(fp)
        cols = self.df_original.select_dtypes(include=["object"]).columns.tolist()
        self.lb_cols.delete(0, "end")
        for c in cols:
            self.lb_cols.insert("end", c)
        self.cb_label_col["values"] = list(self.df_original.columns)
        log_line(self.console, f"[OK] Loaded CSV with shape {self.df_original.shape}")
    except Exception as e:
        messagebox.showerror("CSV Error", str(e))

def on_expand_csv(self):
    if self.df_original is None:
        messagebox.showwarning("No data", "Open a CSV first.")
        return
    df2 = expand_csv_placeholders(self.df_original.copy())
    # choose save path
    default = os.path.splitext(self.var_csv_path.get() or "data.csv")[0] + "_expanded.csv"
    fp = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(default))
    if not fp:
        return
    df2.to_csv(fp, index=False)
    self.var_exp_path.set(fp)
    self.df_expanded = df2
    log_line(self.console, f"[OK] Expanded CSV saved → {fp}")

def _selected_text_columns(self) -> List[str]:
    idxs = self.lb_cols.curselection()
    cols = [self.lb_cols.get(i) for i in idxs]
    self.cfg.set("text_columns", cols)
    return cols
def _generate(self):
        task = self.task_var.get()
        input_type = self.input_type_var.get()
        text = ""

        # Determine input based on selected type
        selected = self.tree.selection()
        if input_type in ("Paragraph", "Title+Paragraph", "URL") and not selected:
            messagebox.showwarning("No record selected", "Select a record first")
            return

        if input_type == "Paragraph":
            text = self.tree.item(selected[0])['values'][5]
        elif input_type == "Title+Paragraph":
            vals = self.tree.item(selected[0])['values']
            text = f"{vals[1]} - {vals[5]}"
        elif input_type == "URL":
            text = self.tree.item(selected[0])['values'][4]
        elif input_type == "RSS Feed":
            text = self.rss_box.get("1.0", "end").strip()
            if not text:
                messagebox.showwarning("No RSS input", "Enter RSS feed URLs")
                return
        if input_type == "Freeform":
            text = self.input_box.get("1.0", "end").strip()
            if not text:
                messagebox.showwarning("No input", "Enter some text in the input box")
                return

        if not getattr(self.model, 'model', None):
            self._log("[GENERATE] Model not loaded, loading default...")
            self.model.load(self.cfg.get('model_name'))

        try:
            output = self.model.run_task(task, text)
            self._log(f"[GENERATE] Task: {task} | Input: {text[:100]}... | Output: {output[:200]}")
            # Display output in a popup
            out_win = tk.Toplevel(self)
            out_win.title(f"Output - {task}")
            st = scrolledtext.ScrolledText(out_win, width=100, height=20)
            st.pack(fill="both", expand=True)
            st.insert("end", output)
            st.config(state="disabled")
        except Exception as e:
            self._log(f"[GENERATE ERROR] {e}")


def _log(self, msg):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        self.console.config(state="normal")
        self.console.insert("end", f"[{ts}] {msg}\n")
        self.console.see("end")
        self.console.config(state="disabled")
    except Exception:
        print(f"[{ts}] {msg}")

def _build_ui(self):
    frm_top = ttk.Frame(self)
    frm_top.pack(fill="x", padx=8, pady=6)

    ttk.Label(frm_top, text="Model: ").pack(side="left")
    self.model_entry = ttk.Entry(frm_top, width=40)
    self.model_entry.insert(0, self.cfg.get("model_name", ""))  # Use get() method
    self.model_entry.pack(side="left", padx=4)

    ttk.Label(frm_top, text="Task:").pack(side="left", padx=(12, 4))
    self.var_task = tk.StringVar(value=self.cfg.get("task_type", "auto"))
    cb_task = ttk.Combobox(frm_top, textvariable=self.var_task, width=14, values=["auto", "classification", "mlm"])
    cb_task.pack(side="left", padx=4)

    ttk.Label(frm_top, text="Device:").pack(side="left", padx=(12, 4))
    self.var_device = tk.StringVar(value=self.cfg.get("device", "auto"))
    cb_dev = ttk.Combobox(frm_top, textvariable=self.var_device, width=10, values=["auto", "cpu", "cuda"])
    cb_dev.pack(side="left", padx=4)

    self.btn_load_model = ttk.Button(frm_top, text="Load Model", command=self.on_load_model)
    self.btn_load_model.pack(side="left", padx=8)

    self.var_remember = tk.BooleanVar(value=self.cfg.get("remember", True))
    ttk.Checkbutton(frm_top, text="Remember settings", variable=self.var_remember, command=self.on_toggle_remember).pack(side="left", padx=8)

    # Notebook
    nb = ttk.Notebook(self)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    # Tabs
    self.tab_data = ttk.Frame(nb)
    self.tab_train = ttk.Frame(nb)
    self.tab_infer = ttk.Frame(nb)
    self.tab_hyper = ttk.Frame(nb)
    self.tab_frag = ttk.Frame(nb)
    self.tab_logs = ttk.Frame(nb)

    nb.add(self.tab_data, text="Data • Crawl • Expand")
    nb.add(self.tab_train, text="Train")
    nb.add(self.tab_infer, text="Inference")
    nb.add(self.tab_hyper, text="Hyperlearning")
    nb.add(self.tab_frag, text="Fragments")
    nb.add(self.tab_logs, text="Logs")

    # Logs (console)
    self.console = scrolledtext.ScrolledText(self.tab_logs, height=20, state="disabled")
    self.console.pack(fill="both", expand=True, padx=8, pady=8)
    self.model.console = self.console  # wire console now

    # ---- Data Tab ----
    self._build_data_tab()

    # ---- Training Tab ----
    self._build_train_tab()

    # ---- Inference Tab ----
    self._build_infer_tab()

    # ---- Hyperlearning Tab ----
    self._build_hyper_tab()

    # ---- Fragments Tab ----
    self._build_frag_tab()

    # Prefill: auto save any edits
    self._wire_autosave()

def _wire_autosave(self):
    def bind_save(var, key):
        def cb(*_):
            self.cfg.set(key, var.get())
        return cb

    self.var_model_path.trace_add("write", bind_save(self.var_model_path, "model_path"))
    self.var_task.trace_add("write", bind_save(self.var_task, "task_type"))
    self.var_device.trace_add("write", bind_save(self.var_device, "device"))

def on_toggle_remember(self):
    self.cfg.set("remember", bool(self.var_remember.get()))

# ============================================================
# Data Tab
# ============================================================
def _build_data_tab(self):
    f = self.tab_data
    pad = dict(padx=8, pady=6)

    row = 0
    ttk.Button(f, text="Open CSV", command=self.on_open_csv).grid(row=row, column=0, **pad, sticky="w")
    self.var_csv_path = tk.StringVar(value=self.cfg.get("csv_path", ""))
    ttk.Entry(f, textvariable=self.var_csv_path, width=80).grid(row=row, column=1, **pad, sticky="w")

    row += 1
    ttk.Button(f, text="Expand ({{crawl:...}})", command=self.on_expand_csv).grid(row=row, column=0, **pad, sticky="w")
    self.var_exp_path = tk.StringVar(value=self.cfg.get("expanded_csv_path", ""))
    ttk.Entry(f, textvariable=self.var_exp_path, width=80).grid(row=row, column=1, **pad, sticky="w")

    row += 1
    ttk.Label(f, text="Detected text columns:").grid(row=row, column=0, sticky="nw", **pad)
    self.lb_cols = tk.Listbox(f, selectmode="extended", height=8, exportselection=False)
    self.lb_cols.grid(row=row, column=1, sticky="nsew", **pad)
    f.rowconfigure(row, weight=1)
    f.columnconfigure(1, weight=1)

    row += 1
    ttk.Label(f, text="Label column (optional for classification):").grid(row=row, column=0, sticky="w", **pad)
    self.var_label_col = tk.StringVar(value=self.cfg.get("label_column", ""))
    self.cb_label_col = ttk.Combobox(f, textvariable=self.var_label_col, width=30, values=[])
    self.cb_label_col.grid(row=row, column=1, sticky="w", **pad)

    # autosave
    self.var_csv_path.trace_add("write", lambda *_: self.cfg.set("csv_path", self.var_csv_path.get()))
    self.var_exp_path.trace_add("write", lambda *_: self.cfg.set("expanded_csv_path", self.var_exp_path.get()))
    self.var_label_col.trace_add("write", lambda *_: self.cfg.set("label_column", self.var_label_col.get()))

def on_open_csv(self):
    fp = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not fp:
        return
    self.var_csv_path.set(fp)
    try:
        self.df_original = pd.read_csv(fp)
        cols = self.df_original.select_dtypes(include=["object"]).columns.tolist()
        self.lb_cols.delete(0, "end")
        for c in cols:
            self.lb_cols.insert("end", c)
        self.cb_label_col["values"] = list(self.df_original.columns)
        log_line(self.console, f"[OK] Loaded CSV with shape {self.df_original.shape}")
    except Exception as e:
        messagebox.showerror("CSV Error", str(e))

def on_expand_csv(self):
    if self.df_original is None:
        messagebox.showwarning("No data", "Open a CSV first.")
        return
    df2 = expand_csv_placeholders(self.df_original.copy())
    # choose save path
    default = os.path.splitext(self.var_csv_path.get() or "data.csv")[0] + "_expanded.csv"
    fp = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(default))
    if not fp:
        return
    df2.to_csv(fp, index=False)
    self.var_exp_path.set(fp)
    self.df_expanded = df2
    log_line(self.console, f"[OK] Expanded CSV saved → {fp}")

def _selected_text_columns(self) -> List[str]:
    idxs = self.lb_cols.curselection()
    cols = [self.lb_cols.get(i) for i in idxs]
    self.cfg.set("text_columns", cols)
    return cols
def _generate(self):
        task = self.task_var.get()
        input_type = self.input_type_var.get()
        text = ""

        # Determine input based on selected type
        selected = self.tree.selection()
        if input_type in ("Paragraph", "Title+Paragraph", "URL") and not selected:
            messagebox.showwarning("No record selected", "Select a record first")
            return

        if input_type == "Paragraph":
            text = self.tree.item(selected[0])['values'][5]
        elif input_type == "Title+Paragraph":
            vals = self.tree.item(selected[0])['values']
            text = f"{vals[1]} - {vals[5]}"
        elif input_type == "URL":
            text = self.tree.item(selected[0])['values'][4]
        elif input_type == "RSS Feed":
            text = self.rss_box.get("1.0", "end").strip()
            if not text:
                messagebox.showwarning("No RSS input", "Enter RSS feed URLs")
                return
        if input_type == "Freeform":
            text = self.input_box.get("1.0", "end").strip()
            if not text:
                messagebox.showwarning("No input", "Enter some text in the input box")
                return

        if not getattr(self.model, 'model', None):
            self._log("[GENERATE] Model not loaded, loading default...")
            self.model.load(self.cfg.get('model_name'))

        try:
            output = self.model.run_task(task, text)
            self._log(f"[GENERATE] Task: {task} | Input: {text[:100]}... | Output: {output[:200]}")
            # Display output in a popup
            out_win = tk.Toplevel(self)
            out_win.title(f"Output - {task}")
            st = scrolledtext.ScrolledText(out_win, width=100, height=20)
            st.pack(fill="both", expand=True)
            st.insert("end", output)
            st.config(state="disabled")
        except Exception as e:
            self._log(f"[GENERATE ERROR] {e}")


        def _log(self, msg):
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            try:
                self.console.config(state="normal")
                self.console.insert("end", f"[{ts}] {msg}\n")
                self.console.see("end")
                self.console.config(state="disabled")
            except Exception:
                print(f"[{ts}] {msg}")

        def _build_ui(self):
            frm_top = ttk.Frame(self)
            frm_top.pack(fill="x", padx=8, pady=6)

            ttk.Label(frm_top, text="Model: ").pack(side="left")
            self.model_entry = ttk.Entry(frm_top, width=40)
            self.model_entry.insert(0, self.cfg.get("model_name", ""))  # Use get() method
            self.model_entry.pack(side="left", padx=4)

            ttk.Label(frm_top, text="Task:").pack(side="left", padx=(12, 4))
            self.var_task = tk.StringVar(value=self.cfg.get("task_type", "auto"))
            cb_task = ttk.Combobox(frm_top, textvariable=self.var_task, width=14, values=["auto", "classification", "mlm"])
            cb_task.pack(side="left", padx=4)

            ttk.Label(frm_top, text="Device:").pack(side="left", padx=(12, 4))
            self.var_device = tk.StringVar(value=self.cfg.get("device", "auto"))
            cb_dev = ttk.Combobox(frm_top, textvariable=self.var_device, width=10, values=["auto", "cpu", "cuda"])
            cb_dev.pack(side="left", padx=4)

            self.btn_load_model = ttk.Button(frm_top, text="Load Model", command=self.on_load_model)
            self.btn_load_model.pack(side="left", padx=8)

            self.var_remember = tk.BooleanVar(value=self.cfg.get("remember", True))
            ttk.Checkbutton(frm_top, text="Remember settings", variable=self.var_remember, command=self.on_toggle_remember).pack(side="left", padx=8)

            # Notebook
            nb = ttk.Notebook(self)
            nb.pack(fill="both", expand=True, padx=8, pady=8)

            # Tabs
            self.tab_data = ttk.Frame(nb)
            self.tab_train = ttk.Frame(nb)
            self.tab_infer = ttk.Frame(nb)
            self.tab_hyper = ttk.Frame(nb)
            self.tab_frag = ttk.Frame(nb)
            self.tab_logs = ttk.Frame(nb)

            nb.add(self.tab_data, text="Data • Crawl • Expand")
            nb.add(self.tab_train, text="Train")
            nb.add(self.tab_infer, text="Inference")
            nb.add(self.tab_hyper, text="Hyperlearning")
            nb.add(self.tab_frag, text="Fragments")
            nb.add(self.tab_logs, text="Logs")

            # Logs (console)
            self.console = scrolledtext.ScrolledText(self.tab_logs, height=20, state="disabled")
            self.console.pack(fill="both", expand=True, padx=8, pady=8)
            self.model.console = self.console  # wire console now

            # ---- Data Tab ----
            self._build_data_tab()

            # ---- Training Tab ----
            self._build_train_tab()

            # ---- Inference Tab ----
            self._build_infer_tab()

            # ---- Hyperlearning Tab ----
            self._build_hyper_tab()

            # ---- Fragments Tab ----
            self._build_frag_tab()

            # Prefill: auto save any edits
            self._wire_autosave()

        def _wire_autosave(self):
            def bind_save(var, key):
                def cb(*_):
                    self.cfg.set(key, var.get())
                return cb

            self.var_model_path.trace_add("write", bind_save(self.var_model_path, "model_path"))
            self.var_task.trace_add("write", bind_save(self.var_task, "task_type"))
            self.var_device.trace_add("write", bind_save(self.var_device, "device"))

        def on_toggle_remember(self):
            self.cfg.set("remember", bool(self.var_remember.get()))

# Main entry point
if __name__ == "__main__":
    app = VedaKynaStudio()
    app.mainloop()
