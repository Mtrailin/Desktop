# vedakyna_studio.py
# Standard library imports
import json
import logging
import os
import random
import time
import tkinter as tk
import uuid
from datetime import datetime
from os import PathLike
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Any, Callable, Dict, List, Optional, Union, Type, TypeVar

# Third-party imports
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
import feedparser
# ML/DL framework imports
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    pipeline
)

APP_TITLE = "VEDA-KYNA™ Studio — Crawl • Expand • Train • Inference • Hyperlearn"
CFG_FILE = "config.json"

class Config:
    def __init__(self):
        self.data: Dict[str, Union[bool, List[str], str]] = {
            'remember': True,
            'urls': []
        }

    def load(self):
        if os.path.exists(CFG_FILE):
            try:
                with open(CFG_FILE, "r", encoding="utf-8") as f:
                    self.data.update(json.load(f))
            except Exception as e:
                print(f"Error loading config: {e}")

    def save(self):
        with open(CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

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
            f.ingest(base)

    def multiplex(self, query: str) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for f in self.fragments:
            reflection = f.reflect(query)
            meta = f.meta(reflection)
            out.append({
                "fragment": f.name,
                "reflection": reflection,
                "meta": meta
            })
        return out

class Throttle:
    def __init__(self, min_delay: float = 3.0, max_delay: float = 9.0):
        self.min_delay: float = min_delay
        self.max_delay: float = max_delay

    def wait(self):
        time.sleep(random.uniform(self.min_delay, self.max_delay))

class Crawler:
    def __init__(self, throttle: Throttle, log_fn: Callable[[str], Any] = print):
        self.throttle: Throttle = throttle
        self.log: Callable[[str], Any] = log_fn
        self.session: requests.Session = requests.Session()
        self.session.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'
        }
        self.user_agents: List[str] = [
            'Mozilla/5.0 (compatible; ResearchBot/1.0)',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        ]

    def fetch(self, url: str) -> Optional[str]:
        headers = {"User-Agent": random.choice(self.user_agents)}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.log(f"[ERROR] Failed to fetch {url}: {e}")
            return None

    def crawl_article(self, url: str) -> Optional[Dict[str, Optional[Union[str, List[str]]]]]:
        html = self.fetch(url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.get_text(strip=True) if soup.title else None
        author = None
        date = None

        # Extract metadata
        for meta in soup.find_all('meta'):
            if meta.get('name') in ['author', 'article:author']:
                author = meta.get('content')
            elif meta.get('property') in ['article:published_time', 'date']:
                date = meta.get('content')

        # Extract paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')
                     if len(p.get_text(strip=True).split()) > 8]

        self.throttle.wait()
        return {
            "url": url,
            "title": title,
            "author": author,
            "date": date,
            "paragraphs": paragraphs
        }

class FlanT5Model:
    def __init__(self,
                 model: Optional[str] = None,
                 device: Optional[str] = None,
                 log_fn: Callable[[str], Any] = print):
        self.model_name: Optional[str] = model
        if device is None or device == "auto":
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.log: Callable[[str], Any] = log_fn
        self.model_type: str = "seq2seq"

    def load(self, path_or_name: Optional[Union[str, PathLike[str]]] = None) -> bool:
        name = path_or_name or self.model_name
        self.log(f"[MODEL] Loading {name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
            if self.device == "cuda":
                self.model = self.model.to("cuda")
            self.log("[MODEL] Successfully loaded model and tokenizer.")
            return True
        except Exception as e:
            self.log(f"[ERROR] Failed to load model: {e}")
            return False

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}

    def run_task(self, task: str, text: str, max_length: int = 128) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        prompt = f"{task}: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt",
                              truncation=True, padding=True)
        inputs = self._move_inputs_to_device(inputs)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def prepare_dataset(
        self,
        records: Union[Dataset, List[Dict[str, Any]]],
        max_input_length: int = 256,
        max_output_length: int = 128
    ) -> Dataset:
        """Prepare a dataset for training"""
        if not isinstance(records, Dataset):
            records = Dataset.from_list(records)

        def tokenize_batch(examples: Dict[str, Any]) -> Dict[str, Any]:
            inputs = examples["text"]
            targets = examples.get("labels", [""] * len(inputs))

            model_inputs = self.tokenizer(
                inputs,
                max_length=max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=max_output_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized = records.map(
            tokenize_batch,
            batched=True,
            remove_columns=records.column_names
        )
        return tokenized

    def train(
        self,
        dataset: Dataset,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        output_dir: Union[str, PathLike[str]] = "./model_output"
    ) -> None:
        """Train the model on a dataset"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=1000,
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=torch.cuda.is_available()
        )

        # Initialize data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # Train the model
        self.log("[TRAIN] Starting training...")
        trainer.train()

        # Save the final model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.log(f"[TRAIN] Model saved to {output_dir}")

class ConfigManager:
    def __init__(self, path=CFG_FILE):
        self.path = path
        self.data = {
            "model_name": "",
            "model_path": "veda-kyna",
            "device": "cuda",
            "csv_path": "",
            "expanded_csv_path": "",
            "text_columns": [],
            "label_column": "",
            "task_type": "auto",
            "epochs": 3,
            "batch_size": 64,
            "max_length": 1024,
            "learning_rate": 0.0003,
            "output_dir": "./veda_output",
            "output_type": "label",
            "remember": True,
            "num_workers": 13,
            "pin_memory": True
        }
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data.update(json.load(f))
            except Exception as e:
                print(f"Error loading config: {e}")

    def save(self):
        if not self.data.get("remember", True):
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key, default=None):
        """Get config value with optional default"""
        return self.data.get(key, default)

    def set(self, key, value):
        """Set config value and save"""
        self.data[key] = value
        self.save()

class ModelHandler:
    def __init__(self, cfg: ConfigManager, console: Optional[tk.Text]):
        self.cfg: ConfigManager = cfg
        self.console: Optional[tk.Text] = console
        self.model: Optional[Union[AutoModelForSequenceClassification, AutoModelForSeq2SeqLM]] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.pipe: Optional[Any] = None
        self.model_type: Optional[str] = None
        self.label_encoder: Optional[LabelEncoder] = None

    def load_model(self, path_or_repo: str, task_type: str = "auto"):
        if not path_or_repo:
            return False

        self.log(f"[INFO] Loading model from {path_or_repo}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path_or_repo)
            if task_type == "classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(path_or_repo)
                self.model_type = "classification"
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(path_or_repo)
                self.model_type = "seq2seq"

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            self.log("[OK] Model loaded successfully")
            return True
        except Exception as e:
            self.log(f"[ERROR] Failed to load model: {e}")
            return False

    def log(self, msg: str):
        if self.console:
            self.console.config(state="normal")
            self.console.insert("end", msg + "\n")
            self.console.see("end")
            self.console.config(state="disabled")
        else:
            print(msg)

    def prepare_dataset(self, df: pd.DataFrame, text_cols: List[str], label_col: Optional[str] = None):
        if not text_cols:
            raise ValueError("No text columns selected")

        df = df.copy()
        df["text_combined"] = df[text_cols].astype(str).agg(" ".join, axis=1)

        if label_col and label_col in df.columns:
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(df[label_col].astype(str))
            df["labels"] = labels

        return Dataset.from_pandas(df)

class VedaKynaStudio(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1200x800")

        self.cfg = ConfigManager()
        self.model = ModelHandler(self.cfg, None)  # console added later

        self.df_original = None
        self.df_expanded = None

        self._build_ui()

    def _build_ui(self):
        # Top frame with model controls
        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=8, pady=6)

        ttk.Label(frm_top, text="Model:").pack(side="left")
        self.var_model = tk.StringVar(value=self.cfg.get("model_name", ""))
        ttk.Entry(frm_top, textvariable=self.var_model, width=40).pack(side="left", padx=4)

        ttk.Label(frm_top, text="Task:").pack(side="left", padx=(12,4))
        self.var_task = tk.StringVar(value=self.cfg.get("task_type", "auto"))
        ttk.Combobox(frm_top, textvariable=self.var_task, values=["auto", "classification", "seq2seq"],
                    width=14).pack(side="left", padx=4)

        ttk.Label(frm_top, text="Device:").pack(side="left", padx=(12,4))
        self.var_device = tk.StringVar(value=self.cfg.get("device", "auto"))
        ttk.Combobox(frm_top, textvariable=self.var_device, values=["auto", "cpu", "cuda"],
                    width=10).pack(side="left", padx=4)

        ttk.Button(frm_top, text="Load Model", command=self._load_model).pack(side="left", padx=8)

        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=6)

        # Create tabs
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_infer = ttk.Frame(self.notebook)
        self.tab_logs = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_data, text="Data")
        self.notebook.add(self.tab_train, text="Train")
        self.notebook.add(self.tab_infer, text="Inference")
        self.notebook.add(self.tab_logs, text="Logs")

        # Build tab contents
        self._build_data_tab()
        self._build_train_tab()
        self._build_infer_tab()
        self._build_logs_tab()

    def _build_data_tab(self):
        f = self.tab_data
        pad = dict(padx=8, pady=6)

        # CSV controls
        ttk.Button(f, text="Open CSV", command=self._open_csv).grid(row=0, column=0, **pad)
        self.var_csv_path = tk.StringVar(value=self.cfg.get("csv_path", ""))
        ttk.Entry(f, textvariable=self.var_csv_path, width=60).grid(row=0, column=1, columnspan=2, **pad)

        # Text columns
        ttk.Label(f, text="Text Columns:").grid(row=1, column=0, **pad)
        self.list_cols = tk.Listbox(f, selectmode="extended", exportselection=False, height=6)
        self.list_cols.grid(row=1, column=1, columnspan=2, sticky="nsew", **pad)

        # Label column for classification
        ttk.Label(f, text="Label Column:").grid(row=2, column=0, **pad)
        self.var_label_col = tk.StringVar(value=self.cfg.get("label_column", ""))
        self.combo_label = ttk.Combobox(f, textvariable=self.var_label_col, width=30)
        self.combo_label.grid(row=2, column=1, **pad)

    def _build_train_tab(self):
        f = self.tab_train
        pad = dict(padx=8, pady=6)

        # Training parameters
        params = [
            ("Epochs:", "epochs", 3),
            ("Batch Size:", "batch_size", 32),
            ("Learning Rate:", "learning_rate", 0.0003),
            ("Max Length:", "max_length", 512)
        ]

        for i, (label, key, default) in enumerate(params):
            ttk.Label(f, text=label).grid(row=i, column=0, **pad)
            var = tk.StringVar(value=str(self.cfg.get(key, default)))
            ttk.Entry(f, textvariable=var, width=10).grid(row=i, column=1, **pad)
            setattr(self, f"var_{key}", var)

        ttk.Button(f, text="Start Training", command=self._start_training).grid(row=len(params), column=0, columnspan=2, **pad)

    def _build_infer_tab(self):
        f = self.tab_infer
        pad = dict(padx=8, pady=6)

        # Input text
        ttk.Label(f, text="Input:").grid(row=0, column=0, **pad)
        self.text_input = tk.Text(f, height=5, width=60)
        self.text_input.grid(row=1, column=0, columnspan=2, **pad)

        # Output
        ttk.Label(f, text="Output:").grid(row=2, column=0, **pad)
        self.text_output = tk.Text(f, height=5, width=60, state="disabled")
        self.text_output.grid(row=3, column=0, columnspan=2, **pad)

        ttk.Button(f, text="Run Inference", command=self._run_inference).grid(row=4, column=0, columnspan=2, **pad)

    def _build_logs_tab(self):
        self.console = scrolledtext.ScrolledText(self.tab_logs, height=20)
        self.console.pack(fill="both", expand=True, padx=8, pady=6)
        self.console.config(state="disabled")
        self.model.console = self.console

    def _load_model(self):
        model_path = self.var_model.get().strip()
        if not model_path:
            messagebox.showerror("Error", "Please enter a model path or name")
            return

        self.model.load_model(model_path, self.var_task.get())

    def _open_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not fp:
            return

        try:
            df = pd.read_csv(fp)
            self.df_original = df
            self.var_csv_path.set(fp)

            # Update column lists
            self.list_cols.delete(0, tk.END)
            self.combo_label.configure(values=list(df.columns))

            for col in df.columns:
                self.list_cols.insert(tk.END, col)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def _start_training(self):
        if self.df_original is None:
            messagebox.showerror("Error", "Please load a CSV file first")
            return

        text_cols = [self.list_cols.get(i) for i in self.list_cols.curselection()]
        if not text_cols:
            messagebox.showerror("Error", "Please select text columns")
            return

        try:
            dataset = self.model.prepare_dataset(
                self.df_original,
                text_cols,
                self.var_label_col.get()
            )

            # Get training parameters
            params = {
                "epochs": int(self.var_epochs.get()),
                "batch_size": int(self.var_batch_size.get()),
                "learning_rate": float(self.var_learning_rate.get()),
                "max_length": int(self.var_max_length.get())
            }

            # Start training in a separate thread to avoid blocking GUI
            import threading
            thread = threading.Thread(
                target=self._train_thread,
                args=(dataset, params)
            )
            thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")

    def _train_thread(self, dataset, params):
        try:
            self.model.train(dataset, **params)
            self.console.config(state="normal")
            self.console.insert("end", "Training completed successfully!\n")
            self.console.see("end")
            self.console.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")

    def _run_inference(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showerror("Error", "Please enter some text")
            return

        try:
            output = self.model.run_task("inference", text)

            self.text_output.config(state="normal")
            self.text_output.delete("1.0", "end")
            self.text_output.insert("1.0", output)
            self.text_output.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {e}")

# Main entry point
if __name__ == "__main__":
    app = VedaKynaStudio()
    app.mainloop()
