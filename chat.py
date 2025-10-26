#!/usr/bin/env python3
import re
import argparse
import sys

# ====================
# FREE SPEECH™ Cipher Package
# ====================

# Large dictionary: code word → English meaning
CODE_MAP = {
    # Pronouns
    "apple": "I",
    "bana": "you",
    "cino": "he",
    "dela": "she",
    "elro": "we",
    "fano": "they",

    # Verbs
    "lala": "want",
    "labsa": "desire",
    "pira": "go",
    "nira": "say",
    "mira": "do",
    "sira": "make",
    "tira": "know",
    "uira": "see",
    "vira": "think",
    "wira": "take",
    "xira": "come",
    "yira": "want",
    "zelo": "have",
    "xeno": "do",
    "yano": "does",
    "qira": "will",
    "veta": "can",

    # Conjunctions & Prepositions
    "palo": "and",
    "rano": "or",
    "sila": "but",
    "talo": "if",
    "varo": "not",
    "kelo": "to",
    "mino": "with",

    # Nouns
    "oloa": "freedom",
    "terno": "knowledge",
    "krina": "truth",
    "shapa": "intent",
    "lermo": "value",
    "gropa": "mind",
    "netsa": "trust",
    "dolan": "choice",
    "bresa": "path",
    "mokra": "question",
    "rilto": "answer",

    # Adjectives
    "sento": "is",
    "hano": "are",

    # Miscellaneous
    "veta": "can",
    "qira": "will",

    # Add more as needed
}

# Reverse dictionary: English meaning → code word
REVERSE_CODE_MAP = {v: k for k, v in CODE_MAP.items()}

# Regex to split words and punctuation
TOKEN_RE = re.compile(r"(\b\w+\b|[^\w\s])")

def preserve_case(original, replacement):
    if original.isupper():
        return replacement.upper()
    elif original[0].isupper():
        return replacement.capitalize()
    else:
        return replacement

def encode(text):
    """Encode English text into coded words."""
    tokens = TOKEN_RE.findall(text)
    encoded_tokens = []
    for token in tokens:
        lw = token.lower()
        if lw in REVERSE_CODE_MAP:
            code_word = REVERSE_CODE_MAP[lw]
            encoded_tokens.append(preserve_case(token, code_word))
        else:
            encoded_tokens.append(token)
    # Rebuild string with spaces only between words, not punctuation
    output = ""
    for i, token in enumerate(encoded_tokens):
        output += token
        # Add space if next token is a word and current is a word
        if i + 1 < len(encoded_tokens):
            if re.match(r"\w", token) and re.match(r"\w", encoded_tokens[i + 1]):
                output += " "
    return output

def decode(text):
    """Decode coded words into English text."""
    tokens = TOKEN_RE.findall(text)
    decoded_tokens = []
    for token in tokens:
        lw = token.lower()
        if lw in CODE_MAP:
            decoded_word = CODE_MAP[lw]
            decoded_tokens.append(preserve_case(token, decoded_word))
        else:
            decoded_tokens.append(token)
    # Rebuild string with spaces only between words, not punctuation
    output = ""
    for i, token in enumerate(decoded_tokens):
        output += token
        # Add space if next token is a word and current is a word
        if i + 1 < len(decoded_tokens):
            if re.match(r"\w", token) and re.match(r"\w", decoded_tokens[i + 1]):
                output += " "
    return output

def main():
    parser = argparse.ArgumentParser(description="FREE SPEECH™ Coded Communication CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-e", "--encode", action="store_true", help="Encode input text to coded words")
    group.add_argument("-d", "--decode", action="store_true", help="Decode coded words to English")
    parser.add_argument("text", nargs="+", help="Text to encode/decode")
    args = parser.parse_args()

    input_text = " ".join(args.text)
    if args.encode:
        result = encode(input_text)
    else:
        result = decode(input_text)

    print(result)

def run_gui():
    import tkinter as tk
    from tkinter import ttk, scrolledtext

    def process():
        text = input_text.get("1.0", tk.END).strip()
        if mode.get() == "encode":
            result = encode(text)
        else:
            result = decode(text)
        output_text.config(state="normal")
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, result)
        output_text.config(state="disabled")

    root = tk.Tk()
    root.title("FREE SPEECH™ Cipher GUI")

    frm = ttk.Frame(root, padding=10)
    frm.grid()

    ttk.Label(frm, text="Input:").grid(column=0, row=0, sticky="w")
    input_text = scrolledtext.ScrolledText(frm, width=50, height=5)
    input_text.grid(column=0, row=1, columnspan=2, pady=2)

    mode = tk.StringVar(value="encode")
    ttk.Radiobutton(frm, text="Encode", variable=mode, value="encode").grid(column=0, row=2, sticky="w")
    ttk.Radiobutton(frm, text="Decode", variable=mode, value="decode").grid(column=1, row=2, sticky="w")

    ttk.Button(frm, text="Process", command=process).grid(column=0, row=3, pady=5, sticky="w")

    ttk.Label(frm, text="Output:").grid(column=0, row=4, sticky="w")
    output_text = scrolledtext.ScrolledText(frm, width=50, height=5, state="disabled")
    output_text.grid(column=0, row=5, columnspan=2, pady=2)

    root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        run_gui()
