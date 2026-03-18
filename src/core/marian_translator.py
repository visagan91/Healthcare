import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import MarianMTModel, MarianTokenizer


class MarianTranslator:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.supported_languages = self.config["supported_languages"]
        self.lang_tokens = self.config.get("lang_tokens", {})
        self.model_names = self.config["models"]

        self.tokenizers: Dict[str, MarianTokenizer] = {}
        self.models: Dict[str, MarianMTModel] = {}

        self._load_models()

    def _load_models(self):
        for key, model_name in self.model_names.items():
            self.tokenizers[key] = MarianTokenizer.from_pretrained(model_name)
            self.models[key] = MarianMTModel.from_pretrained(model_name).to(self.device)

    def _get_route(self, source_lang: str, target_lang: str) -> str | None:
        if source_lang == "en" and target_lang in ["ta", "ml", "te"]:
            return "en_dra"
        if source_lang in ["ta", "ml", "te"] and target_lang == "en":
            return "dra_en"
        if source_lang == "en" and target_lang == "hi":
            return "en_hi"
        if source_lang == "hi" and target_lang == "en":
            return "hi_en"
        return None

    def translate(self, text: str, source_lang: str, target_lang: str, max_length: int = 256) -> str:
        source_lang = (source_lang or "en").lower()
        target_lang = (target_lang or "en").lower()

        if source_lang == target_lang:
            return text

        route = self._get_route(source_lang, target_lang)
        if route is None:
            raise ValueError(f"Unsupported translation route: {source_lang} -> {target_lang}")

        tokenizer = self.tokenizers[route]
        model = self.models[route]

        input_text = text.strip()
        if route == "en_dra":
            lang_token = self.lang_tokens[target_lang]
            input_text = f"{lang_token} {input_text}"

        encoded = tokenizer(
            [input_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()