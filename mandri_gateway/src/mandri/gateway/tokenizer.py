import os
from pathlib import Path


def configure() -> None:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("CUSTOM_TIKTOKEN_CACHE_DIR", str(Path(__file__).with_name("tokenizers")))
