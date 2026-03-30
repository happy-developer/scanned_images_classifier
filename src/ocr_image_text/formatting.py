from __future__ import annotations

import re

INSTRUCTION = (
    "Transcribe all visible text from this invoice image exactly as seen. "
    "Preserve line breaks when possible and do not summarize."
)


def normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    compact = "\n".join(line for line in lines if line)
    return compact.strip()


def record_to_messages(record: "OCRRecord", image_obj) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTION},
                {"type": "image", "image": image_obj},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": normalize_text(record.ocr_text)}],
        },
    ]
