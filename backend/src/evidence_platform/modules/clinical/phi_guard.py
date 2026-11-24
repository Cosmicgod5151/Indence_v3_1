"""Local PHI (protected health information) guard — runs BEFORE any cloud call.

Indence Case Mode accepts only synthetic or de-identified case text (NFR-13).
This guard scans the text on the user's machine for direct identifiers and
blocks the request when any are found, returning the exact spans so the
clinician can redact them. Patterns follow the HIPAA Safe Harbor identifier
list and Indian identifiers (Aadhaar, ABHA, UHID/MRN, +91 phones, PIN codes).

It is deliberately conservative: a false positive costs one redaction; a false
negative could send personal data to a third-party model provider.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

_RULES: list[tuple[str, re.Pattern]] = [
    ("Patient name field", re.compile(r"\b(?:patient(?:'s)?\s+name|name\s+of\s+(?:the\s+)?patient|pt\.?\s*name|\bname)\s*[:\-]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z.]+){0,3}")),
    ("Name with honorific", re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Shri|Smt|Kumari|Master)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")),
    ("Email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("Phone number", re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b|\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    ("Aadhaar / national ID", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")),
    ("ABHA health ID", re.compile(r"\b\d{2}-\d{4}-\d{4}-\d{4}\b")),
    ("Medical record / hospital number", re.compile(r"\b(?:MRN|UHID|IP\s*No|OP\s*No|CR\s*No|Hospital\s*(?:No|ID)|Reg(?:istration)?\.?\s*No|Record\s*No)\.?\s*[:#\-]?\s*[A-Z0-9/-]{4,}\b", re.I)),
    ("Date of birth", re.compile(r"\b(?:DOB|D\.O\.B\.?|date\s+of\s+birth|born\s+on)\s*[:\-]?\s*\d{1,2}[/.\- ](?:\d{1,2}|[A-Za-z]{3,9})[/.\- ]\d{2,4}\b", re.I)),
    ("Street address / PIN code", re.compile(r"\b\d{1,4},?\s+[A-Z][a-z]+\s+(?:Road|Rd|Street|St|Nagar|Colony|Lane|Marg|Avenue|Sector)\b|\b(?:PIN|Pincode)\s*[:\-]?\s*\d{6}\b", re.I)),
]


@dataclass
class PHIFinding:
    kind: str
    text: str
    start: int
    end: int


def scan_phi(text: str) -> dict:
    """Return {"blocked": bool, "findings": [...]} for the given free text."""
    findings: list[PHIFinding] = []
    for kind, rx in _RULES:
        for m in rx.finditer(text or ""):
            if any(f.start <= m.start() < f.end for f in findings):
                continue
            findings.append(PHIFinding(kind, m.group(0), m.start(), m.end()))
    findings.sort(key=lambda f: f.start)
    return {
        "blocked": bool(findings),
        "findings": [asdict(f) for f in findings],
        "message": (
            "Possible identifiers found. Case Mode only accepts synthetic or de-identified text — "
            "remove the highlighted items before any text is sent to the model provider."
            if findings else "No direct identifiers detected (local scan; nothing has left this device)."
        ),
    }
