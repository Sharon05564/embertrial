"""
Static feature extraction for Win32 PE files.

Reproduces the same feature set the training pipeline builds from the
EMBER2024 dataset's pre-extracted `general` / `strings` / `imports` JSON
blobs (see notebooks/detection-model.ipynb), but computed directly from raw
file bytes so the live classifier demo can score an arbitrary uploaded
.exe rather than a row that already has those fields precomputed.

Column order matches training exactly:
    file_size, file_entropy, is_pe,
    numstrings, avg_string_length, num_printables, string_entropy,
    num_imported_dlls, num_imported_functions, avg_functions_per_dll,
    has_KERNEL32.dll, has_ADVAPI32.dll, has_USER32.dll,
    has_WS2_32.dll, has_SHELL32.dll, has_OLE32.dll
"""

import math
import re
from collections import Counter

import pandas as pd
import pefile

COMMON_DLLS = [
    "KERNEL32.dll",
    "ADVAPI32.dll",
    "USER32.dll",
    "WS2_32.dll",
    "SHELL32.dll",
    "OLE32.dll",
]

FEATURE_COLUMNS = (
    ["file_size", "file_entropy", "is_pe"]
    + ["numstrings", "avg_string_length", "num_printables", "string_entropy"]
    + ["num_imported_dlls", "num_imported_functions", "avg_functions_per_dll"]
    + [f"has_{dll}" for dll in COMMON_DLLS]
)

# EMBER-style ASCII string extraction: runs of >=5 printable characters
_STRING_RE = re.compile(rb"[\x20-\x7e]{5,}")


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def extract_general_features(data: bytes, is_pe: bool) -> dict:
    return {
        "file_size": len(data),
        "file_entropy": shannon_entropy(data),
        "is_pe": int(is_pe),
    }


def extract_string_features(data: bytes) -> dict:
    strings = _STRING_RE.findall(data)
    numstrings = len(strings)
    total_len = sum(len(s) for s in strings)
    avg_length = (total_len / numstrings) if numstrings else 0.0

    joined = b"".join(strings)
    return {
        "numstrings": numstrings,
        "avg_string_length": avg_length,
        "num_printables": total_len,
        "string_entropy": shannon_entropy(joined),
    }


def extract_import_features(data: bytes) -> dict:
    """
    Parse the PE import table into {dll_name: [function_names]} and reduce
    it to the same numeric features used at training time. Returns all
    zeros/flags-off if the file isn't a parseable PE or has no imports.
    """
    imports: dict[str, list] = {}
    try:
        pe = pefile.PE(data=data, fast_load=True)
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
        )
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            dll_name = entry.dll.decode("ascii", errors="replace") if entry.dll else "UNKNOWN"
            funcs = [
                (imp.name.decode("ascii", errors="replace") if imp.name else f"ordinal_{imp.ordinal}")
                for imp in entry.imports
            ]
            imports[dll_name] = funcs
        pe.close()
    except pefile.PEFormatError:
        pass

    features = {"num_imported_dlls": len(imports)}
    features["num_imported_functions"] = sum(len(v) for v in imports.values())
    features["avg_functions_per_dll"] = (
        features["num_imported_functions"] / len(imports) if imports else 0.0
    )

    lower_names = {name.lower() for name in imports}
    for dll in COMMON_DLLS:
        features[f"has_{dll}"] = int(dll.lower() in lower_names)

    return features


def is_pe_file(data: bytes) -> bool:
    if len(data) < 0x40 or data[:2] != b"MZ":
        return False
    try:
        pe_offset = int.from_bytes(data[0x3C:0x40], "little")
        return data[pe_offset:pe_offset + 4] == b"PE\x00\x00"
    except (IndexError, ValueError):
        return False


def extract_features(data: bytes) -> dict:
    """Extract the full flat feature dict for one file's raw bytes."""
    pe_flag = is_pe_file(data)
    features = {}
    features.update(extract_general_features(data, pe_flag))
    features.update(extract_string_features(data))
    features.update(extract_import_features(data) if pe_flag else _empty_import_features())
    return features


def _empty_import_features() -> dict:
    features = {"num_imported_dlls": 0, "num_imported_functions": 0, "avg_functions_per_dll": 0.0}
    for dll in COMMON_DLLS:
        features[f"has_{dll}"] = 0
    return features


def features_to_dataframe(features: dict) -> pd.DataFrame:
    """Order the feature dict into a single-row DataFrame matching training column order."""
    return pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
