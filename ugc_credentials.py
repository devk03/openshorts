"""Self-hosted AI Shorts credentials stay server-side; cloud remains BYOK."""
import os


def resolve_key(name, supplied=None, *, managed=False):
    if name not in {"FAL_API_KEY", "ELEVENLABS_API_KEY"}:
        raise ValueError("Unsupported UGC credential")
    return (supplied or ("" if managed else os.environ.get(name, ""))).strip()


def public_config(*, managed=False):
    if managed:
        return {}
    return {
        "fal": bool(resolve_key("FAL_API_KEY")),
        "elevenlabs": bool(resolve_key("ELEVENLABS_API_KEY")),
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
        "model": os.environ.get("OPENROUTER_MODEL", "").strip(),
    }
