"""AI Shorts can use OpenRouter without Google, with usable scene plans and safe errors."""
import json
from types import SimpleNamespace
import sys

import httpx
import pytest

import saasshorts
import ugc_openrouter


def script():
    return {
        "title": "Camera roll", "duration_seconds": 23, "hook_text": "Your camera roll",
        "full_narration": "No photos. Try ProfileBump. Link in bio.",
        "actor_description": "An adult man wearing a casual tee",
        "segments": [
            {"type": kind, "start": start, "end": end, "visual": visual,
             "narration": "Example narration", "broll_prompt": "Phone camera roll" if visual == "broll" else None}
            for kind, start, end, visual in [
                ("hook", 0, 5, "actor_talking"), ("problem", 5, 9, "broll"),
                ("solution", 9, 16, "actor_talking"), ("demo", 16, 21, "broll"),
                ("cta", 21, 23, "actor_talking"),
            ]
        ],
    }


def serve(monkeypatch, handler):
    client_type = httpx.Client
    monkeypatch.setattr(ugc_openrouter.httpx, "Client",
                        lambda **kw: client_type(transport=httpx.MockTransport(handler), **kw))


def completion(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


def test_switch_models_and_generate_without_google(monkeypatch):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append((body["model"], request.headers["authorization"]))
        return completion("```json\n" + json.dumps([script()]) + "\n```")
    serve(monkeypatch, handler)
    # If the OpenRouter path accidentally imports Google, this workflow fails.
    monkeypatch.setitem(sys.modules, "google.genai", None)
    for model in ("vendor/model-a", "vendor/model-b"):
        result = saasshorts.generate_scripts({"product_name": "ProfileBump"}, None, 1,
            llm_provider="openrouter", openrouter_key="test-secret", llm_model=model)
        assert result[0]["title"] == "Camera roll"
        assert result[0]["segments"][3]["visual"] == "broll"
    assert seen == [("vendor/model-a", "Bearer test-secret"), ("vendor/model-b", "Bearer test-secret")]


def test_website_analysis_uses_openrouter(monkeypatch):
    serve(monkeypatch, lambda request: completion('{"product_name":"ProfileBump","pain_points":[]}'))
    result = saasshorts.analyze_saas({"url": "https://example.test", "title": "Photo tool",
        "meta_description": "", "headings": [], "main_content": "Six profile photos",
        "additional_pages": ""}, None, llm_provider="openrouter", openrouter_key="key", llm_model="vendor/model")
    assert result["product_name"] == "ProfileBump"
    assert "_web_sources" not in result


def test_gemini_remains_available(monkeypatch):
    import types
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.types = SimpleNamespace(GenerateContentConfig=lambda **kw: kw)
    genai.Client = lambda **kw: SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kw: SimpleNamespace(text=json.dumps([script()]))))
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    assert saasshorts.generate_scripts({}, "gemini-key", 1)[0]["title"] == "Camera roll"


@pytest.mark.parametrize("status, message", [(401, "API key"), (402, "credits"), (404, "model not found"), (429, "rate limit"), (500, "HTTP 500")])
def test_provider_errors_are_actionable_and_do_not_echo_secrets(monkeypatch, status, message):
    serve(monkeypatch, lambda request: httpx.Response(status, json={"error": {"message": "test-secret"}}))
    with pytest.raises(ValueError, match=message) as error:
        ugc_openrouter.complete("private product input", "test-secret", "vendor/model")
    assert "test-secret" not in str(error.value)
    assert "private product input" not in str(error.value)


@pytest.mark.parametrize("body", [
    {"choices": []}, {"choices": [{"message": {"content": None}}]},
    {"error": {"message": "provider failed"}},
    {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
])
def test_unusable_provider_responses_fail_before_video_generation(monkeypatch, body):
    serve(monkeypatch, lambda request: httpx.Response(200, json=body))
    with pytest.raises(ValueError):
        ugc_openrouter.complete("prompt", "key", "vendor/model")


def test_timeout_is_readable(monkeypatch):
    def timeout(request):
        raise httpx.ReadTimeout("private data", request=request)
    serve(monkeypatch, timeout)
    with pytest.raises(ValueError, match="timed out"):
        ugc_openrouter.complete("prompt", "key", "vendor/model")


@pytest.mark.parametrize("response", ["not JSON", "[]", '[{"title":"Missing scenes"}]'])
def test_invalid_scripts_are_rejected(monkeypatch, response):
    serve(monkeypatch, lambda request: completion(response))
    with pytest.raises(ValueError):
        saasshorts.generate_scripts({}, None, 1, llm_provider="openrouter", openrouter_key="key", llm_model="vendor/model")


@pytest.mark.parametrize("change", ["long", "gap", "broll", "visual"])
def test_invalid_scene_plans_are_rejected(change):
    value = script()
    if change == "long": value["segments"][-1]["end"] = 31
    if change == "gap": value["segments"][1]["start"] = 6
    if change == "broll": value["segments"][1]["broll_prompt"] = None
    if change == "visual": value["segments"][1]["visual"] = "actor_talking"
    with pytest.raises(ValueError):
        ugc_openrouter.validate_scripts([value], 1)
