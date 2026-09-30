import json

from lipreading.vsr import llm


def test_options_parses_and_limits():
    ask = lambda msgs: json.dumps({"options": ["uno", " dos ", "", "tres", "cuatro"]})  # noqa: E731
    assert llm.options("lectura", ask=ask) == ["uno", "dos", "tres"]


def test_context_is_sent_and_n_is_in_the_prompt():
    seen = {}

    def ask(msgs):
        seen["msgs"] = msgs
        return json.dumps({"options": ["a"] * 8})

    assert len(llm.options("lectura", "¿Tienes frío?", n=8, ask=ask)) == 8
    assert "¿Tienes frío?" in seen["msgs"][1]["content"]
    assert "propose the 8 sentences" in seen["msgs"][0]["content"]


def test_falls_back_to_the_reading_when_the_llm_fails():
    def ask(msgs):
        raise RuntimeError("network down")

    assert llm.options("lectura cruda", ask=ask) == ["lectura cruda"]
