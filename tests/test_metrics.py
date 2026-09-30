from lipreading.metrics import cer, edit_distance, normalize, summarize, wer


def test_normalize_strips_accents_and_punctuation():
    assert normalize("¿Cómo amaneciste HOY?") == "como amaneciste hoy"
    assert normalize("Añoranza, sí.") == "anoranza si"


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1


def test_wer():
    assert wer("un café por favor", "un café por favor") == 0
    assert wer("un café por favor", "un té por favor") == 0.25
    assert wer("", "") == 0 and wer("", "hola") == 1


def test_cer_ignores_spaces():
    assert cer("hola mundo", "holamundo") == 0


def test_summarize():
    s = summarize([0, 0.2, 0.5, 1.0])
    assert s["n"] == 4 and s["exact"] == 0.25 and s["understandable"] == 0.5
    assert summarize([])["wer"] is None
