from lipreading.visemes import rerank_by_visemes, to_visemes, viseme_similarity


def test_lookalike_consonants_share_a_class():
    # p/b/m look the same on the lips
    assert to_visemes("pan") == to_visemes("man") == to_visemes("ban")
    assert to_visemes("vaca") == to_visemes("faca")


def test_digraphs_and_silent_h():
    assert to_visemes("chile")[0] == "Y"
    assert to_visemes("hola") == to_visemes("ola")


def test_similarity_bounds():
    assert viseme_similarity("mesa", "besa") == 1.0
    assert 0 <= viseme_similarity("mesa", "gato") < 1


def test_rerank_prefers_candidates_that_fit_the_mouth_shapes():
    reading = "con los huevos con la cama y un café"
    candidates = ["Sí, llama a tu hermana", "Unos huevos con arepa y un café"]
    assert rerank_by_visemes(reading, candidates)[0] == "Unos huevos con arepa y un café"
