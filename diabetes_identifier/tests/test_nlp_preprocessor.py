"""Unit tests for NLPPreprocessor.

Covers Requirements 3.1, 3.2, 3.4, 11.1.
"""
import pytest

from diabetes_identifier.nlp.preprocessing import NLPPreprocessor
from diabetes_identifier.utils.config import NLPConfig


@pytest.fixture
def preprocessor():
    return NLPPreprocessor()


# ---------------------------------------------------------------------------
# Req 3.2 — PHI token replacement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi_token", ["[NAME]", "[DOB]", "[ID]", "[PHONE]", "[ADDRESS]", "[EMAIL]"])
def test_phi_token_replaced_with_phi_placeholder(preprocessor, phi_token):
    """Each PHI token should be replaced with <phi> after lowercasing (Req 3.2)."""
    note = f"Patient {phi_token} was seen today."
    result = preprocessor.preprocess([note])
    assert "<phi>" in result[0]
    assert phi_token.lower() not in result[0]


def test_all_phi_tokens_replaced_in_single_note(preprocessor):
    """All PHI tokens in one note should all be replaced (Req 3.2)."""
    note = "Name: [NAME], DOB: [DOB], ID: [ID], Phone: [PHONE], Addr: [ADDRESS], Email: [EMAIL]"
    result = preprocessor.preprocess([note])
    for token in ["[name]", "[dob]", "[id]", "[phone]", "[address]", "[email]"]:
        assert token not in result[0]
    assert result[0].count("<phi>") == 6


# ---------------------------------------------------------------------------
# Req 3.1 — Lowercasing
# ---------------------------------------------------------------------------

def test_output_is_lowercased(preprocessor):
    """All text should be lowercased (Req 3.1)."""
    note = "Patient Has HIGH Glucose Levels."
    result = preprocessor.preprocess([note])
    assert result[0] == result[0].lower()


# ---------------------------------------------------------------------------
# Req 3.4 — Empty / null note substitution and warning log
# ---------------------------------------------------------------------------

def test_empty_string_substituted_with_default(preprocessor):
    """Empty string note should be replaced with the default text (Req 3.4)."""
    result = preprocessor.preprocess([""])
    assert preprocessor.config.default_notes_text in result[0]


def test_whitespace_only_string_substituted_with_default(preprocessor):
    """Whitespace-only note should be treated as empty and substituted (Req 3.4)."""
    result = preprocessor.preprocess(["   "])
    assert preprocessor.config.default_notes_text in result[0]


def test_empty_note_logs_warning(preprocessor):
    """A warning should be logged when an empty note is encountered (Req 3.4)."""
    from unittest.mock import patch
    import diabetes_identifier.nlp.preprocessing as preproc_module

    with patch.object(preproc_module.logger, "warning") as mock_warn:
        preprocessor.preprocess([""])
    mock_warn.assert_called_once()
    call_msg = mock_warn.call_args[0][0].lower()
    assert "empty" in call_msg or "null" in call_msg


def test_none_like_empty_note_substituted(preprocessor):
    """A note that is falsy (empty string) should use the configured default (Req 3.4)."""
    config = NLPConfig(default_notes_text="custom default text")
    proc = NLPPreprocessor(config=config)
    result = proc.preprocess([""])
    assert "custom default text" in result[0]


# ---------------------------------------------------------------------------
# Req 3.3 — preprocess() returns a list of strings
# ---------------------------------------------------------------------------

def test_preprocess_returns_list(preprocessor):
    """preprocess() should return a list (Req 3.3)."""
    result = preprocessor.preprocess(["Patient has high glucose."])
    assert isinstance(result, list)


def test_preprocess_returns_list_of_strings(preprocessor):
    """Each element in the returned list should be a string (Req 3.3)."""
    notes = ["Patient has high glucose.", "No symptoms noted."]
    result = preprocessor.preprocess(notes)
    assert all(isinstance(item, str) for item in result)


def test_preprocess_output_length_matches_input(preprocessor):
    """Output list length should equal input list length (Req 3.3)."""
    notes = ["Note one.", "Note two.", "Note three."]
    result = preprocessor.preprocess(notes)
    assert len(result) == len(notes)


def test_preprocess_single_note_returns_single_element_list(preprocessor):
    """A single input note should produce a single-element list (Req 3.3)."""
    result = preprocessor.preprocess(["Single note."])
    assert len(result) == 1
    assert isinstance(result[0], str)
