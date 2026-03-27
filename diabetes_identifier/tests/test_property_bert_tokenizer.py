"""Property-based tests for BERT tokenizer encode→decode round-trip (task 5.2).

**Validates: Requirements 9.3**
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Optional import guard — skip entire module if transformers is unavailable
# ---------------------------------------------------------------------------

try:
    from transformers import AutoTokenizer  # type: ignore

    _transformers_available = True
except ImportError:
    _transformers_available = False

pytestmark = pytest.mark.skipif(
    not _transformers_available,
    reason="transformers library is not installed",
)

# ---------------------------------------------------------------------------
# Tokenizer fixture — loaded once per session to avoid repeated downloads
# ---------------------------------------------------------------------------

BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


@pytest.fixture(scope="session")
def bert_tokenizer():
    """Load the Bio_ClinicalBERT tokenizer once for the entire test session."""
    return AutoTokenizer.from_pretrained(BERT_MODEL_NAME)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_nonempty_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters=".,;:!?-'\"()/",
    ),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip())  # ensure at least one non-whitespace character


# ---------------------------------------------------------------------------
# Property 3: BERT tokenizer encode→decode round-trip
# ---------------------------------------------------------------------------


@given(text=_nonempty_text)
@settings(max_examples=20)
def test_bert_tokenizer_roundtrip(bert_tokenizer, text: str):
    """For any non-empty notes string, encode(text) == encode(decode(encode(text))).

    The round-trip property is that re-encoding the decoded token sequence
    produces the same token IDs as the original encoding — not that the
    decoded text equals the original (tokenization is lossy for some chars).

    **Validates: Requirements 9.3**
    """
    # Step 1: encode original text → token IDs
    token_ids = bert_tokenizer.encode(text, add_special_tokens=False)

    # Step 2: decode token IDs back to a string
    decoded_text = bert_tokenizer.decode(token_ids)

    # Step 3: re-encode the decoded string
    token_ids_roundtrip = bert_tokenizer.encode(decoded_text, add_special_tokens=False)

    assert token_ids == token_ids_roundtrip, (
        f"Round-trip token mismatch.\n"
        f"Original text:    {text!r}\n"
        f"Decoded text:     {decoded_text!r}\n"
        f"Original tokens:  {token_ids}\n"
        f"Round-trip tokens:{token_ids_roundtrip}"
    )
