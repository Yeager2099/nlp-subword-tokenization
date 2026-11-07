"""
HW02: Text Processing & BPE
CIS 5300: Natural Language Processing | Module 02
Author: Yeager Ye
"""

import re
import string
import copy
from collections import Counter, defaultdict

import numpy as np
import nltk
import tiktoken

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

print("All imports successful.")


# ============================================================
# Section 1: Regular Expressions
# ============================================================

def has_double_o(word: str) -> bool:
    """Check if a word contains two or more consecutive 'o' characters.

    Uses re.search() with a pattern that matches 'o' repeated 2+ times.
    Case-insensitive.

    Args:
        word: A string to check.

    Returns:
        True if the word contains 'oo' (or 'ooo', etc.), False otherwise.
    """
    return bool(re.search(r'o{2,}', word, re.IGNORECASE))


def extract_emails(text: str) -> list:
    """Extract all email addresses from the given text.

    Uses re.findall() to find all substrings matching an email pattern.

    Args:
        text: A string that may contain email addresses.

    Returns:
        A list of strings, each an email address found in the text.
    """
    pattern = r'[\w.\-]+@[\w.\-]+\.[a-zA-Z]{2,4}'
    return re.findall(pattern, text)


def reformat_dates(text: str) -> str:
    """Convert all dates from MM/DD/YYYY to YYYY-MM-DD format.

    Uses re.sub() with capturing groups to rearrange date components.

    Args:
        text: A string that may contain dates in MM/DD/YYYY format.

    Returns:
        The text with all MM/DD/YYYY dates converted to YYYY-MM-DD.
    """
    pattern = r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(\d{4})\b'
    return re.sub(pattern, r'\3-\1-\2', text)


# ============================================================
# Section 2: Tokenization
# ============================================================

def whitespace_tokenize(text: str) -> list:
    """Tokenize text by splitting on whitespace.

    Args:
        text: A string to tokenize.

    Returns:
        A list of tokens (strings), split on whitespace. Empty tokens
        from multiple consecutive spaces should not be included.
    """
    return text.split()


def improved_tokenize(text: str) -> list:
    """Tokenize text by splitting on whitespace and separating punctuation.

    For each whitespace-delimited token, separate leading and trailing
    non-alphanumeric characters into their own tokens. Interior punctuation
    (e.g., hyphens in 'state-of-the-art', apostrophes in "don't") should
    also be separated.

    Args:
        text: A string to tokenize.

    Returns:
        A list of tokens with punctuation separated. Empty tokens should
        not be included.
    """
    # Use regex to split into alphanumeric sequences and individual non-alphanumeric/non-whitespace chars
    return re.findall(r'[a-zA-Z0-9]+|[^a-zA-Z0-9\s]', text)


def nltk_preprocess(text: str) -> list:
    """Tokenize and preprocess text using NLTK.

    Steps:
        1. Tokenize with nltk.word_tokenize()
        2. Lowercase all tokens
        3. Remove English stopwords
        4. Keep only alphabetic tokens (str.isalpha())

    Args:
        text: A string to tokenize and preprocess.

    Returns:
        A list of preprocessed tokens (lowercase, no stopwords, alphabetic only).
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens


# ============================================================
# Section 3: Text Normalization
# ============================================================

# Words to stem
STEM_WORDS = [
    "running", "runner", "ran",
    "better", "best", "good",
    "studies", "studying", "studied",
    "university", "universities",
    "generalization", "generalize",
    "relational", "relation", "relate",
    "organization", "organize",
    "happiness", "happy", "happily",
]


def compare_stemmers(words: list) -> dict:
    """Apply Porter and Snowball stemmers to a list of words.

    Args:
        words: A list of words to stem.

    Returns:
        A dict mapping each word to a tuple (porter_stem, snowball_stem).
        Example: {"running": ("run", "run"), "studies": ("studi", "studi")}
    """
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    return {word: (porter.stem(word), snowball.stem(word)) for word in words}


def compare_stem_vs_lemma(words: list) -> dict:
    """Compare stemming and lemmatization on a list of words.

    Args:
        words: A list of words to process.

    Returns:
        A dict mapping each word to a tuple:
            (lemma_as_noun, lemma_as_verb, porter_stem)
    """
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    result = {}
    for word in words:
        lemma_n = lemmatizer.lemmatize(word, pos='n')
        lemma_v = lemmatizer.lemmatize(word, pos='v')
        p_stem = porter.stem(word)
        result[word] = (lemma_n, lemma_v, p_stem)
    return result


# ============================================================
# Section 4: Byte-Pair Encoding
# ============================================================

def compute_word_freq(corpus_docs: list, min_freq: int = 50) -> dict:
    """Compute word frequencies from a corpus, keeping only frequent words.

    Steps:
        1. For each document, tokenize using nltk_preprocess()
        2. Count the frequency of each word across all documents
        3. Filter to keep only words with frequency >= min_freq
        4. Convert each word to a tuple of characters

    Args:
        corpus_docs: A list of document strings.
        min_freq: Minimum frequency threshold. Words below this are discarded.

    Returns:
        A dict mapping character tuples to their frequency counts.
        Example: {('h', 'e', 'l', 'l', 'o'): 42, ('w', 'o', 'r', 'l', 'd'): 38}
    """
    word_counts = Counter()
    for doc in corpus_docs:
        tokens = nltk_preprocess(doc)
        word_counts.update(tokens)

    # Filter by min_freq and convert to character tuples
    result = {}
    for word, freq in word_counts.items():
        if freq >= min_freq:
            result[tuple(word)] = freq
    return result


def compute_vocab(word_freqs: dict) -> list:
    """Build the initial character-level vocabulary from word frequencies.

    Collects all unique characters across all words in the frequency dict
    and returns them as a sorted list.

    Args:
        word_freqs: Dict mapping character tuples to frequency counts.

    Returns:
        A sorted list of unique characters (strings of length 1).
    """
    chars = set()
    for word_tuple in word_freqs.keys():
        for ch in word_tuple:
            chars.add(ch)
    return sorted(chars)


def compute_pair_freqs(word_freqs: dict) -> dict:
    """Count adjacent token pair frequencies across all words.

    For each word (a tuple of tokens) with a given frequency, count every
    adjacent pair and weight by the word's frequency.

    Args:
        word_freqs: Dict mapping token tuples to frequency counts.

    Returns:
        A dict mapping (token1, token2) pairs to their weighted frequency.
        Example: {('t', 'h'): 5400, ('h', 'e'): 4200, ...}
    """
    pair_counts = defaultdict(int)
    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] += freq
    return dict(pair_counts)


def merge_pair(pair: tuple, word_freqs: dict) -> dict:
    """Merge all occurrences of a token pair in the word frequency dictionary.

    Creates a NEW dictionary (does not modify the input) where every occurrence
    of the adjacent pair is replaced with the merged token.

    Args:
        pair: A tuple of two tokens to merge, e.g., ('e', 's').
        word_freqs: Dict mapping token tuples to frequency counts.

    Returns:
        A new dict with the pair merged in all words.
    """
    new_word_freqs = {}
    merged_token = pair[0] + pair[1]

    for word_tuple, freq in word_freqs.items():
        tokens = list(word_tuple)
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        key = tuple(new_tokens)
        if key in new_word_freqs:
            new_word_freqs[key] += freq
        else:
            new_word_freqs[key] = freq

    return new_word_freqs


def train_bpe(word_freqs: dict, base_vocab: list, target_vocab_size: int = 1000) -> tuple:
    """Train BPE by iteratively merging the most frequent pair.

    Args:
        word_freqs: Dict mapping character tuples to frequency counts.
        base_vocab: List of initial characters in the vocabulary.
        target_vocab_size: Target total vocabulary size (base chars + merges).

    Returns:
        A tuple of (vocab, merges, final_word_freqs) where:
            vocab: list of all tokens (base characters + merged tokens)
            merges: list of (pair, merged_token) tuples in the order learned.
                    e.g., [((\"e\", \"s\"), \"es\"), ((\"es\", \"t\"), \"est\"), ...]
            final_word_freqs: the word_freqs dict after all merges
    """
    vocab = list(base_vocab)
    merges = []
    current_word_freqs = copy.deepcopy(word_freqs)

    while len(vocab) < target_vocab_size:
        pair_freqs = compute_pair_freqs(current_word_freqs)
        if not pair_freqs:
            break

        # Find best pair: highest frequency, ties broken by smallest tuple (lexicographic)
        sorted_pairs = sorted(pair_freqs.items(), key=lambda item: (-item[1], item[0]))
        best_pair = sorted_pairs[0][0]

        # Merge
        merged_token = best_pair[0] + best_pair[1]
        current_word_freqs = merge_pair(best_pair, current_word_freqs)
        vocab.append(merged_token)
        merges.append((best_pair, merged_token))

        if len(merges) % 100 == 0:
            print(f"  Merges done: {len(merges)}, Vocab size: {len(vocab)}")

    return vocab, merges, current_word_freqs


def bpe_tokenize(word: str, merges: list) -> list:
    """Tokenize a single word using learned BPE merge rules.

    Applies merge rules in the order they were learned during training.

    Args:
        word: A string to tokenize (should be lowercase).
        merges: List of (pair, merged_token) tuples from train_bpe().

    Returns:
        A list of subword tokens.
        Example: bpe_tokenize("newest", merges) -> ['new', 'est']
    """
    tokens = list(word)

    for pair, merged_token in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


# ============================================================
# Section 5: Modern Tokenizers
# ============================================================

def tiktoken_analyze(text: str, encoding_name: str = "cl100k_base") -> tuple:
    """Tokenize text using tiktoken and return analysis.

    Args:
        text: Text to tokenize.
        encoding_name: Name of the tiktoken encoding (default: GPT-4's encoding).

    Returns:
        A tuple of (num_tokens, token_strings) where:
            num_tokens: int, number of tokens
            token_strings: list of str, the decoded text for each token
    """
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)
    token_strings = [enc.decode([tid]) for tid in token_ids]
    return len(token_ids), token_strings


def compare_tokenizers(words: list, merges: list, encoding_name: str = "cl100k_base") -> list:
    """Compare BPE and tiktoken tokenization for a list of words.

    Args:
        words: List of words to tokenize.
        merges: Merge rules from your BPE training.
        encoding_name: tiktoken encoding name.

    Returns:
        A list of tuples: (word, bpe_tokens, tiktoken_tokens, bpe_count, tiktoken_count)
    """
    enc = tiktoken.get_encoding(encoding_name)
    results = []
    for word in words:
        bpe_tokens = bpe_tokenize(word, merges)
        tt_ids = enc.encode(word)
        tt_tokens = [enc.decode([tid]) for tid in tt_ids]
        results.append((word, bpe_tokens, tt_tokens, len(bpe_tokens), len(tt_tokens)))
    return results


# ============================================================
# Quick sanity check
# ============================================================
if __name__ == "__main__":
    # Section 1 checks
    assert has_double_o("food") == True
    assert has_double_o("fox") == False
    assert has_double_o("OOZE") == True
    assert has_double_o("potato") == False
    print("has_double_o: OK")

    text1 = "Contact us at support@example.com or john.doe@university.edu for help."
    assert extract_emails(text1) == ['support@example.com', 'john.doe@university.edu']
    assert extract_emails("alice_99@mail-server.info test") == ['alice_99@mail-server.info']
    print("extract_emails: OK")

    text_d = "The meeting is on 03/15/2026 and the deadline is 12/01/2025."
    assert reformat_dates(text_d) == "The meeting is on 2026-03-15 and the deadline is 2025-12-01."
    print("reformat_dates: OK")

    # Section 2 checks
    assert whitespace_tokenize("Hello world!") == ["Hello", "world!"]
    assert whitespace_tokenize("  extra   spaces  ") == ["extra", "spaces"]
    print("whitespace_tokenize: OK")

    assert improved_tokenize("Hello, world!") == ["Hello", ",", "world", "!"]
    assert improved_tokenize("(test)") == ["(", "test", ")"]
    print("improved_tokenize: OK")

    result = nltk_preprocess("The quick brown fox jumps over the lazy dog.")
    assert "the" not in result
    assert "quick" in result
    print("nltk_preprocess: OK")

    # Section 3 checks
    stem_results = compare_stemmers(STEM_WORDS)
    assert stem_results["running"][0] == "run"
    print("compare_stemmers: OK")

    lemma_results = compare_stem_vs_lemma(STEM_WORDS)
    assert lemma_results["studies"][1] == "study"
    print("compare_stem_vs_lemma: OK")

    # Section 4 mini-test (no corpus needed)
    test_wf = {('n', 'e', 'w', 'e', 's', 't'): 6}
    result = merge_pair(('e', 's'), test_wf)
    assert ('n', 'e', 'w', 'es', 't') in result
    # Original unchanged
    assert ('n', 'e', 'w', 'e', 's', 't') in test_wf
    # Duplicate char test
    test_wf2 = {('a', 's', 's', 'e', 's', 's'): 3}
    result2 = merge_pair(('s', 's'), test_wf2)
    assert ('a', 'ss', 'e', 'ss') in result2
    print("merge_pair: OK")

    print("\nAll quick sanity checks passed!")
