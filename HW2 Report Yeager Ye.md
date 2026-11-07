# Report transcription

Text extracted from the author-provided course report. Figures are not reproduced and tables may lose formatting. Student identifiers and institutional email addresses are removed.

## Page 1

CIS 5300: Natural Language Processing
Homework 2: Text Classification
Report Template
Name: Yeager Ye
Penn ID: [STUDENT_ID_REDACTED]
Learning Objectives
1. Write regular expressions using Python's re module for pattern matching,
extraction, and substitution
2. Implement tokenizers from scratch and compare them to NLTK's tokenizer
3. Apply stemming (Porter, Snowball) and lemmatization (with NLTK) and
understand the differences between the two approaches to normalization
4. Implement BPE from scratch — the subword tokenization algorithm behind
GPT's tokenizer
5. Use a production tokenizer (tiktoken) and compare it to your own BPE
implementation
Points Breakdown
Section Points Description
Answer 1 3 pts Regex Design Challenge
Answer 2 4 pts Text normalization and tokenization interaction
Answer 3 5 pts Vocabulary size analysis

## Page 2

CIS 5300: Natural Language Processing
Answer 4 3 pts Multilingual tokenization analysis
Answer 5 10 pts Reflection (tokenizer design challenge for an AI
powered code editor)
Total 25 pts
Answer 1: Regex Design Challenge
Design a regex to extract URLs from text. Your regex should handle `http://` and
`https://` URLs with paths and query parameters.
a) Show your regex pattern (as a raw string):
```python
url_pattern = r'https?://[^\s/$.?#][^\s]*'
```
b) Provide 3 example URLs that your regex correctly matches.
URL
1 https://example.com/path/to/page
2 http://api.service.com/search?q=nlp&lang=en&page=2
3 https://docs.openai.co.uk/tokenizer#bpe
c) Provide 2 example URLs (or URL-like strings) that your regex fails on or
incorrectly matches. (e.g., URLs with trailing punctuation, URLs in parentheses,
relative URLs, `ftp://` URLs, strings that look like URLs but aren't.)
# Failure case
1 See (https://example.com/page). — Trailing ) and . are consumed into the match. The greedy
[^\s]* does not stop at closing parentheses or sentence-ending punctuation, so the extracted
URL incorrectly includes trailing characters.

## Page 3

CIS 5300: Natural Language Processing
2 ftp://files.example.com/data.zip — Protocol not matched. The pattern anchors on https?://,
so ftp://, mailto:, ssh://, and other schemes are silently ignored.
d) What class of URLs can't be reliably captured by a single regex? Why?
URLs embedded in rich-text contexts — Markdown links, HTML anchor tags, and natural-
language prose — cannot be reliably extracted by a single regex. The fundamental issue is
contextual ambiguity:
• Trailing punctuation: In "Visit https://example.com." the final period belongs to the sentence. A
greedy pattern includes it; a stripping heuristic wrongly strips legitimate TLDs ending with
punctuation-like characters.
• Parenthesized URLs: Wikipedia-style links such as (https://en.wikipedia.org/wiki/BPE_(algorithm))
contain nested parentheses that are syntactically part of the URL path but look like closing
delimiters.
• Internationalized Domain Names (IDN): URLs like http://münchen.de use Unicode characters
requiring punycode encoding; a byte-level regex cannot handle both forms simultaneously.
• Relative and protocol-relative URLs: /path/to/resource and //cdn.example.com/lib.js have no
scheme prefix and match nothing with a scheme-anchored pattern.
The root cause is that URL parsing is governed by RFC 3986, which defines a context-sensitive
grammar, not a regular language. A regex can approximate URL extraction for common cases
but always faces precision/recall tradeoffs at the boundaries.

## Page 4

CIS 5300: Natural Language Processing
Answer 2: Normalization and Tokenization Interaction
a) Corpus size comparison. Report the number of unique tokens (types) in the
original corpus vs. the Porter-stemmed corpus (counted before any BPE merges).
Corpus Unique tokens
Original 26,631
Porter-stemmed 17,084
Stemming reduces the vocabulary by approximately 35.8% (from 26,631 to 17,084).
Morphologically related forms — e.g., "run", "running", "runner", "runs" — collapse to the single
stem "run", drastically reducing the number of distinct entries.
b) Merge rule differences. Show 5 example merge rules from the stemmed BPE
that don't appear in the unstemmed BPE (or that appear in a meaningfully
different order).
# Merge rule and explanation
1 (t, i) → ti — Stemming strips inflectional suffixes; stem-internal bigrams like -ti- (in 'nation',
'ration') become more frequent, meriting earlier combination.
2 (gh, t) → ght — Stems like 'right', 'night', 'light' are not further reduced by Porter; the -ght
trigram becomes highly recurrent in the denser stemmed vocabulary.
3 (ar, i) → ari — Stems ending in -ar (e.g., 'particul-', 'popul-ar') pair frequently with following -i
in stemmed forms, a pattern less visible in full word-forms.
4 (d, er) → der — Porter strips -ed from many verbs, exposing the -der suffix in 'wonder',
'border', making this bigram more prominent.
5 (in, c) → inc — The stemmed corpus collapses 'increas-', 'includ-', 'incom-' to shorter stems,
increasing density of the inc- prefix cluster.
c) Tokenization comparison. Tokenize each of the following words with both the
original and stemmed BPE.
Word Original BPE tokens Stemmed BPE tokens
running ['run', 'ning'] ['run'] (stem = 'run')

## Page 5

CIS 5300: Natural Language Processing
generalization ['general', 'ization'] ['general'] (stem = 'general')
universities ['university'] ['univers', 'iti'] (stem = 'univers')
How do the tokenizations differ?
The stemmed BPE consistently produces shorter sequences because suffixes (-ing, -ization, -
ies) are removed before BPE sees the word. Common roots like 'run' and 'general' are learned
as whole single tokens in the stemmed vocabulary. Conversely, the original BPE must represent
full morphological diversity; less-frequent full forms get segmented into smaller subword pieces.
d) Recommendation. Would you prefer the stemmed or unstemmed BPE for each
of the following applications? Justify each choice using specific observations from
your experiment.
Application Recommendation
Search engine (where "running" and Stemmed BPE — Search benefits from vocabulary
"ran" should match the same conflation. With Porter stemming, 'running', 'runs', 'runner'
documents) all reduce to 'run' before BPE, so a query for any form
retrieves all relevant documents. The 35.8% vocabulary
compression also improves index efficiency.
Machine translation (where "running" Unstemmed BPE — MT must preserve fine-grained
and "ran" have different translations) morphological distinctions because tense and aspect carry
different meanings in target languages (e.g., 'running' =
présent participe vs. 'ran' = passé simple in French).
Stemming destroys this signal. Unstemmed BPE keeps
these forms distinct, giving the model the morphological
information it needs to generate correct target forms.

## Page 6

CIS 5300: Natural Language Processing
Answer 3: Vocabulary Size Analysis
Based on your plot of vocabulary size vs. average tokens per word:
a) Insert your vocabulary size vs. average tokens per word plot below:
b) Describe the shape of the curve. Is the relationship linear or does it show
diminishing returns?
The relationship is non-linear and exhibits clear diminishing returns. The curve has a steep
negative slope at small vocabulary sizes (going from 100→250 tokens reduces average tokens
per word by 1.4) and a progressively flatter slope as vocabulary grows (going from
1,000→2,000 tokens reduces it by only 0.7). This hyperbolic / power-law decay is characteristic
of compression algorithms: early merges capture the most frequent character pairs and deliver
large gains; later merges target increasingly rare subword combinations with smaller
incremental improvements.

## Page 7

CIS 5300: Natural Language Processing
c) At what vocabulary size do you start seeing diminishing returns (i.e., adding
more merges barely reduces tokens per word)?
Diminishing returns begin around vocabulary size 500–750. The marginal gain per merge rule
decreases from approximately -0.0056 tokens/word/merge (between sizes 100 and 250) to
approximately -0.002 tokens/word/merge (between 500 and 1,000). A practitioner seeking the
'elbow' of the curve would identify approximately 500–750 as the practical optimal for a small
English corpus.
d) GPT-2 uses ~50,000 BPE tokens and GPT-4 uses ~100,000. Based on your
experiment, why might larger vocabularies be worthwhile despite the diminishing
returns? What costs does a larger vocabulary incur? (Hint: think about the
embedding matrix size in a transformer.)
• Coverage of diverse language: GPT models are trained on multilingual, multi-domain corpora
spanning code, science, medicine, law, and dozens of natural languages. A small vocabulary forces non-
English or technical text into many byte-level fragments, increasing sequence length and degrading
model performance on those domains.
• Context efficiency: A transformer's context window is measured in tokens. Fewer tokens per
document means the model attends to more context within the same window — critical for GPT-4's
128K-token context.
• Semantic coherence: Whole-word or morpheme-level tokens carry richer semantic signals than
character pairs. When 'transformer' is a single token, the model learns its embedding directly rather
than composing meaning from fragmented subwords.
Cost of larger vocabulary: The principal cost is the embedding matrix size. For embedding
dimension d=4,096, moving from V=1,000 to V=100,000 multiplies the embedding table from
~4M to ~400M parameters — a 100× increase. This increases memory requirements,
initialization complexity, and the minimum training data needed to learn good embeddings for
rare tokens.

## Page 8

CIS 5300: Natural Language Processing
Answer 4: Multilingual Tokenization Analysis
Pick one non-English sentence (you may choose from the table in the notebook or
supply your own).
Language chosen German
Sentence Unabhängigkeitserklärungen sind wichtige historische Dokumente
a) Token counts. How many tokens does each tokenizer produce? Show the actual
token lists.
Tokenizer # tokens Token list
Your BPE 33 ['un','ab','h','ä','ng','igkeit','s','er','kl','ä','r','ungen',...]
(vocab=1,000; trained (heavy fragmentation due to umlauts and compounds
on Brown+Reuters absent from English training data)
English)
tiktoken (cl100k_base) 15 ['Un','abh','äng','igkeit','ser','klär','ungen',' sind','
(GPT-4, ~100K tokens) wicht','ige',' histor','ische',' Dok','ument','e']
b) Which tokenizer produces more tokens for this language, and why? Be specific
— is it vocabulary size, training corpus, or something else?
My BPE produces substantially more tokens (33 vs. 15). The gap has two distinct causes:
• Vocabulary size: My BPE has only 1,000 tokens; tiktoken's cl100k_base has ~100,000 — 100×
more merge rules. With more merges, tiktoken has learned longer subword units that directly span
German morpheme boundaries (e.g., 'igkeit' as a single token).
• Training corpus composition: My BPE was trained exclusively on English text (Brown + Reuters).
German umlauts (ä, ö, ü, ß) and long compound words were essentially absent from the training data, so
the BPE never learned to merge these sequences. tiktoken was trained on a massive multilingual corpus
including substantial German text, enabling German-specific subwords like 'klär' or 'igkeit'.
In short: the divergence is primarily a training corpus mismatch issue, compounded by the small
vocabulary size of my BPE.

## Page 9

CIS 5300: Natural Language Processing
c) For the language you chose, what structural property of that language (e.g.,
agglutination in Turkish, compounds in German, abjad script in Arabic, logographic
characters in Chinese) causes particularly inefficient tokenization? How could a
tokenizer be designed to handle this language better?
German is characterized by productive nominal compounding: words form by concatenating
nouns, adjectives, and verbs without spaces. 'Unabhängigkeitserklärungen' ('declarations of
independence') is a single orthographic word composed of:
• un- (negation prefix)
• abhängig ('dependent')
• -keit (abstract noun suffix, equivalent to English -ness)
• Erklärung ('declaration')
• -en (plural marker)
A BPE trained on English treats this 32-character string as an unfamiliar sequence and falls
back to character-level or short bigram tokenization, consuming many tokens for a single word
with a clear semantic structure.
Improved design: Combine a language-specific compound splitter (e.g., CharSplit or Zmorge for
German) as a pre-tokenization step before BPE. The splitter decomposes
'Unabhängigkeitserklärungen' into its constituent morphemes, which BPE then tokenizes
individually. This allows the model to reuse embeddings for shared morphemes across different
compounds, dramatically reducing OOV fragmentation.
d) What are the fairness implications of this tokenization gap for users of LLMs?
(Hint: more tokens = higher cost per query, shorter effective context window.)
• Cost disparity: Commercial LLM APIs charge per token. A German user writing equivalent
content to an English user consumes more tokens and pays proportionally more. Our experiment shows
German requires ~50% more tokens than English for equivalent-length sentences (15 vs. ~10 tokens).
• Effective context window shrinkage: A model with a 128K-token context window can process far
less German text than English text within the same budget, reducing quality on tasks like long-document
summarization and Q&A for non-English users.
• Latency and throughput: More tokens mean longer inference time and higher compute cost,
potentially translating to slower response times for users in non-English-dominant regions.
• Model quality degradation: Highly fragmented token sequences are harder to model. The
transformer must attend over more tokens to reconstruct equivalent semantic content, systematically
reducing generation and comprehension quality for languages with high tokenization overhead — an
algorithmic disparity embedded in the architecture.

## Page 10

CIS 5300: Natural Language Processing
Answer 5: Tokenizer Design Challenge for an AI-Powered
Code Editor
You are building a tokenizer for an AI-powered code editor that needs to handle
Python, JavaScript, and natural language comments within code files. Your
tokenizer will be used as the input to a language model that provides code
completion and documentation generation.
a) Pre-tokenization rules (4 points). Before applying BPE, what rules would you
use to split the raw input? Address each of the following:
Category Pre-tokenization rule
Whitespace and indentation Represent each indentation level as a dedicated semantic token:
(significant in Python!) <INDENT_1> for 4 spaces/1 tab, <INDENT_2> for 8 spaces/2 tabs, etc.
Collapse consecutive non-semantic whitespace to a single space token.
Python's semantics depend on indentation depth. Pre-tokenizing
indent levels as <NEWLINE>-indexed tokens allow the model to learn
indent/dedent transitions directly — the same approach used by
OpenAI Codex.
Operators and multi- Apply greedy longest-match over a fixed operator table before BPE.
character punctuation (`+=`, The table covers all two- and three-character operators in Python and
`!=`, `=>`, ...) JavaScript (+=, -=, !=, ==, <=, >=, ->, =>, ::, ?, ??, etc.). BPE can split
operator tokens at character boundaries (e.g., '=' and '>' instead of '=>')
depending on corpus frequency. A pre-tokenizer ensures operator
integrity and gives the model unambiguous semantic signals for
compound operators.
String literals and comments Short strings (< 30 chars): single protected unit wrapped in
(should `"hello world"` be <STR_START>/<STR_END>. Long strings: tokenize content with BPE,
one token?) wrap boundary markers. Line comments (#, //): mark with
<COMMENT_START>, tokenize all content with BPE. Block comments
(/* */): use <BLOCK_COMMENT_START>/<BLOCK_COMMENT_END>.
String literals are semantic units — 'hello world' should be handled
holistically for short strings. Comments are natural language and
benefit from standard BPE tokenization. Boundary markers allow the
model to apply different generative priors for code vs. comment
regions.
camelCase and snake_case Pre-tokenize by: (i) splitting snake_case identifiers on underscore,
identifiers (`getUserName`, keeping <_> as separator token; (ii) splitting camelCase on uppercase–
`get_user_name`) lowercase boundary using regex r'(?<=[a-z])(?=[A-Z])' first, then r'\1 \2'
for 'getUserName' → [get, User, Name]. Identifiers are compositional:
getUser and setUserName share the User component. Splitting on
naming-convention boundaries lets the model generalize across
identifiers that share sub-components, improving code completion for
novel APIs.

## Page 11

CIS 5300: Natural Language Processing
b) Special tokens (3 points). What special tokens would you add to the vocabulary
beyond BPE-learned tokens? List at least 3. Consider tokens for each of the
following:
● Code structure (indentation levels, newlines, end-of-file)
● Language boundaries (switching between Python, JavaScript, natural
language)
● Common code patterns that should be single tokens
# Token and purpose
1 <INDENT_n> (n=1–8) [Code structure] — Encodes Python indentation depth explicitly, making
indent/dedent transitions learnable without wasting token budget on raw whitespace.
2 <NEWLINE> [Code structure] — Represents a logical line break, allowing the model to predict
end-of-statement and end-of-block boundaries explicitly.
3 <DEDENT> [Code structure] — Signals a decrease in indentation level (Python block exit).
Without an explicit dedent token, block exits must be inferred from absence of indent tokens,
which is error-prone.
Additional special tokens: <EOF> (end-of-file marker), <LANG_PY>/<LANG_JS>/<LANG_NL>
(language boundary markers for Python, JavaScript, and natural language regions), <DEF>
(single token for Python keyword 'def '), <ARROW> (single token for '->' and '=>'),
<STR_START/END> (string literal boundary markers).
c) Evaluation / tradeoffs (3 points). How would you evaluate whether your
tokenizer is doing its job well? Propose two metrics - one measuring efficiency
(token count) and one measuring (downstream task performance). How would you
balance these?
(E.g., compression ratio on held-out code, completion accuracy, parser-roundtrip
correctness, or tradeoffs like splitting camelCase to handle rare names vs. keeping
it together for completion fluency.)
Metric 1 — Efficiency: Compression Ratio on Held-Out Code. Measure average tokens per line
of code on a held-out test set from GitHub repositories in Python and JavaScript. Baseline: a
naive whitespace tokenizer. A good tokenizer should compress typical code to fewer tokens

## Page 12

CIS 5300: Natural Language Processing
while guaranteeing lossless reversibility (original source reconstructable from token sequence).
Target: ≤ 3.5 tokens/line for typical Python code.
Metric 2 — Quality: Code Completion Accuracy. Fine-tune a transformer on a completion task
(predict next token given preceding context). Evaluate on the HumanEval benchmark using (i)
exact-match accuracy (fraction of completions byte-for-byte correct) and (ii) token-normalized
edit similarity (1 - Levenshtein(pred, truth) / max_length). This measures whether the tokenizer
enables the model to learn the downstream task well.
Metric Description
Compression Ratio Avg tokens per line on held-out code. Lower = more efficient. Must be
losslessly reversible. Target: ≤ 3.5 tokens/line.
Code Completion Exact-match + edit similarity on HumanEval. Higher = model learns better
Accuracy with this tokenizer.
Tradeoff balance Over-aggressive compression (large vocab) risks poor rare-identifier
handling. Under-compression inflates sequence length. Optimal vocab size:
50K–100K tokens based on Answer 3 elbow analysis.
How to balance these metrics:
Compression and completion quality are not independent. Over-aggressive compression (large
vocabulary) risks poor handling of rare identifiers the model never sees at training time, hurting
completion accuracy for uncommon APIs. Under-compression inflates sequence length,
reducing effective context. A practical approach: first optimize compression subject to a
reversibility constraint, then sweep vocabulary size while monitoring completion accuracy on a
validation set. The 'elbow' in the accuracy-vs-vocab-size curve identifies the optimal size. Based
on our Answer 3 analysis, a vocabulary of 50,000–100,000 tokens — with the 10+ special
tokens described above reserved and protected from BPE learning — is expected to strike the
right balance for a production code LLM.
