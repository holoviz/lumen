# Prompt authoring guidance

Design notes and conventions for the Jinja2 prompt templates in this directory.
The goal is one consistent house style grounded in current prompt/context-engineering
practice. When in doubt, prefer the smallest, clearest prompt that reliably produces
the behavior you want.

## Core principles

1. **Aim for the right altitude.** Be specific enough to steer behavior, flexible
   enough to leave the model good heuristics. Avoid both extremes: brittle if-else
   logic hardcoded into prose, and vague hand-waving that assumes shared context.
   If a rule encodes a one-off workaround, it probably belongs in code, not the prompt.

2. **Minimal high-signal context, not maximal.** The model has a finite attention
   budget and recall degrades as the context grows ("context rot"). Minimal does not
   mean short — it means no low-signal filler. A rule stated once, clearly, beats the
   same rule restated three times with rising emphasis. If the model keeps ignoring an
   instruction, diagnose why rather than adding a louder copy.

3. **Examples are pictures worth a thousand words — so curate them.** Few-shot
   examples are strongly encouraged, but diversity beats volume: a few diverse,
   canonical examples outperform a long list of edge cases. Trim examples that
   illustrate the same pattern, and prefer abstract placeholders (`<N> rows`, `<table>`)
   over realistic-looking numbers that the model might mistake for live data.

4. **Specify the output contract.** State the expected output shape (fields, format,
   length) explicitly. If you do not, the model picks one, and it may not match the
   downstream parser or response model.

5. **Tools are part of the prompt.** Keep tool sets minimal and non-overlapping. If a
   human can't say which tool applies in a situation, the model can't either.

6. **Iterate from a minimal baseline against real failures.** Start minimal, observe
   actual failure modes, then add the smallest instruction or example that fixes the
   observed failure — not a speculative rule for a failure you haven't seen.

## Repo conventions

### Structure & headings

- Use `##` for top-level sections and `###` for subsections. Do **not** use single `#`
  for markdown section headers (it is reserved for nothing — keep it out so YAML `#`
  comments inside code fences stay unambiguous).
- Group content into clear sections (`## Instructions`, `## Examples`, `## Context`,
  output description). Markdown headers are the house delimiter.
- No trailing colons on headings (`## Current Knowledge`, not `## Current Knowledge:`).
- Title Case, not ALL CAPS (`## Examples`, not `## EXAMPLES`). Two spellings of one
  section read as two sections.
- **Never put a `{%- … -%}` tag directly under a heading.** The leading `-` strips the
  newline that separates the heading from its body, so

  ```jinja
  ## Instructions
  {%- if memory.get('sql') -%}
  Analyze if the SQL query…
  ```

  renders as `## InstructionsAnalyze if the SQL…`. Drop the leading `-` on the tag
  that follows a heading. This is invisible in the source and only shows up in the
  rendered prompt, so it is easy to introduce and hard to spot.

### Examples

- Put examples under a `## Examples` header and mark them as illustrative when they
  contain realistic-looking tables, columns, metrics, or tool output, e.g.
  `## Examples (illustrative — names below are placeholders, not real data)`.
  This prevents the model from conflating example content with the user's live data,
  which renders in the same prompt with the same formatting.

### Conditional sections

- **Guard on the value, not the key.** `{% if 'sql' in memory %}` is true whenever the
  key exists, including when it holds `''`, `[]` or `None` — so the section's header
  renders with nothing under it, and the model is left to interpret an empty promise.
  Use `{% if memory.get('sql') %}`, or `{% if memory.get('data') is not none %}` when
  an empty value is itself meaningful (an empty result set is a finding; a missing one
  is not).

### Inheritance

- Extend the base templates (`Actor/main.jinja2`, `Agent/main.jinja2`) and override
  blocks rather than rewriting them.
- When a child needs the parent's section plus additions, call `{{ super() }}` and add
  the delta. Do **not** copy-paste and re-implement a parent block — the copies drift.
  (Today `VegaLiteAgent/main_altair` and `DeckGLAgent/main_pydeck` re-implement the
  base context block and have already diverged; new agents should not follow that.)
- Keep shared rules (e.g. the Snowflake upper-case note) in the base context block so
  children inherit one canonical wording.

### Style

- One voice per prompt. The repo historically had two dialects (measured prose in the
  orchestration/data agents; terse caps-and-emoji in the view agents). Prefer the
  clearer prose style, then compress it with caveman compression (see below).
- **At most two emphasis markers per template** (`CRITICAL`, `MUST`, `NEVER`, `ALWAYS`,
  `IMPORTANT`). They only carry signal while they are rare; a prompt where every rule
  shouts has no load-bearing rules.
- **No emoji or pictographs.** Say `Blocked` rather than `❌ BLOCKED`, `Valid` rather
  than `✓ VALID`. They cost tokens without adding anything the word does not already
  say, and they are what the caps-and-emoji dialect was made of. Two exceptions are
  functional rather than decorative: `★`, which `schemas.py` emits to mark derived
  tables, and `°` in example unit strings.

## The data-summary contract (highest-leverage area)

Several agents inject `memory['data']` (a capped, sampled summary produced by
`describe_data`) directly into the prompt: `ChatAgent`, `AnalysisAgent`, `SQLAgent`,
`ValidationAgent`, `FunctionTool`. Because this summary drives what the model reports
to the user, get the framing right:

- **Label it as a summary, not the dataset.** Say "a summary of the data," not
  "the current dataset." It is statistics + samples, not the rows themselves.
- **Don't present sampled or capped values as the whole truth.** Summaries are often
  truncated or sampled for display. When a prompt surfaces figures (row counts,
  cardinalities), make clear which are display limits or samples rather than the
  table's true values.
- **Don't reuse one label for two things.** If a prompt shows both a catalog/metaset
  summary and a query-result summary, give them distinct headers.

## Sizing injected payloads

Anything a template injects — data summaries, schemas, doc chunks, tool results —
competes with the instructions for the model's attention budget, so cap it in the code
that produces it, not in the template.

- **Cap in tokens, not characters** (`lumen.ai.utils.truncate_to_tokens`). Characters
  per token vary by roughly 1.6x across the content we inject: English prose runs near
  4, dense YAML and whitespace-aligned numeric tables nearer 2.5. One nominal character
  cap therefore admits wildly different token counts, and the payloads that blow the
  budget are exactly the dense ones.
- **State the ratio you assumed** where you set a constant, so the next person can tell
  a measured budget from a guessed one.
- **Say what was dropped.** `truncate_to_tokens` appends
  `... (truncated, showing N of M tokens)`. A bare ellipsis leaves the model to guess
  whether it is looking at everything, and guessing wrong drives redundant tool calls.
- **Report full shape before any sample.** A preview that says `53 rows x 7 columns`
  followed by 5 rows saves the model a `SELECT COUNT(*)` round trip; 50 rows with no
  total does not.

## Caveman compression

Prompt prose competes for the model's attention budget alongside every injected
payload. Compress instructions, context descriptions, and rules by stripping
grammatical scaffolding while preserving every content word that carries meaning.
The technique is aggressive but mechanical: remove stop words, keep semantics.

### Remove

- Articles: a, an, the
- Auxiliary verbs: is, are, was, were, am, be, been, being, have, has, had,
  do, does, did
- Prepositions when meaning stays clear: of, for, to, in, on, at
- Pronouns when context is clear: it, this, that, these, those
- Intensifiers: very, quite, rather, somewhat, really, extremely

### Keep

- Nouns, main verbs, adjectives that add meaning
- Numbers and quantifiers (at least, approximately, more than, many)
- Uncertainty qualifiers (appears to be, seems, might)
- Prepositions that define relationships (from, with, without, stuck to)
- Time and frequency words (every Tuesday, weekly, daily, always, never)
- Names, titles, technical terms, domain-specific language
- Negations (not, no, never, without)

### Judgment calls

- Keep a preposition when it defines a relationship ("made from wood" keeps
  *from*; "system for processing" drops *for*).
- Keep in/on/at when they specify location or position; drop when grammatical only.
- Drop is/are/was/were unless part of a passive voice that carries meaning.
- **Never compress inside fenced code blocks.** Preserve SQL, YAML, JSON, Python,
  column names, and example specs verbatim — compress only the prose surrounding them.
- Few-shot example responses may be compressed, but the demonstrated output
  format (code, specs, structured fields) must stay intact.

### Examples

```
"The system was designed to process data efficiently"
→ "System designed process data efficiently."
(Removed: The, was, to)

"There were at least 20 people"
→ "At least 20 people."
(Kept: at least — quantifier matters; removed: There were)

"Made from wood and metal"
→ "Made from wood and metal."
(Kept: from — shows material relationship)
```

## Author checklist

Before adding or editing a prompt, confirm:

- [ ] Top-level sections use `##`, subsections `###`; no stray `#` headers; Title Case.
- [ ] No heading is followed by a `{%- … -%}` tag that strips the newline after it.
- [ ] Conditional sections guard on the value (`memory.get('k')`), not the key.
- [ ] Examples are under `## Examples` and marked illustrative if they look like real data.
- [ ] No rule is stated more than once; at most two emphasis markers; no emoji.
- [ ] Prose is caveman-compressed: articles, auxiliary verbs, and redundant
      prepositions removed; content words preserved; code blocks untouched.
- [ ] Shared content comes from the base via `{{ super() }}`, not copy-paste.
- [ ] The output contract (fields/format/length) is stated.
- [ ] Any injected `memory['data']` is labeled a summary and capped counts are flagged.
- [ ] Injected payloads are capped in tokens, and the cap says what was dropped.
- [ ] The template parses (`jinja2.Environment().parse(...)`).

`lumen/tests/ai/test_prompts.py` enforces the mechanical items above across every
template in this directory, and checks that each `PROMPTS_DIR / …` path registered in
the code exists on disk. If you break one of these conventions, CI will tell you which
file and line.
