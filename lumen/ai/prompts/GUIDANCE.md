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

### Examples

- Put examples under a `## Examples` header and mark them as illustrative when they
  contain realistic-looking tables, columns, metrics, or tool output, e.g.
  `## Examples (illustrative — names below are placeholders, not real data)`.
  This prevents the model from conflating example content with the user's live data,
  which renders in the same prompt with the same formatting.

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
  clearer prose style; reserve `CRITICAL`/`MUST`/`NEVER` for the rare load-bearing rule.

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

## Author checklist

Before adding or editing a prompt, confirm:

- [ ] Top-level sections use `##`, subsections `###`; no stray `#` headers.
- [ ] Examples are under `## Examples` and marked illustrative if they look like real data.
- [ ] No rule is stated more than once; emphasis is reserved for genuinely load-bearing rules.
- [ ] Shared content comes from the base via `{{ super() }}`, not copy-paste.
- [ ] The output contract (fields/format/length) is stated.
- [ ] Any injected `memory['data']` is labeled a summary and capped counts are flagged.
- [ ] The template parses (`jinja2.Environment().parse(...)`).
