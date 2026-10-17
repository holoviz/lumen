# Contribution issue ranking for HoloViz Lumen

## Assessment record

| Item | Value |
|---|---|
| Repository | [holoviz/lumen](https://github.com/holoviz/lumen) |
| Local checkout | `C:\Users\zainp\Desktop\lumen\lumen-forkes` |
| Code branch / commit inspected | `main` / `61c4c7f25c5fe5c2d18d63be0fe2cc0a32748d60` |
| Assessment date | 2026-10-17 (Asia/Calcutta) |
| Issue discovery window | 2026-07-19 through 2026-10-17, approximately 90 days |
| Older issue policy | Included only when still open, relevant, actionable, and not covered by an active or merged PR |
| Primary onboarding reference | [`REPOSITORY_EXPLORATION.md`](REPOSITORY_EXPLORATION.md) |
| Source verification | GitHub issue/PR metadata, issue bodies and comments, local implementation and tests |
| Implementation status | No source code, branch, fix, or pull request was created |

The recency window is a practical review boundary, not a repository rule. It captures the current issue stream while allowing older issues with clear unresolved value to remain visible. GitHub metadata showed active issues created as recently as 2026-10-15 and older active issues with updates in August/September 2026.

## Method and exclusion rules

Candidates were first checked for state, discussion, linked PRs, current code presence, and testability. Only then were surviving candidates scored. The following rules were applied before scoring:

- Exclude closed issues, issues demonstrably fixed by a merged PR, and duplicates where the duplicate relationship is supported by GitHub discussion.
- Exclude issues whose proposed work is already covered by an open PR when the PR description and changed scope match the issue.
- Keep issues with an active PR only when a meaningful contribution remains, such as missing tests, review-requested corrections, or incomplete implementation. Lower their opportunity score.
- Exclude vague requests when the expected outcome, reproduction, or actionable scope is too unclear to justify implementation planning. Keep them as “needs clarification” only when the issue is still plausibly valuable.
- Do not recommend a contribution that duplicates a PR authored by the current agent. No current-agent GitHub PR identity was available, and no local PR branch was created.

The ranking is therefore a contribution shortlist, not a list of every open issue.

## Scoring model

Each criterion is scored from 0 to 5. Higher is favorable for every criterion, including safety: a high safety score means lower blast radius and lower regression risk. The weighted score is out of 5.

| Criterion | Weight | Meaning of a high score |
|---|---:|---|
| User impact and severity | 25% | The issue affects an important or user-visible workflow, with meaningful consequences. |
| Feasibility | 15% | The likely change is understood, local enough to implement, and does not require unusual infrastructure. |
| Reproduction/evidence quality | 15% | The issue has a runnable reproduction, concrete traceback, or strong implementation evidence. |
| Expected-outcome clarity | 15% | The desired behavior and acceptance boundary are specific. |
| Safety / blast-radius control | 10% | The change can be isolated and regression risk is manageable. |
| Testability | 10% | The behavior can be tested deterministically with repository tools or mocks. |
| Remaining contribution opportunity | 10% | No active PR covers the work, or a clear unclaimed part remains. |

An active PR does not automatically reject an issue, but it reduces the opportunity score. A missing external service reduces evidence, feasibility, and testability even when the user impact is high. Scores are judgment calls grounded in the evidence below; they are not GitHub labels.

## Ranked candidates

### 1. [#2184 — Anthropic provider cannot use Claude 4.7 and later, and the default Haiku 4.5 is due for retirement](https://github.com/holoviz/lumen/issues/2184)

| Field | Assessment |
|---|---|
| Category | Bug / maintenance / provider compatibility |
| GitHub status | Open; created 2026-10-15; 0 comments; label `lumen-ai` |
| Linked PR | No linked PR found in the inspected issue/PR search |
| Reproducibility | Likely, not confirmed: the issue gives exact provider rejection reasons but says the author could not test against the API |
| Impact | High: Anthropic users cannot use current model generations, and the default model has a stated retirement timeline |
| Relevant code | `lumen/ai/llm.py:2798-3127`, especially `Anthropic`, `AnthropicBedrock`, `temperature`, `mode`, `model_kwargs`, and `select_models`; tests in `lumen/tests/ai/test_llm.py` |
| Blast radius | Medium-high. The provider base, Instructor mode, Anthropic streaming/tool calls, model selection, and Bedrock aliases are involved. Other providers should remain unchanged. |
| Feasibility | Medium. The issue proposes model-aware omission of temperature and a structured-output mode change, but exact Anthropic API semantics need verification. |
| Testability | Medium. Request construction and retry behavior can be mocked; real compatibility requires provider credentials and current API models. |

The issue identifies two concrete incompatibilities: Claude 4.7+ rejects non-default sampling parameters, while newer Sonnet/Opus models reject the forced `tool_choice` behavior used by the current Instructor mode. The local implementation confirms the reported design pressure: `Anthropic.temperature` defaults to `0.7`, the provider advertises only `claude-haiku-4-5`, `claude-sonnet-4-6`, and `claude-opus-4-6`, and the source comment explicitly says newer models fail because of temperature and forced tool choice (`lumen/ai/llm.py:2810-2825`). Both `AnthropicBedrock` and `Bedrock` duplicate the model-selection boundary (`lumen/ai/llm.py:3095-3168`).

Expected behavior: current supported Anthropic models should be selectable and should receive only parameters/modes accepted by the selected model; the default should not depend on a model approaching retirement. The likely safe implementation boundary is provider-specific request construction, with model-aware tests for omitted temperature, supported output modes, aliases, and Bedrock IDs. The main uncertainty is whether the issue’s stated model behavior is still accurate on the assessment date and which Instructor mode is accepted by each current model.

**Score:** impact 5, feasibility 3, evidence 3, clarity 4, safety 2, testability 3, opportunity 5 → **3.60 / 5**.

**Why rank first:** it is the highest-impact issue without an identified active PR. It also aligns with an existing, well-defined provider abstraction and test suite. The trade-off is external API validation: a contributor must not claim completion from mocked tests alone.

### 2. [#2134 — SourceAgent derives table names that start with a digit](https://github.com/holoviz/lumen/issues/2134)

| Field | Assessment |
|---|---|
| Category | Confirmed bug |
| GitHub status | Open; created 2026-10-08; label `bug`, `lumen-ai` |
| Linked PR | [#2139](https://github.com/holoviz/lumen/pull/2139), open, one commit, two files, changes requested; authored by `rimsha152007-dotcom` |
| Reproducibility | Confirmed locally on this checkout |
| Impact | Medium-high for source actions whose shortest argument is numeric; the generated table cannot be queried |
| Relevant code | `lumen/ai/agents/source.py:485-523`, `lumen/util.py:753-766`, `lumen/sources/duckdb.py`, source tests |
| Blast radius | Low-medium. The issue is local to derived table naming, but `normalize_table_name` is a shared utility and may affect all source names. |
| Feasibility | High. The failure and a narrow fix are clear. |
| Testability | High. DuckDB and naming tests can run without a live service. |
| Remaining opportunity | Low-medium. The open PR already changes the two obvious locations, but review identifies missing edge-case tests and collision questions. |

The issue’s reproduction was executed exactly against the current checkout. `SourceAgent._derive_table_name(...)` returned `20_load_cells`; using that name in `SELECT * FROM 20_load_cells` raised DuckDB `ParserException: syntax error at or near "20"`. The local code confirms the cause: it builds `parts = items[:MAX_PARTS]` and appends `action_name` afterward (`lumen/ai/agents/source.py:516-523`).

PR #2139 moves `action_name` to the front and prefixes digit-leading normalized names. It is still open, and review requested tests for empty/symbol-only names and possible prefix collisions. A useful contribution remains only if the contributor coordinates with the PR author and adds robust tests or resolves the review concerns; starting an independent fix would duplicate active work.

**Score:** impact 4, feasibility 5, evidence 5, clarity 5, safety 3, testability 5, opportunity 1 → **4.10 / 5**.

**Why rank second:** it is the strongest technically verified bug and the easiest to validate, but the active PR materially reduces contribution opportunity. The best next-stage action is review/support work on #2139, not an independent implementation.

### 3. [#2155 — Validation reports multi-chart answers as incomplete because it only sees the last chart](https://github.com/holoviz/lumen/issues/2155)

| Field | Assessment |
|---|---|
| Category | Confirmed bug |
| GitHub status | Open; created 2026-10-15; label `bug`, `lumen-ai` |
| Linked PR | [#2156](https://github.com/holoviz/lumen/pull/2156), open, one commit, five files, reviewer requested; authored by `ghostiee-11` |
| Reproducibility | Likely from the supplied two-run reproduction; not independently executed because it needs an LLM/UI flow |
| Impact | Medium-high: multi-chart answers are rendered but falsely reported as incomplete, damaging trust in validation and follow-up behavior |
| Relevant code | `lumen/ai/agents/vega_lite.py:936-1009`, `lumen/ai/agents/validation.py`, `lumen/ai/report.py`, chart/editor tests |
| Blast radius | Medium-high. It crosses Vega-Lite editor output, plan context, validation prompts, and UI feedback. |
| Feasibility | Medium-high. The reported last-editor return is a precise suspected cause. |
| Testability | Medium-high with mocked editors and validation inputs; end-to-end behavior requires an LLM/UI harness. |
| Remaining opportunity | Low-medium because PR #2156 directly addresses the issue. |

The issue reports three charts appearing in the All tab while ValidationAgent sees only the final chart. Local code confirms the key path: `VegaLiteAgent.respond` ends with `out_context = await editors[-1].render_context()` (`lumen/ai/agents/vega_lite.py:1009`), matching the issue’s diagnosis. PR #2156 proposes passing all chart specs as `views` while retaining the final chart as `view`, with 42 additions and 2 deletions. A contributor could add regression tests for one chart, multiple charts, annotation-only editors, and validation prompt serialization, but should coordinate with the PR author.

**Score:** impact 4, feasibility 4, evidence 4, clarity 5, safety 3, testability 4, opportunity 1 → **3.65 / 5**.

**Why rank third:** the problem is well specified and architecturally important, but the active PR is already aimed at the exact failure. Contribution value is primarily test/review support.

### 4. [#2146 — Add examples in lumen](https://github.com/holoviz/lumen/issues/2146)

| Field | Assessment |
|---|---|
| Category | Documentation / examples |
| GitHub status | Open; created 2026-10-14; label `help wanted`, `lumen-ai`; one comment asks for more ideas |
| Linked PR | None found |
| Reproducibility | Not applicable; the gap is an example-content request |
| Impact | Medium-low individually, but examples can improve onboarding for geospatial and AI users |
| Relevant code/docs | `examples/gallery/`, `examples/ai/`, `docs/examples/`, source extensions such as `lumen/sources/` and geospatial optional extras in `pyproject.toml` |
| Blast radius | Low. Main risk is example reliability and optional dependency burden. |
| Feasibility | Medium-high once the example scope is agreed |
| Testability | Medium. Examples can be syntax-checked and solved through the gallery environment, but live data/services may be required. |
| Remaining opportunity | High; no PR was found, and the issue is explicitly marked help wanted |

The issue is recent and openly invites ideas, but its description only says “examples related to geospatial data” and does not define a dataset, user journey, required provider, or acceptance criteria. The repository already has gallery examples and optional `geo` support (`examples/gallery/`, `docs/examples/gallery/`, `pyproject.toml` extras), so a contributor can propose a concrete example. The missing requirement is maintainer agreement on scope.

**Score:** impact 3, feasibility 4, evidence 2, clarity 2, safety 5, testability 4, opportunity 5 → **3.35 / 5**.

**Why rank fourth:** it offers a clean, unclaimed contribution and is low risk, but it is not yet sufficiently specified for implementation. The next stage should produce a concrete example proposal before coding.

### 5. [#1979 — Write docs on lumen extensions](https://github.com/holoviz/lumen/issues/1979)

| Field | Assessment |
|---|---|
| Category | Documentation / developer onboarding |
| GitHub status | Open; created 2026-07-28; label `documentation`; updated 2026-10-17; two comments ask/offer a generic guide |
| Linked PR | None found |
| Reproducibility | Not applicable |
| Impact | Medium. It could reduce extension-author onboarding cost and make the documented subclassing contracts usable. |
| Relevant code/docs | `docs/extending.md`, `lumen/sources/`, `lumen/ai/agents/`, `lumen/ai/tools/`, `docs/configuration/` |
| Blast radius | Low for runtime code; documentation accuracy depends on several extension contracts. |
| Feasibility | High after scope confirmation |
| Testability | Medium. Documentation builds can validate links and syntax; example projects need separate checks. |
| Remaining opportunity | High; no PR found, but scope remains unresolved |

The issue body points to the Panel Extension Copier Template and `lumen-ourworldindata`. The discussion converges on a generic guide using the Copier template plus a worked example, likely extending `docs/extending.md`, but no maintainer clarification is recorded. The current docs already state that Sources, Agents, Tools, and Analyses are subclassable and recommend the template (`docs/extending.md:1-31`), so the contribution should extend and verify that guide rather than duplicate it.

**Score:** impact 3, feasibility 4, evidence 2, clarity 2, safety 5, testability 4, opportunity 5 → **3.35 / 5**.

**Why rank fifth:** it is a practical unclaimed documentation task, but the scope question must be resolved first. It is a safer alternative to provider or coordinator changes, with lower direct user impact.

## Issues retained for further investigation but not ranked

### [#2184](https://github.com/holoviz/lumen/issues/2184) provider validation caveat

This is ranked despite incomplete live reproduction because the code and issue provide a clear compatibility boundary. The next stage must verify current Anthropic model behavior before implementation planning.

### [#1079 — Loading a big CSV is slow](https://github.com/holoviz/lumen/issues/1079)

Open since 2025-02-27, with `bug`, `lumen-ai`, and `source` labels, but it has no reproduction timing, file-size profile, traceback, or comments. The issue remains valuable, but a contribution should first establish a benchmark through the file-loading and DuckDB paths (`lumen/sources/base.py`, `lumen/sources/duckdb.py`, `lumen/ai/ui.py:552-713`). It is not ranked because impact is plausible but evidence is too weak.

### [#1033 — Document how to configure DuckDB datasource](https://github.com/holoviz/lumen/issues/1033)

This older documentation/source issue is still open and has a concrete production configuration example, but it needs a current gap analysis against `docs/configuration/sources.md` and the current DuckDBSource API. It is a possible follow-up if the requested configuration is still unsupported.

### [#1027 — Document how to use a SQL data source with lumen ai](https://github.com/holoviz/lumen/issues/1027)

The issue has useful negative evidence: its author reported no useful `IntakeSQLSource` docs or tests. However, the current repository now documents SQL/Database startup paths and includes SQL source tests, so the original gap may have narrowed. It requires a current documentation audit before contribution planning.

## Rejected or deprioritized issues

| Issue | Decision | Evidence |
|---|---|---|
| [#2130](https://github.com/holoviz/lumen/issues/2130) | Reject as already addressed by open PR | [#2183](https://github.com/holoviz/lumen/pull/2183) directly implements FunctionTool notebook cells, adds tests, and was updated 2026-10-17. Do not duplicate it. |
| [#2107](https://github.com/holoviz/lumen/issues/2107) | Reject as addressed by open PR | [#2113](https://github.com/holoviz/lumen/pull/2113) adds the requested bearer-token warning. A duplicate [#2112](https://github.com/holoviz/lumen/pull/2112) is closed, not merged. |
| [#2084](https://github.com/holoviz/lumen/issues/2084) | Reject as already addressed by open PR | [#2124](https://github.com/holoviz/lumen/pull/2124) implements column-level SQL lineage in the result inspector. Its checklist says tests are not yet added, so review/testing may remain, but a new implementation would duplicate the PR. |
| [#2066](https://github.com/holoviz/lumen/issues/2066) | Reject as already addressed by open PR | [#2083](https://github.com/holoviz/lumen/pull/2083) directly adds scroll/sizing behavior and is open. |
| [#2057](https://github.com/holoviz/lumen/issues/2057) | Reject as already addressed by open PR | [#2063](https://github.com/holoviz/lumen/pull/2063) implements dynamic hvPlot parameter/documentation handling, though its PR is stale/merge-conflicted. Any work should be review/support on that PR, not duplicate implementation. |
| [#1983](https://github.com/holoviz/lumen/issues/1983) | Reject as already claimed/addressed by an unlinked PR reference; needs maintainer cleanup | The discussion describes a proposed implementation and later claims a PR was created, but gives `[PR Link Here]` rather than a verifiable URL. This is not safe to recommend without resolving the missing PR identity and current state. |
| [#1960](https://github.com/holoviz/lumen/issues/1960) | Exclude from ranked list pending clarification | The issue has no concrete failure or model/format scope. A comment asks which model sizes, quantization formats, and tool-calling behavior are intended; no answer or reproduction is recorded. |
| [#2089](https://github.com/holoviz/lumen/issues/2089) | Exclude from ranked list pending governance/scope review | It is a provider proposal with explicit author disclosure of employment and a revenue-share program. The technical work may be feasible, but contribution suitability depends on maintainer governance and a clear decision that is absent from the inspected issue. |

## Shortlist and next-stage recommendation

The strongest shortlist is:

1. **#2184** — best unclaimed, high-impact compatibility issue; requires provider/API dependency mapping first.
2. **#2134** — best reproduced bug; contribute through tests and review on PR #2139 rather than an independent fix.
3. **#2155** — clear validation bug; contribute regression tests or review support on PR #2156.
4. **#2146** — safe, unclaimed help-wanted example task, but scope must be agreed.
5. **#1979** — useful developer documentation task, also requiring scope confirmation.

Recommend **#2184** for the next stage. It has the strongest combination of user impact, no active implementation PR, and a code-local integration boundary. It is preferable to #2134 and #2155 because those issues already have active implementation PRs; it is preferable to #2146 and #1979 because its expected behavior is more concrete and user-facing. The main uncertainty is external: the reported Anthropic model restrictions, retirement schedule, and Instructor behavior must be verified against current provider documentation or a controlled API test.

Before implementation planning, build an issue-specific dependency map for #2184 covering:

- `Anthropic` and `AnthropicBedrock` request construction and model aliases in `lumen/ai/llm.py`.
- `AnthropicMixin` and provider environment handling in `lumen/ai/services.py`.
- Instructor modes and the exact `tool_choice` behavior used for structured output.
- Adaptive bad-request handling (`ADAPTIVE_KWARGS`) and whether it can safely learn model-specific parameter omissions.
- Existing provider tests in `lumen/tests/ai/test_llm.py`, especially temperature omission and retry tests.
- Current Anthropic model/API documentation and, if credentials are available, a minimal no-data request for each target mode.

Do not begin by editing the shared `Llm` base. First prove the smallest provider-specific change that preserves existing OpenAI, Bedrock, and other provider behavior. A successful implementation plan must include mocked request-shape tests, model alias tests, and a documented live-provider verification limitation if Anthropic credentials are unavailable.
