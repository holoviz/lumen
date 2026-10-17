# Implementation Plan: GitHub Issue #2184

## Investigation baseline

This plan is based on:

- `DEPENDENCY_MAP.md` in this workspace.
- `REPOSITORY_EXPLORATION.md`.
- `ISSUE_RANKING.md`.
- GitHub issue [holoviz/lumen#2184](https://github.com/holoviz/lumen/issues/2184).
- Closed model-list update PR [#2118](https://github.com/holoviz/lumen/pull/2118).
- Local branch `main` at commit `61c4c7f25c5fe5c2d18d63be0fe2cc0a32748d60`.

No source code, branch, or pull request was changed during this investigation.

`metadata.json` was requested as an initial source but was not present in the workspace. Its absence did not prevent the code and GitHub investigation. Any metadata-dependent conclusion remains unverified.

## 1. Canonical requirements and acceptance criteria

The full requirement traceability is in `DEPENDENCY_MAP.md`, Section 2. The implementation must satisfy these requirements:

- R1: list only verified current Anthropic models.
- R2: omit temperature for models that reject a custom value.
- R3: use a compatible structured-output mode for each target model.
- R4: apply the policy to direct Anthropic and Anthropic Bedrock.
- R5: preserve service-specific model IDs.
- R6: preserve default/edit model roles and settings-dialog behavior.
- R7: preserve unrelated providers.
- R8: keep the change provider-local unless shared changes are proven necessary.
- R9: do not hide unrelated API errors through broad retries.
- R10: test structured calls and native tool loops.
- R11: preserve custom `model_kwargs` configuration.

The key acceptance criteria are:

1. Every listed direct Anthropic model is valid and configured.
2. Every listed Bedrock model uses a valid Bedrock ID.
3. Affected models receive no unsupported `temperature` field.
4. Compatible models retain their configured temperature behavior.
5. The selected Instructor mode does not send an unsupported forced tool choice.
6. Cached clients remain correct for each mode.
7. Existing model selection, cache, stream, and tool-loop behavior remains valid.
8. Existing unrelated provider behavior remains unchanged.
9. The final verification report distinguishes mock-test evidence from live-provider evidence.

Each criterion maps to AC1-AC10 in `DEPENDENCY_MAP.md`.

## 2. Recommended solution

## 2.1 Internet verification result

Official documentation now confirms the most important external facts:

- Current Anthropic model IDs include `claude-haiku-5-5`, `claude-sonnet-5-5`, and `claude-opus-5-5`.
- Anthropic rejects non-default `temperature`, `top_p`, and `top_k` values on Claude 4.7 and later.
- Anthropic supports native JSON structured outputs and strict tool use.
- Instructor's normal Anthropic `Mode.TOOLS` uses a forced tool for a normal structured response model, with special automatic behavior when extended thinking is enabled.
- Bedrock has separate Anthropic model IDs, and newer models may require cross-region inference-profile IDs.

Sources: [Anthropic models](https://platform.claude.com/docs/en/models/overview), [Anthropic deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations), [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [Instructor Anthropic integration](https://github.com/567-labs/instructor/blob/main/docs/integrations/anthropic.md), and [AWS Anthropic Bedrock model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-anthropic.html).

This removes the need to guess about the temperature behavior or the existence of newer model families. It does not yet select the final Lumen implementation. The next required check is a controlled request experiment that compares Instructor's current modes with Anthropic native JSON outputs and Lumen's native tool loop.

Use a provider-local, model-aware compatibility policy in `lumen/ai/llm.py`.

The policy should define, for each supported Anthropic model family or model ID:

- the concrete model ID;
- whether `temperature` must be omitted;
- the structured-output mode;
- the service-specific identifier for Bedrock when applicable.

The policy must be applied after the logical model role is resolved and before the final `messages.create` call. It must not change the common `Llm._client_kwargs` behavior for other providers.

The recommended design is conditional model configuration, not a global default change:

```text
logical role
    -> model_kwargs entry
    -> Anthropic compatibility policy
    -> request kwargs and Instructor mode
    -> direct Anthropic or Bedrock request
```

The exact policy values must be filled only after current Anthropic and AWS documentation, or a controlled API test, confirms them.

### Why this approach is recommended

- It follows the existing provider boundary.
- It limits regression risk for OpenAI, Google, Mistral, and other providers.
- It can preserve old Claude behavior while enabling newer models.
- It can represent different direct and Bedrock IDs.
- It supports deterministic tests of the final request shape.
- It avoids a broad retry mechanism that could repeat provider calls or hide unrelated errors.

### Approaches not recommended initially

#### Change the common temperature default to `None`

This would affect every provider and every model. It could change output behavior for models that accept the current value. The local evidence supports provider-specific handling instead.

#### Change `Anthropic.mode` globally to `ANTHROPIC_JSON`

This may make one model family work but may break older models or native tool interactions. The valid mode must be verified per model family first.

#### Reuse OpenAI adaptive retry logic

The existing adaptive logic is implemented in `OpenAI` (`lumen/ai/llm.py:2600-2633`). It searches an OpenAI error type and learns only known OpenAI parameter fixes. It is not a generic Anthropic solution. Reuse is acceptable only if the error type, parameter identification, bounded retry, and idempotency are proven for Anthropic.

#### Change the UI before the provider contract is known

The settings dialog already reads provider model choices and preserves custom current models. A UI change is not justified until the provider policy and unavailable-model behavior are defined.

## 3. Ordered implementation steps

### Step 1: Confirm external model and mode requirements

**Objective:** Resolve the external facts that control the design.

**Inspect or use:**

- Anthropic current model and deprecation documentation.
- Anthropic structured-output documentation.
- Current Instructor documentation and installed version behavior for Anthropic modes.
- AWS Bedrock model and inference-profile documentation.
- `lumen/ai/llm.py:2810-2825`, `2853-2870`, `2935-2944`.

**Actions:**

1. Confirm the exact replacement or additional direct Anthropic model IDs.
2. Confirm the exact current default model recommendation.
3. Confirm which models reject custom temperature and whether omission is sufficient.
4. Confirm which Instructor mode avoids the rejected forced tool choice.
5. Confirm Bedrock IDs by region or inference profile.
6. Confirm that the chosen mode supports Pydantic structured output.
7. Confirm that the chosen mode does not break Lumen-native tools.

**Requirement links:** R1-R5, R10.

**Verification:** Record provider documentation URLs or controlled request results. If a fact cannot be confirmed, stop before editing the provider list and ask for maintainer direction.

### Step 2: Define the compatibility policy before changing defaults

**Objective:** Choose a data representation that preserves custom provider configuration.

**Expected file:** `lumen/ai/llm.py`.

**Expected symbols:** `Anthropic`, `AnthropicBedrock`, `Anthropic.get_client`, `_get_cached_client`, and request-kwargs handling.

**Actions:**

1. Decide whether policy data belongs in `model_kwargs` entries or in a private provider helper.
2. Keep `model` as the existing public model key.
3. Use clearly named private metadata for compatibility behavior if metadata is stored beside `model`.
4. Ensure routing-only metadata cannot reach the Anthropic SDK. The existing `_get_model_kwargs` stripping behavior must remain intact.
5. Define how a user-supplied `model_kwargs` entry inherits or overrides compatibility defaults.
6. Define behavior for an unknown model. It must not silently receive a compatibility policy intended for a different model unless family matching is explicit and tested.

**Requirement links:** R2, R3, R6, R8, R9, R11.

**Verification:** Add or update pure unit tests for policy resolution before integrating it into `messages.create`.

### Step 3: Add direct Anthropic request-shape regression tests

**Objective:** Make the expected request behavior executable before changing the production defaults.

**Expected file:** `lumen/tests/ai/test_llm.py`.

**Tests to add or extend:**

1. Model configuration exposes every confirmed direct model.
2. The default model appears in `select_models`.
3. A temperature-restricted model sends no `temperature` key.
4. A temperature-compatible model retains the configured temperature.
5. The selected mode is passed to the Instructor wrapper.
6. The generated request does not contain an unsupported forced tool choice.
7. A structured response with no native tools works through the selected mode.
8. A native Lumen tool followed by a structured response still uses the tool loop.
9. Existing prompt cache settings remain in the request.
10. Existing stream handling remains intact.

Use mocks or monkeypatches for the Anthropic client and Instructor wrapper. The test must capture the final call arguments. Do not assert only on class parameters; that would not prove the generated request shape.

**Requirement links:** R2, R3, R7, R9, R10, R11; AC1, AC3-AC6, AC8-AC9.

**Verification command:**

```text
pytest -q --basetemp .pytest-basetemp lumen/tests/ai/test_llm.py -k "anthropic or temperature or model"
```

Run with the repository's supported environment. If the command hangs or cannot create temporary directories, record the environment condition and use a writable workspace basetemp.

### Step 4: Implement model-aware request construction

**Objective:** Apply the confirmed policy to the actual Anthropic call.

**Expected file:** `lumen/ai/llm.py`.

**Expected change area:** `Anthropic.get_client`, `Anthropic._get_cached_client`, and/or a private helper called from `Anthropic.run_client`.

**Actions:**

1. Resolve the concrete model ID from `model_kwargs`.
2. Resolve the compatibility policy for that model.
3. Omit `temperature` only when the policy requires omission.
4. Select the confirmed Instructor mode for the model.
5. Keep model and mode part of the client-cache identity. A cached wrapper for one mode must not be reused for another mode.
6. Keep `max_tokens`, system-message conversion, cache control, streaming, and native tools unchanged unless live testing proves a provider-specific adjustment is required.
7. Preserve explicit user configuration when it is valid.
8. Raise a clear error for an invalid compatibility configuration rather than silently selecting a different model.

**Requirement links:** R2-R5, R7-R11; AC3-AC7.

**Verification:** Run the new mocked request-shape tests after this step. Review the diff for changes outside the Anthropic provider.

### Step 5: Apply and test Bedrock-specific configuration

**Objective:** Support the related Bedrock path without using direct Anthropic IDs.

**Expected files:** `lumen/ai/llm.py`, `lumen/tests/ai/test_llm.py`.

**Actions:**

1. Add only confirmed Bedrock model IDs.
2. Preserve the `AnthropicBedrock` client constructor and AWS credential mapping.
3. Reuse the compatibility policy only where direct and Bedrock behavior is identical.
4. Override policy data where Bedrock model family or mode behavior differs.
5. Test that the direct provider never receives Bedrock IDs and Bedrock never receives direct IDs.

**Requirement links:** R4-R5, R10; AC2, AC7.

**Verification:** Run mocked Bedrock request tests. Run a live minimal Bedrock request only when valid credentials, region, model access, and Anthropic use-case approval are available.

### Step 6: Update model lists and defaults

**Objective:** Make verified models available through configuration and the settings dialog.

**Expected file:** `lumen/ai/llm.py`.

**Actions:**

1. Update `Anthropic.model_kwargs` with the confirmed default and edit models.
2. Update `Anthropic.select_models` with the same compatible direct model aliases or IDs.
3. Update `AnthropicBedrock.model_kwargs` and `select_models` with confirmed Bedrock IDs.
4. Keep the current model visible when a user supplies a custom model.
5. Do not list a model that has not passed request-shape and provider verification.

**Requirement links:** R1, R4-R6; AC1, AC2, AC8.

**Verification commands:**

```text
pytest -q --basetemp .pytest-basetemp lumen/tests/ai/test_llm.py -k "default_model_is_selectable or anthropic"
```

Also exercise the settings dialog test that confirms model selection uses instance-level choices.

### Step 7: Run live provider verification

**Objective:** Verify external compatibility that mocks cannot prove.

**Minimum live cases:**

1. Direct Anthropic default model, ordinary text response.
2. Direct Anthropic target model, structured response.
3. Direct Anthropic target model with one native Lumen tool and a structured final response.
4. Direct Anthropic streaming response.
5. Direct Anthropic prompt cache path if credentials permit.
6. Anthropic Bedrock equivalent cases for each supported Bedrock family.

**Expected evidence:** Capture model ID, mode, request-field policy, success or provider error, and dependency versions. Do not store API keys or sensitive prompts in the repository.

**Requirement links:** R1-R5, R7, R10; AC1-AC7, AC10.

**Stop condition:** If live access is unavailable, do not claim full completion. Mark AC1-AC7 as partially verified and state the exact unverified cases.

### Step 8: Run regression and quality checks

**Objective:** Confirm that the provider fix does not change unrelated behavior.

**Commands:**

```text
pytest -q --basetemp .pytest-basetemp lumen/tests/ai/test_llm.py
pytest --collect-only -q lumen/tests/ai
pytest -q --basetemp .pytest-basetemp lumen/tests/test_config.py lumen/tests/test_pipeline.py
```

Use the repository's full test task when the supported environment is available:

```text
pixi run -e test-312 test-unit
```

Run lint and type checks when the implementation changes typed provider code:

```text
pixi run -e lint lint
pixi run -e lint typecheck
```

The exact environment may differ. Record command, exit status, skips, failures, and environmental errors.

**Requirement links:** R7-R10; AC8-AC10.

## 4. Test-to-requirement matrix

| Verification | Requirements | What it proves | What it does not prove |
|---|---|---|---|
| Model-list/default unit tests | R1, R4-R6 | Configuration is internally consistent. | Provider accepts the model at runtime. |
| Temperature request-shape tests | R2, R7-R9 | The final mock call includes or omits temperature as intended. | Anthropic's live validation of the field. |
| Instructor mode tests | R3, R7, R10 | The selected wrapper and mode are used. | The external API accepts the generated tool choice. |
| Native-tool loop tests | R3, R7, R10 | Lumen tools still execute and the final structured call remains reachable. | All model-specific tool semantics. |
| Bedrock ID tests | R4, R5 | Direct and Bedrock identifiers are not mixed. | AWS access and region availability. |
| Cache and streaming tests | R7, R10 | Provider-specific lifecycle behavior remains intact. | Long-running production sessions. |
| Live direct Anthropic tests | R1-R3, R7, R10 | Current external API accepts the request. | Future API changes. |
| Live Bedrock tests | R4-R5, R10 | Current AWS endpoint accepts the request and ID. | Other regions or accounts. |
| Full AI suite | R7-R11 | No observed regression across repository AI consumers. | External service compatibility without credentials. |

## 5. Risk mitigation and recovery

### Baseline failures

If the baseline AI tests fail before the implementation changes, capture the failing test, dependency versions, and environment. Do not attribute the failure to #2184. Continue only with isolated tests that demonstrate the issue-specific behavior.

### Wider architectural change

If the provider cannot select mode or request kwargs per model without changing `Llm`, stop and document the required shared contract. Obtain maintainer agreement before changing the common abstraction. Add regression tests for every affected provider before implementation.

### External API change during implementation

If Anthropic changes the model or mode requirements, update the requirement table and evidence record first. Do not preserve stale model IDs only because they were in issue #2184.

### Failed implementation

Keep the change provider-local and revert only the issue-specific changes if the design fails. Do not use destructive repository reset commands. Preserve test evidence and the failure request shape for the next attempt.

### No live credentials

Use deterministic mocks for request construction. Mark live acceptance criteria as unverified. Request maintainer-provided verification or a controlled test account before claiming the issue is fully resolved.

## 6. Unresolved decisions requiring clarification

The implementation must not choose these values silently:

1. The exact replacement default model.
2. The exact edit model.
3. The exact direct and Bedrock model identifiers.
4. The exact Instructor mode for each model family.
5. Whether model compatibility is represented in public `model_kwargs` or private provider metadata.
6. Whether users may override the compatibility policy.
7. Whether an unavailable default should fail, fall back, or prompt the user.

The first four decisions require provider evidence. The last three require maintainer or API-contract judgment if the existing behavior does not answer them.

## 7. Completion and review criteria

Implementation is complete only when all applicable conditions are true:

1. R1-R11 are satisfied or explicitly marked not applicable with evidence.
2. AC1-AC9 have passing deterministic verification.
3. Live provider results verify AC1-AC7, or AC10 clearly documents the unavailable live verification.
4. Direct Anthropic and Bedrock paths have separate model-ID coverage.
5. Structured response, native tools, streaming, caching, and settings selection are covered.
6. Existing core and AI regression checks pass in the supported environment, or failures are classified with evidence.
7. No unrelated files or provider behavior changed.
8. The final diff contains no credentials, temporary files, generated artifacts, or broad refactors.
9. The final review checks the common `Llm` contract, the provider cache key, model selection behavior, and error handling.
10. Any remaining limitation is stated in the handoff and is not presented as verified behavior.

## Handoff summary

First confirm the current Anthropic and Bedrock model IDs and the valid Instructor mode for each target model. Then add a provider-local compatibility policy. Add request-shape tests before changing defaults. Implement model-aware omission of unsupported temperature and model-aware structured-output mode selection. Apply the policy to both direct Anthropic and `AnthropicBedrock`. Update defaults only with verified models. Test native tools, structured output, streaming, caching, model selection, and old-model compatibility. Run the targeted and full AI checks. Use live provider requests when credentials permit. If live verification is unavailable, report the exact remaining limitation instead of claiming complete resolution.
