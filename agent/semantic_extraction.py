"""Opt-in extraction adapter; raw model calls and state transitions stay auditable."""
from dataclasses import asdict
import json
from jsonschema import Draft202012Validator

from llm.fact_contract_runtime import DELTA_SCHEMA, decoding_schema, messages
from llm.deferred_scope import adjust_delta, VERSION as SCOPE_VERSION


class SemanticExtractor:
    def __init__(self, client, *, structured=True, intent=None, boundary_version='v1', recovery_mode='off', repair_client=None, prompt_version='locked', contract_version=None, representation_mode='finite', delivery_guard=None):
        from llm.fact_contract_runtime import VERSION, V2
        self.contract_version = contract_version or VERSION
        if self.contract_version not in {VERSION, V2}:
            raise ValueError('unsupported fact contract')
        if representation_mode not in {'finite','finite_v4','finite_v5','off'}:
            raise ValueError('unsupported representation mode')
        self.representation_mode = representation_mode
        if boundary_version not in {'v1', 'v2', 'v3', 'v4'} or recovery_mode not in {'off', 'checked_v16', 'guarded_v1', 'guarded_v2', 'guarded_v3', 'guarded_v4', 'guarded_v5', 'guarded_v6', 'guarded_v7', 'guarded_v8', 'guarded_v9', 'guarded_v10', 'guarded_v11', 'guarded_v12','guarded_v13'}:
            raise ValueError('invalid extraction boundary/recovery mode')
        if prompt_version not in {'locked','g6_v7','coverage_v1','coverage_v2','coverage_v3','coverage_v4','coverage_v5','coverage_v6','coverage_v7','coverage_v8'}:
            raise ValueError('invalid semantic prompt version')
        if ((representation_mode == 'finite_v4') != (prompt_version == 'coverage_v7')
                or (representation_mode=='finite_v5') != (prompt_version in {'coverage_v8','g6_v7'})
                or representation_mode in {'finite_v4','finite_v5'} and self.contract_version != V2
                or recovery_mode=='guarded_v13' and representation_mode!='finite_v5'):
            raise ValueError('finite_v4 requires V2 contract and coverage_v7 prompt together')
        if (prompt_version=='g6_v7' or recovery_mode=='checked_v16') and not (
                prompt_version=='g6_v7' and recovery_mode=='checked_v16' and boundary_version=='v1'
                and representation_mode=='finite_v5' and self.contract_version==V2):
            raise ValueError('G6 requires V2, V7 plain serialization, finite_v5 and checked_v16 together')
        from llm.fact_lexical_policy import VERSION as POLICY_VERSION
        from llm.fact_projection_policy import VERSION as V5
        self.representation_policy = (V5 if representation_mode=='finite_v5' else
                                      POLICY_VERSION if representation_mode == 'finite_v4' else None)
        self.client = client
        self.structured = structured
        self.intent = intent
        self.boundary_version = boundary_version
        self.recovery_mode = recovery_mode
        self.repair_client = repair_client
        self.prompt_version = prompt_version
        if delivery_guard not in {None, 'delivery_grounding_guard_v1'}:
            raise ValueError('unsupported delivery guard')
        self.delivery_guard = delivery_guard

    def run(self, memory, text):
        inp, result, attempts = None, None, []
        from llm.runtime_extraction_guard import classify, check_empty, check_explicit_conflicts, check_emphasis_omission, check_attribute_negation, check_denial_and_duration, check_supported_rewrite, check_attribute_transactions, check_semantic_types, check_problem_corrections, check_conditional_coverage, repair_messages
        from llm.fact_lexical_policy import PAIN_ALIASES
        classification = classify(text, additional_terms=tuple(PAIN_ALIASES)+('心慌',)
            if self.representation_policy and self.prompt_version!='g6_v7' else ())
        if memory.case_state['fact_ledger'].get('representation_policy_version') != self.representation_policy:
            raise ValueError('extractor and session representation policy differ; start a new session')
        try:
            inp = memory.fact_input(text)
            if inp['contract_version'] != self.contract_version:
                raise ValueError('extractor and session contract differ; start a new session')
            serializer = messages
            if self.boundary_version == 'v2':
                from llm.input_boundary import messages as serializer
            elif self.boundary_version == 'v3':
                from llm.input_boundary import canonical_messages as serializer
            elif self.boundary_version == 'v4':
                from llm.input_boundary import schema_order_messages as serializer
            original = serializer(inp)
            from llm.semantic_prompt import augment
            original = augment(original,'coverage_v7' if self.prompt_version=='g6_v7' else self.prompt_version)
            for attempt in range(1 if self.recovery_mode in {'off','checked_v16'} else 2):
                caller = self.client if attempt == 0 or self.repair_client is None else self.repair_client
                result = caller.call(original if attempt == 0 else repair_messages(original,
                                          version=self.recovery_mode.removeprefix('guarded_')), require_json=True,
                                          **({'json_schema': decoding_schema(inp)} if self.structured else {}))
                entry = {'attempt': attempt + 1, 'model': asdict(result), 'raw_delta': None, 'status': 'error'}
                attempts.append(entry)
                try:
                    if not result.ok:
                        raise ValueError('provider:' + result.status)
                    raw_delta = json.loads(result.content)
                    entry['raw_delta'] = raw_delta
                    errors = list(Draft202012Validator(DELTA_SCHEMA).iter_errors(raw_delta))
                    if errors:
                        raise ValueError('fact_schema: ' + errors[0].message)
                    if self.delivery_guard:
                        from llm.delivery_grounding_guard import check as check_grounding, VERSION as GROUNDING_VERSION
                        check_grounding(inp, raw_delta)
                        entry['grounding_guard_version'] = GROUNDING_VERSION
                    delta = raw_delta
                    representation_changes = []
                    from llm.fact_contract_runtime import V2
                    if self.representation_policy:
                        from llm.fact_lexical_policy import check_heart_rewrite, check_degree_rewrite
                        check_heart_rewrite(raw_delta)
                        check_degree_rewrite(raw_delta)
                    if self.contract_version == V2 and self.representation_mode in {'finite','finite_v4','finite_v5'}:
                        from llm.fact_representation import normalize_representation
                        from llm.fact_lexical_policy import ALIAS_VERSION
                        from llm.fact_projection_policy import ALIAS_VERSION as V5_ALIASES
                        delta, representation_changes = normalize_representation(raw_delta, inp,
                            alias_version=V5_ALIASES if self.representation_mode=='finite_v5' else
                                ALIAS_VERSION if self.representation_policy else None)
                        # Audit metadata must retain the same shape after JSONL persistence.
                        # This converts tuples in alignment records, never facts or labels.
                        representation_changes = json.loads(json.dumps(representation_changes,ensure_ascii=False))
                        entry.update(converted_delta=delta, representation_changes=representation_changes)
                    if self.delivery_guard:
                        # Same-concept field alignment must not bypass grounding.
                        check_grounding(inp,delta)
                    if self.representation_policy:
                        from llm.fact_lexical_policy import check_pain_negation
                        check_pain_negation(delta)
                    if self.recovery_mode=='checked_v16':
                        from llm.runtime_delivery_checks import check_discarded_transactions, VERSION as DELIVERY_GUARD
                        check_discarded_transactions(inp,raw_delta,representation_changes)
                        entry['delivery_guard_version'] = DELIVERY_GUARD
                        from llm.linked_fact_guard_v16 import check
                        check(inp,delta,classification)
                    elif self.recovery_mode=='guarded_v13':
                        from llm.fact_transaction_guard import check_transactions
                        check_transactions(inp,delta,classification)
                    elif self.recovery_mode == 'guarded_v10':
                        check_semantic_types(inp, delta, classification, modern=True)
                    elif self.recovery_mode == 'guarded_v11':
                        check_problem_corrections(inp, delta, classification)
                    elif self.recovery_mode == 'guarded_v12':
                        check_conditional_coverage(inp, delta, classification)
                    if self.recovery_mode == 'guarded_v1':
                        check_empty(raw_delta, classification)
                    elif self.recovery_mode == 'guarded_v2':
                        check_explicit_conflicts(raw_delta, classification)
                    elif self.recovery_mode == 'guarded_v3':
                        check_emphasis_omission(inp, raw_delta, classification)
                    elif self.recovery_mode == 'guarded_v4':
                        check_attribute_negation(inp, raw_delta, classification)
                    elif self.recovery_mode == 'guarded_v5':
                        check_denial_and_duration(inp, raw_delta, classification)
                    elif self.recovery_mode == 'guarded_v6':
                        check_supported_rewrite(inp, raw_delta, classification)
                    elif self.recovery_mode in {'guarded_v7','guarded_v8'}:
                        check_attribute_transactions(inp, raw_delta, classification)
                    elif self.recovery_mode=='guarded_v9':
                        check_semantic_types(inp, raw_delta, classification)
                    effective_delta, changes = adjust_delta(inp, delta)
                    # apply_fact_delta validates a copy before committing; a
                    # rejected attempt cannot change facts or advance the turn.
                    entry['status'] = 'ok'
                    memory.apply_fact_delta(inp, effective_delta, model_result=asdict(result),
                        extraction_policy_trace={'policy_version': SCOPE_VERSION,
                            'extractor_raw_delta': raw_delta, 'scope_adjustments': changes,
                            'representation_changes': representation_changes,
                            'representation_policy_version': self.representation_policy,
                            'boundary_version': self.boundary_version, 'recovery_mode': self.recovery_mode, 'prompt_version':self.prompt_version,
                            'input_classification': classification, 'attempts': attempts})
                    return {'status': 'ok', 'input': inp, 'model': asdict(result),
                            'raw_delta': raw_delta, 'converted_delta': delta, 'representation_changes': representation_changes,
                            'representation_policy_version': self.representation_policy,
                            'scope_adjustments': changes, 'scope_policy_version': SCOPE_VERSION,
                            'boundary_version': self.boundary_version, 'recovery_mode': self.recovery_mode, 'prompt_version':self.prompt_version,
                            'input_classification': classification, 'attempts': attempts}
                except (ValueError, TypeError, KeyError) as exc:
                    entry.update(status='error', error=str(exc))
                    if attempt or self.recovery_mode in {'off','checked_v16'} or not result.ok:
                        raise
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            # Failure turns retain every raw request, including failed first
            # attempts; the caller sees an explicit error instead of empty facts.
            memory.record_fact_failure(text, inp, error, model_result=asdict(result) if result else None,
                extraction_policy_trace={'boundary_version': self.boundary_version, 'recovery_mode': self.recovery_mode, 'prompt_version':self.prompt_version,
                                         'input_classification': classification, 'attempts': attempts})
            return {'status': 'error', 'error': error, 'input': inp,
                    'model': asdict(result) if result else None, 'attempts': attempts,
                    'input_classification': classification, 'recovery_mode': self.recovery_mode}
