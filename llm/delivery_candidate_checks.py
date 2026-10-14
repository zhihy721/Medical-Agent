"""Read-only delivery alarms for delta benchmarks; no fabricated ledger state.

The historical G6 score stays separate. Check both raw and converted facts,
including transactions discarded by alias deduplication, like the live actor.
"""
import json
from llm.delivery_grounding_guard import check
from llm.fact_representation import normalize_representation
from llm.fact_projection_policy import ALIAS_VERSION
from llm.runtime_delivery_checks import check_discarded_transactions


def error(inp, prediction):
    try:
        raw=json.loads(prediction) if isinstance(prediction,str) else prediction
        check(inp,raw)
        converted,changes=normalize_representation(raw,inp,alias_version=ALIAS_VERSION)
        check(inp,converted)
        check_discarded_transactions(inp,raw,changes)
    except (ValueError,TypeError,KeyError) as exc:
        return str(exc)
    return ''
