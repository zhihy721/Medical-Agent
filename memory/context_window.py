"""Bounded model context, independent from the append-only conversation archive."""


def build_context(summary, history, *, token_budget=2048, token_counter=None):
    """Prefer recent complete messages. Count bytes conservatively by default.

    This is the content budget only; model callers must reserve system/template
    and generation tokens separately. A supplied tokenizer can count precisely.
    State facts used by the semantic extractor are separately validated, never
    silently truncated here. The latest message must fit or fail explicitly.
    """
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError('positive context budget required')
    count = token_counter or (lambda text: len(text.encode('utf8')))
    recent = list(history)
    selected = []
    for role, text in reversed(recent):
        candidate = [(role, text), *selected]
        rendered = 'recent_history:\n' + '\n'.join(f'{r}: {t}' for r, t in candidate)
        if count(rendered) > token_budget:
            if not selected:
                raise ValueError('latest message exceeds model context budget; archive retained')
            break
        selected = candidate
    rendered = ('recent_history:\n' + '\n'.join(f'{r}: {t}' for r, t in selected)) if selected else ''
    full = 'summary: ' + summary + ('\n' + rendered if rendered else '')
    if count(full) <= token_budget:
        rendered = full
        summary_included = True
    else:
        summary_included = False
    return dict(text=rendered, estimated_tokens=count(rendered), token_budget=token_budget,
                summary_included=summary_included, retained_messages=len(selected),
                omitted_messages=len(recent) - len(selected), archive_modified=False)
