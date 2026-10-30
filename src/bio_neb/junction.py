"""Exact scoped review answer decisions, independent of domain consequences."""


def evaluate_scoped_answer(
    requests, receipts, request_event_id, response, *, validate, refusal
):
    """One exact replay/conflict/currentness decision, with domain-owned legality.

    Requests can remain pending independently. A previously answered identity is
    acknowledged only for the identical response; it cannot authorize more work.
    """
    retained = [r for r in receipts if r.request_event_id == request_event_id]
    if len(retained) > 1:
        raise ValueError("Review has duplicate answer identities")
    if retained:
        if retained[0].response != response:
            raise refusal("conflict", request_event_id)
        return None, retained[0]
    pending = [
        r for r in requests if r.request_event_id == request_event_id and r.pending
    ]
    if len(pending) != 1:
        raise refusal("stale", request_event_id)
    validate(pending[0], response)
    return pending[0], None
