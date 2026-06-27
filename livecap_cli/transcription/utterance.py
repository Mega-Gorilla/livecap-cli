"""Utterance lifecycle event types for ``StreamTranscriber``.

Public API surface for ``on_utterance_settled`` callback (Issue #332).

A "settled" event signals that one logical utterance has finished
post-processing: filter / coalescer / energy_gate / engine all decided
whether to emit a final result or drop the utterance silently. Consumers
(GUI overlays, OBS adapters, etc.) use it to clear stale interim state
when a drop happens that ``on_result`` would never observe.

See ``StreamTranscriber.set_callbacks`` for delivery ordering details
(callback path vs async generator path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Optional

# Closed set of static drop reasons fired from Tier 1 hook points
# (see Issue #332 rev6). ``engine_error`` is a dynamic family:
# ``f"engine_error:{type(cause).__name__}"`` is constructed at raise time
# and is therefore not represented as a ``Final`` constant.
REASON_EMPTY_AUDIO: Final = "segment:empty_audio"
REASON_ENERGY_GATE: Final = "energy_gate:low_rms"
REASON_FILTER_REJECT: Final = "confidence_filter:reject"
REASON_ENGINE_EMPTY: Final = "engine:empty_text"

#: Type alias for the closed set of static settled reasons (4 values).
#: ``engine_error:<ExceptionType>`` reasons are typed as plain ``str``.
StaticSettledReason = Literal[
    "segment:empty_audio",
    "energy_gate:low_rms",
    "confidence_filter:reject",
    "engine:empty_text",
]


@dataclass(frozen=True)
class UtteranceSettledEvent:
    """Notification that one logical utterance has settled.

    "Settled" means filter / coalescer / energy_gate / engine post-processing
    completed and the outcome is determined: either ``emitted=True`` (a final
    result reached the delivery boundary) or ``emitted=False`` (silently
    dropped, with ``reason`` indicating which guard fired).

    Coalescer merging maps N VAD segments to 1 settled event for the merged
    output. Conversely, ``ResultCoalescer.push`` can flush a pending result
    AND emit a new one in the same call (window-out flush, see
    ``result_coalescer.py:75-98``); in that case **2 separate settled events
    fire**, one per logical utterance.

    Attributes:
        emitted: ``True`` if the producer committed delivery of a final
            result via callback / queue / yield. Consumers may not actually
            observe the result when the delivery channel is a generator and
            the caller breaks; ``emitted=True`` is a producer-side commit,
            not a consumer-receipt guarantee.
        reason: Drop reason when ``emitted=False``. One of ``REASON_*``
            constants exported from this module, or a dynamic
            ``"engine_error:<ExceptionType>"`` string. ``None`` when
            ``emitted=True``.
        source_id: Same identifier as the matching ``TranscriptionResult`` /
            ``InterimResult``. Each ``StreamTranscriber`` instance owns one
            ``source_id`` and emits N settled events (multi-source
            aggregation is out of scope for Issue #332).
        utterance_start_time: Start time of the utterance in stream seconds.
            For coalesced outputs this is the start of the earliest VAD
            segment; for drops, it is the dropped segment's start.
        utterance_end_time: End time of the utterance (same conventions).
    """

    emitted: bool
    reason: Optional[str]
    source_id: str
    utterance_start_time: float
    utterance_end_time: float


__all__ = [
    "UtteranceSettledEvent",
    "StaticSettledReason",
    "REASON_EMPTY_AUDIO",
    "REASON_ENERGY_GATE",
    "REASON_FILTER_REJECT",
    "REASON_ENGINE_EMPTY",
]
