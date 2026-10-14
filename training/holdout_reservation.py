"""Require reserved final candidates in the registry before actual teacher calls."""
import json
from pathlib import Path

from training.teacher_data import read_rows, sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESERVATION = ROOT/'evaluation/datasets/final_fact96_reservation_v2.json'


def verify_reserved_holdout(registry, reservation=DEFAULT_RESERVATION):
    reservation = Path(reservation)
    if not reservation.exists():
        raise ValueError('holdout reservation missing; generation protection cannot be verified')
    record = json.loads(reservation.read_text(encoding='utf-8'))
    cards_path = ROOT/record['cards_path']
    if sha(cards_path) != record['cards_sha256']:
        raise ValueError('reserved final candidates changed')
    reserved = read_rows(cards_path)
    indexed = {row['sample_id']: row for row in registry}
    if not reserved or any(row['split'] != 'holdout' for row in reserved):
        raise ValueError('invalid holdout reservation')
    if any(indexed.get(row['sample_id']) != row for row in reserved):
        raise ValueError('global registry omits or alters reserved final candidates')
    return dict(reservation_sha256=sha(reservation), cards_sha256=sha(cards_path),
                protected_rows=len(reserved), frozen=bool(record.get('frozen', False)))
