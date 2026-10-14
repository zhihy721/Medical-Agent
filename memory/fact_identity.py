"""Finite engineering identity; related medical concepts are never equated."""
VERSION = 'respiratory_concept_identity_v1'


def same_concept(a, b, *, version=None):
    if version not in {None, VERSION}:
        raise ValueError('unsupported concept identity version')
    if a['value'] != b['value']:
        return False
    if a['field'] == b['field']:
        return True
    return (version == VERSION and a['value'] == '呼吸困难'
            and {a['field'], b['field']} == {'symptoms', 'red_flags'})
