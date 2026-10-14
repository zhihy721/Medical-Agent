"""Lossless JSON state deltas; messages remain separate immutable events."""
from copy import deepcopy
import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                     separators=(',',':')).encode('utf8')).hexdigest()


def difference(old,new,path=()):
    if old==new:return []
    if isinstance(old,dict) and isinstance(new,dict):
        result=[dict(op='delete',path=[*path,k]) for k in old if k not in new]
        for k,v in new.items():
            if k in old:result.extend(difference(old[k],v,(*path,k)))
            else:result.append(dict(op='set',path=[*path,k],value=deepcopy(v)))
        return result
    if isinstance(old,list) and isinstance(new,list) and len(new)>=len(old) and new[:len(old)]==old:
        return [dict(op='append',path=list(path),value=deepcopy(new[len(old):]))]
    return [dict(op='set',path=list(path),value=deepcopy(new))]


def apply(old,operations):
    result=deepcopy(old)
    for change in operations:
        path=change['path']
        if not isinstance(path,list) or not all(isinstance(k,str) for k in path):
            raise ValueError('invalid state delta path')
        if not path:
            if change['op']!='set':raise ValueError('invalid root delta')
            result=deepcopy(change['value']);continue
        parent=result
        for key in path[:-1]:
            if not isinstance(parent,dict) or key not in parent:raise ValueError('state delta parent missing')
            parent=parent[key]
        if not isinstance(parent,dict):raise ValueError('state delta parent not an object')
        key=path[-1]
        if change['op']=='set':parent[key]=deepcopy(change['value'])
        elif change['op']=='delete':
            if key not in parent:raise ValueError('state delta deletion missing')
            del parent[key]
        elif change['op']=='append':
            if not isinstance(parent.get(key),list) or not isinstance(change['value'],list):
                raise ValueError('state delta append not an array')
            parent[key].extend(deepcopy(change['value']))
        else:raise ValueError('unknown state delta operation')
    return result
