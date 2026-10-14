"""Lazy NDJSON transport: emit the urgent reply before resuming tool work."""
import json
from flask import Response


def stream_response(entry, text, coords, final_payload, request_finish=None):
    def generate():
        agent=entry['agent'];stream=None
        with entry['lock']:
            try:
                if hasattr(agent,'run_stream'):
                    stream=agent.run_stream(text,user_coords=coords)
                    for event in stream:
                        if event['kind']=='final':
                            event={**event,**final_payload(agent,event['response'])}
                            if request_finish:request_finish('completed',event)
                        yield json.dumps(event,ensure_ascii=False)+'\n'
                else:
                    response=agent.run(text,user_coords=coords)
                    event={'kind':'final',**final_payload(agent,response)}
                    if request_finish:request_finish('completed',event)
                    yield json.dumps(event,ensure_ascii=False)+'\n'
            except Exception:
                event={'kind':'error','response':'本轮未能完整完成，已保留会话记录，请检查上游连接。若已显示安全提示，请先按提示处理。'}
                if request_finish:request_finish('failed',event)
                yield json.dumps(event,ensure_ascii=False)+'\n'
            finally:
                if stream is not None:stream.close()
                if request_finish:request_finish('interrupted',None)
    return Response(generate(),mimetype='application/x-ndjson',headers={
        'Cache-Control':'no-store','X-Accel-Buffering':'no','X-Content-Type-Options':'nosniff'})
