"""Owned loopback backend. Establish private paths before importing app.py."""
import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import uuid

MODES=('rule','sft','g6-rule','g6-plan','g6-review','full888-rule','full888-plan','full888-review',
       'full888-safe-rule','full888-safe-plan','full888-safe-review','qwen-prototype','qwen-grounded')

# Keep the lease referenced until actual process exit, including graph workers
# draining after the HTTP loop returns. No early close or atexit unlock.
_workspace_lease = None


def configure(root, data, mode):
    root=Path(root).resolve();data=Path(data).resolve()
    if mode not in MODES or not (root/'app.py').is_file():
        raise ValueError('invalid workspace or mode')
    data.mkdir(parents=True,exist_ok=True)
    os.environ['MEDICAL_AGENT_CONFIG_PATH']=str(data/'config.env')
    sys.path.insert(0,str(root));os.chdir(root)
    from agent.runtime_profiles import DESKTOP_STRATEGIES, delivery_settings,full888_settings,full888_safe_settings
    from config_manager import DEFAULT_CONFIG,read_env_file,_serialize_config
    config=dict(DEFAULT_CONFIG)
    if (data/'config.env').exists():config.update(read_env_file(data/'config.env'))
    # Do not copy project API keys or silently select a cloud model.
    config.update(LLM_PROVIDER='mock',DATA_DIR=str(data/'data'),LOG_DIR=str(data/'logs'))
    # Desktop owns these settings; inherited AGENT_* cannot redirect a role.
    for key in list(os.environ):
        if key.startswith('AGENT_'):os.environ.pop(key)
    for key in list(config):
        if key.startswith('AGENT_'):config.pop(key)
    if mode=='qwen-grounded':
        from agent.runtime_profiles import qwen_grounded_settings
        config.update(qwen_grounded_settings(root))
    elif mode=='qwen-prototype':
        from agent.runtime_profiles import qwen_prototype_settings
        config.update(qwen_prototype_settings(root))
    elif mode=='sft':
        config.update(AGENT_RUNTIME='langgraph',AGENT_PLANNER_MODE='llm',AGENT_REVIEWER_MODE='llm',
            AGENT_POLICY_PROMPT_VERSION='v3',AGENT_EXTRACTOR_MODE='semantic',AGENT_INTENT_MODE='model',
            AGENT_INTENT_PROMPT_VERSION='v2',AGENT_EXTRACTOR_BOUNDARY_VERSION='v2',
            AGENT_EXTRACTOR_RECOVERY_MODE='guarded_v3',AGENT_RESPONDER_MODE='model',
            AGENT_RESPONDER_BUDGET_MODE='serving_tokenizer',
            AGENT_MODEL_ROLES_CONFIG=str(root/'llm/semantic_roles.formal_candidate_v1.yaml'))
    elif mode.startswith('full888-safe-'):
        strategy={'full888-safe-rule':'rule_rule','full888-safe-plan':'llm_rule','full888-safe-review':'llm_llm'}[mode]
        config.update(full888_safe_settings(root,strategy))
    elif mode.startswith('full888-'):
        strategy={'full888-rule':'rule_rule','full888-plan':'llm_rule','full888-review':'llm_llm'}[mode]
        config.update(full888_settings(root,strategy))
    elif mode in DESKTOP_STRATEGIES:
        config.update(delivery_settings(root,DESKTOP_STRATEGIES[mode]))
    else:config.update(AGENT_RUNTIME='langgraph')
    (data/'config.env').write_text(_serialize_config(config),encoding='utf8')
    secret=data/'flask-secret.txt'
    if not secret.exists():secret.write_text(secrets.token_urlsafe(48),encoding='ascii')
    os.environ['FLASK_SECRET_KEY']=secret.read_text(encoding='ascii').strip()
    return root,data


def install(app_module, data, mode, token, ui):
    from flask import abort,jsonify,request,send_from_directory,session
    from memory.file_store import read_json_file,write_json_file
    app=app_module.app
    app.config.update(SESSION_COOKIE_NAME='medical_desktop',SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE='Strict')
    lock=threading.RLock();index_path=data/'session-index.json'

    @app.before_request
    def desktop_auth():
        # A random per-launch capability is injected by Electron, never handed
        # to renderer JS, stored in ready JSON or placed in a URL.
        if not hmac.compare_digest(request.headers.get('X-Medical-Desktop',''),token):abort(403)
        origin=request.headers.get('Origin')
        if origin and origin!=request.host_url.rstrip('/'):abort(403)
        if request.host.split(':')[0]!='127.0.0.1':abort(403)
        if request.path!='/desktop/health':session.permanent=True
        # Desktop config editing must not reroute shared startup directories.
        if request.path.startswith('/api/config'):abort(404)

    @app.after_request
    def desktop_headers(response):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; media-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        if request.path in {'/chat','/status','/reset','/desktop/api/open'} and response.status_code==200 and session.get('session_id'):
            if hasattr(app_module.session_store, 'list_sessions'):
                return response
            with lock:
                sid=session['session_id'];history=app_module.session_store.get_history(sid,[])
                title=next((str(msg[1])[:32] for msg in history if msg[0]=='user'),'新问诊')
                index=read_json_file(index_path,default={})
                index[sid]={'id':sid,'user_id':session['user_id'],'title':title}
                write_json_file(index_path,index)
        return response

    @app.get('/desktop/')
    def desktop_home():return send_from_directory(ui,'index.html')

    @app.get('/desktop/assets/<path:name>')
    def desktop_asset(name):return send_from_directory(ui,name)

    @app.get('/desktop/health')
    def health():
        if mode in {'qwen-prototype','qwen-grounded'}:
            roles={'extractor':'delivery-4b-full888','intent':'qwen3.5-4b joint','planner':'qwen3.5-4b (rule fallback)','reviewer':'qwen3.5-4b before tools (rule fallback)','responder':'qwen3.5-4b/template'}
        elif mode.startswith(('g6-','full888-')):
            roles={'extractor':'delivery-4b-full888' if mode.startswith('full888-') else 'g6-4b-mixed',
                   'planner':'rule' if mode.endswith('-rule') else 'qwen3.5-4b',
                   'reviewer':'qwen3.5-4b' if mode.endswith('-review') else 'rule',
                   'intent':'qwen3.5-4b','responder':'qwen3.5-4b'}
        elif mode=='sft':roles={'extractor':'extractor-formal-epoch2','other_roles':'qwen3.5-4b'}
        else:roles={'extractor':'rule/mock','planner':'rule','reviewer':'rule','responder':'template'}
        return jsonify(ready=True,mode=mode,scope='engineering_demo',
            roles=roles)

    @app.get('/desktop/api/sessions')
    def sessions():
        with lock:index=read_json_file(index_path,default={})
        uid=session.get('user_id')
        store = getattr(app_module, 'session_store', None)
        if hasattr(store, 'list_sessions') and uid:
            # Import only this user's old indexed JSON tails; preserve the old
            # index for rollback, never invent missing historic conversation.
            for record in index.values():
                if record['user_id']==uid:
                    store.register_session(record['id'],uid)
            return jsonify(current=session.get('session_id'),items=[
                {'id':r['session_id'],'title':r['title'] or '新问诊'} for r in store.list_sessions(uid)])
        return jsonify(current=session.get('session_id'),items=[{'id':r['id'],'title':r['title']} for r in index.values() if r['user_id']==uid])

    @app.get('/desktop/api/facts')
    def facts():
        from memory.fact_ledger import statuses
        entry=app_module._get_agent()
        with entry['lock']:state=entry['agent'].memory.get_case_state()
        ledger=state.get('fact_ledger')
        if not ledger:return jsonify(desktop_facts=[])
        live=statuses(ledger);by_id={r['id']:r for r in ledger['records']}
        status_labels={'active':'有效','past':'过去记录','disputed':'有冲突，待澄清','superseded':'已更正','resolved':'已缓解/结束'}
        subject_labels={'patient':'本人','other':'他人','unknown':'主体待确认'}
        time_labels={'current':'当前','historical':'过去','hypothetical':'假设','future':'未来','unknown':'时间待确认'}
        return jsonify(desktop_facts=[{**{k:r[k] for k in ('id','field','value','polarity','evidence')},
            'about_label':by_id.get(r.get('about'),{}).get('value'),
            'status':live[r['id']],'status_label':status_labels.get(live[r['id']],live[r['id']]),
            'subject_label':subject_labels.get(r['subject'],r['subject']),
            'temporality_label':time_labels.get(r['temporality'],r['temporality'])} for r in ledger['records']])

    @app.post('/desktop/api/open')
    def open_session():
        sid=(request.get_json(silent=True) or {}).get('id','')
        try:uuid.UUID(sid)
        except (ValueError,TypeError,AttributeError):abort(400)
        with lock:index=read_json_file(index_path,default={})
        store = getattr(app_module, 'session_store', None)
        if hasattr(store, 'list_sessions'):
            uid=session.get('user_id')
            if uid and sid in index and index[sid]['user_id']==uid:
                store.register_session(sid,uid)
            if not uid or sid not in {r['session_id'] for r in store.list_sessions(uid)}:abort(404)
        elif sid not in index or index[sid]['user_id']!=session.get('user_id'):abort(404)
        session['session_id']=sid
        return app_module.status()

    return app


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--ready-file',type=Path,required=True);p.add_argument('--mode',choices=MODES,default='rule')
    p.add_argument('--ui',type=Path,required=True)
    return p


def main():
    global _workspace_lease
    a=parser().parse_args()
    token=json.loads(sys.stdin.readline()).get('token','')
    if not isinstance(token,str) or len(token)<32:raise ValueError('invalid capability')
    sys.path.insert(0,str(a.project.resolve()))
    from desktop.ownership import WorkspaceLease,recover_owned_pending
    # Acquire BEFORE rewriting config or opening the shared private archive.
    # An OS-released lifetime lock proves no old desktop backend still owns it.
    if _workspace_lease is not None:raise RuntimeError('backend main already started in this process')
    _workspace_lease=WorkspaceLease(a.data)
    root,data=configure(a.project,a.data,a.mode)
    import app as application
    recovered=recover_owned_pending(application.session_store,_workspace_lease)
    (data/'backend-recovery.json').write_text(json.dumps(dict(recovered=recovered,automatic_replay=False)),encoding='utf8')
    from werkzeug.serving import make_server
    server=make_server('127.0.0.1',0,install(application,data,a.mode,token,a.ui.resolve()),threaded=True)
    a.ready_file.write_text(json.dumps({'port':server.server_port,'pid':os.getpid(),
        'parent_pid':os.getppid(),'mode':a.mode}),encoding='utf8')
    # Normal close and parent death both close stdin. Shut down our own MCP
    # children gracefully rather than relying only on a Windows process kill.
    def watch_parent():
        sys.stdin.readline()
        server.shutdown()
    threading.Thread(target=watch_parent,daemon=True).start()
    try:server.serve_forever()
    finally:server.server_close();application.shutdown_mcp_servers()


if __name__=='__main__':main()
