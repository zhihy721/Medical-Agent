import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest

from flask import Flask
from desktop.backend import install
from memory.memory import ConversationMemory
from training.test_fact_contract import fact


class DesktopBackendTests(unittest.TestCase):
    def test_cli_accepts_same_modes_as_backend_configuration(self):
        from desktop.backend import parser,MODES,configure
        from unittest.mock import patch
        from config_manager import read_env_file
        for mode in MODES:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
                args=parser().parse_args(['--project',str(Path.cwd()),'--data',tmp,'--ready-file',str(Path(tmp)/'ready.json'),
                    '--ui',str(Path.cwd()/'desktop/ui'),'--mode',mode])
                configure(args.project,args.data,args.mode)
                if mode.startswith('full888-safe-'):
                    self.assertEqual(read_env_file(Path(tmp)/'config.env')['AGENT_DELIVERY_INPUT_SAFETY'],'literal_breathing_safety_v2')
                if mode=='qwen-grounded':
                    config=read_env_file(Path(tmp)/'config.env')
                    self.assertEqual(config['AGENT_QWEN_PLAN_CONTRACT'],'basis_v1')
                    self.assertEqual(config['AGENT_QWEN_REVIEW_REASON'],'visible')
                    self.assertEqual(config['AGENT_PLANNER_MODE'],'llm')
                    self.assertEqual(config['AGENT_REVIEWER_MODE'],'llm')
    def test_config_path_is_selected_before_import_and_web_default_is_preserved(self):
        root=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            custom=Path(tmp)/'private.env';custom.write_text('LLM_PROVIDER=mock\nDATA_DIR=private-data\n',encoding='utf8')
            code="import config_manager,json;print(json.dumps({'path':str(config_manager.CONFIG_PATH),'data':config_manager.read_config()['DATA_DIR']}))"
            env=dict(os.environ,MEDICAL_AGENT_CONFIG_PATH=str(custom))
            result=subprocess.run([os.sys.executable,'-X','utf8','-c',code],cwd=root,env=env,capture_output=True,text=True,check=True)
            selected=json.loads(result.stdout)
            self.assertEqual(Path(selected['path']),custom);self.assertEqual(selected['data'],'private-data')
            env.pop('MEDICAL_AGENT_CONFIG_PATH')
            result=subprocess.run([os.sys.executable,'-X','utf8','-c',code],cwd=root,env=env,capture_output=True,text=True,check=True)
            self.assertEqual(Path(json.loads(result.stdout)['path']),root/'config.env')

    def test_fact_view_exposes_authoritative_correction_negation_and_binding(self):
        m=ConversationMemory();m.enable_fact_mode()
        text='头疼一周，没有胸痛。'
        m.apply_fact_delta(m.fact_input(text),{'facts':[fact('头疼','头疼'),
            fact('一周','一周',id='f2',field='duration',about='f1'),
            fact('胸痛','没有胸痛',id='f3',polarity='negated')]})
        m.apply_fact_delta(m.fact_input('刚才说错了，是两天。'),{'facts':[
            fact('两天','两天',id='f1',field='duration',about='t1:f1',operation='correct',replaces='t1:f2')]})
        application=Flask(__name__);application.secret_key='test-only'
        module=SimpleNamespace(app=application,_get_agent=lambda:{'agent':SimpleNamespace(memory=m),'lock':threading.Lock()})
        with tempfile.TemporaryDirectory() as tmp:
            install(module,Path(tmp),'sft','x'*64,Path('desktop/ui'))
            with application.test_client() as c:
                self.assertEqual(c.get('/desktop/api/facts').status_code,403)
                reply=c.get('/desktop/api/facts',headers={'X-Medical-Desktop':'x'*64,'Host':'127.0.0.1'})
                self.assertEqual(reply.status_code,200)
                rows=reply.get_json()['desktop_facts']
                old=next(r for r in rows if r['value']=='一周');new=next(r for r in rows if r['value']=='两天')
                self.assertEqual(old['status'],'superseded');self.assertEqual(new['status'],'active')
                self.assertEqual(new['about_label'],'头痛')
                self.assertEqual(next(r for r in rows if r['value']=='胸痛')['polarity'],'negated')
                self.assertIn("script-src 'self'",reply.headers['Content-Security-Policy'])
                self.assertEqual(c.get('/api/config/status',headers={'X-Medical-Desktop':'x'*64,'Host':'127.0.0.1'}).status_code,404)


if __name__=='__main__':unittest.main()
