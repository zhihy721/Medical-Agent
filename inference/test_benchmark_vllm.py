import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from datetime import datetime

from inference.benchmark_vllm import consume_stream, percentile, read_workload, summarize, run
from inference.build_extractor_workload import build


class StreamingBenchmarkTests(unittest.TestCase):
    def test_stream_ignores_role_and_reasoning_for_ttft(self):
        clock=iter([1.3,1.7,2.1]).__next__
        events=[
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"internal"}}]}',
            'data: {"choices":[{"delta":{"content":"你好"}}]}',
            'data: {"choices":[{"delta":{"content":"。"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}',
            'data: [DONE]',
        ]
        result=consume_stream(events,1.0,clock)
        self.assertAlmostEqual(result['ttft_seconds'],.3)
        self.assertAlmostEqual(result['e2e_seconds'],1.1)
        self.assertAlmostEqual(result['tpot_seconds'],.4)
        self.assertEqual(result['completion_tokens'],3)
        self.assertEqual(result['finish_reason'],'length')

    def test_missing_usage_and_incomplete_stream_rejected(self):
        with self.assertRaisesRegex(ValueError,'completion_tokens'):
            consume_stream(['data: {"choices":[{"delta":{"content":"a"}}]}',
                            'data: [DONE]'],1.0,lambda:2.0)
        with self.assertRaisesRegex(ValueError,'without .DONE.'):
            consume_stream(['data: {"choices":[{"delta":{"content":"a"}}]}'],
                           1.0,lambda:2.0)

    def test_summary_excludes_failed_and_tpot_single_token(self):
        rows=[dict(status='ok',ttft_seconds=.1,e2e_seconds=.5,tpot_seconds=None,
                   completion_tokens=1),
              dict(status='ok',ttft_seconds=.2,e2e_seconds=.8,tpot_seconds=.15,
                   completion_tokens=5),
              dict(status='failed',e2e_seconds=1.0)]
        result=summarize(rows,2.0)
        self.assertEqual((result['succeeded'],result['failed'],result['tpot_eligible']),(2,1,1))
        self.assertEqual(result['aggregate_output_tokens_per_second'],3.0)
        self.assertEqual(result['finish_reasons'],{'unknown':2})
        self.assertEqual(percentile([1,2,3,4],.5),2.5)

    def test_workload_needs_unique_ids_and_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'workload.jsonl'
            path.write_text('{"sample_id":"a","messages":[{"role":"user","content":"你好"}]}\n',encoding='utf-8')
            self.assertEqual(len(read_workload(path)),1)
            path.write_text(path.read_text(encoding='utf-8')*2,encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'duplicate'):
                read_workload(path)

    def test_exported_workload_excludes_assistant_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'workload.jsonl'
            from llm.fact_contract import build_input,messages
            from training.test_fact_contract import fact
            source=Path(temp)/'source.jsonl'
            rows=[]
            for i,value in enumerate(['咳嗽','头痛','牙痛']):
                inp=build_input(value+'。');target={'facts':[fact(value)]}
                rows.append(dict(sample_id=f'fixture-{i}',source_family_id=f'family-{i}',split='train',review_status='reviewed',input=inp,target=target,messages=messages(inp,target)))
            source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
            report=build(source,path)
            self.assertEqual(report['records'],3)
            workload=read_workload(path)
            self.assertTrue(all(r['messages'][-1]['role']=='user' for r in workload))
            self.assertTrue(report['training_answers_excluded'])

    def test_run_records_utc_bounds_for_resource_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            workload = Path(temp)/'workload.jsonl'
            workload.write_text(json.dumps({'sample_id':'one',
                'messages':[{'role':'user','content':'你好'}]},ensure_ascii=False)+'\n',encoding='utf-8')
            reply = {'sample_id':'one','status':'ok','ttft_seconds':.1,'e2e_seconds':.2,
                     'tpot_seconds':.01,'completion_tokens':2,'finish_reason':'stop'}
            with patch('inference.benchmark_vllm.request_one', return_value=reply):
                report = run('http://127.0.0.1:8000/v1/chat/completions','test',workload,
                             Path(temp)/'out',concurrencies=(1,),rounds=1,warmups=0)
            batch = report['results'][0]
            bounds = [datetime.fromisoformat(report['started_at_utc']),
                      datetime.fromisoformat(batch['started_at_utc']),
                      datetime.fromisoformat(batch['finished_at_utc']),
                      datetime.fromisoformat(report['finished_at_utc'])]
            self.assertTrue(all(value.tzinfo is not None for value in bounds))
            self.assertEqual(bounds, sorted(bounds))


if __name__=='__main__':
    unittest.main()
