"""Portable CPU regressions: no reports, API keys, models or GPU required."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from inference.server_timing_metrics import measure,one,counters
from inference import role_server_trace as trace

def fixture(n=6):
    events=[dict(kind='http_received',request_id='rsm-test',monotonic=10.,wall=100.),
        dict(kind='engine_finished',request_id='rsm-test',http_received_monotonic=10.,
             queued_monotonic=10.2,scheduled_monotonic=10.5,first_token_monotonic=11.,
             last_token_monotonic=12.,frontend_finished_monotonic=12.1,
             prompt_tokens=8,output_tokens=n,num_cached_tokens=0,num_preemptions=0,lora_name=None,finish_reason='stop'),
        dict(kind='http_completed',request_id='rsm-test',monotonic=12.2,status=200)]
    result=dict(status='ok',status_code=200,response=dict(usage=dict(prompt_tokens=8,completion_tokens=n),
        metrics=dict(time_to_first_token_ms=500.,queue_time_ms=300.,generation_time_ms=1000.,
                     mean_itl_ms=1000./(n-1) if n>1 else None,tokens_per_second=n/1.5)))
    return result,events

class MetricDefinitions(unittest.TestCase):
    def test_server_entry_ttft_includes_prepare_and_queue(self):
        result,events=fixture();v=measure(result,events)
        self.assertEqual(v['status'],'valid')
        self.assertEqual(v['server_ttft_ms'],1000.)
        self.assertAlmostEqual(v['queue_ms'],300.)
        self.assertEqual(v['prefill_ms'],500.)
        self.assertAlmostEqual(v['api_prepare_ms'],200.)
        self.assertEqual(v['average_tpot_ms'],200.)
        self.assertEqual(v['server_last_token_ms'],2000.)
        self.assertAlmostEqual(v['server_http_complete_ms'],2200.)

    def test_one_token_tpot_is_excluded(self):
        result,events=fixture(1);events[1]['last_token_monotonic']=11.
        result['response']['metrics'].update(generation_time_ms=0.,tokens_per_second=2.)
        v=measure(result,events)
        self.assertEqual(v['status'],'valid');self.assertIsNone(v['average_tpot_ms'])

    def test_no_server_trace_does_not_use_client_timestamps(self):
        result,events=fixture();result.update(started_at=100.,finished_at=102.)
        v=measure(result,[events[0],events[2]])
        self.assertEqual(v['status'],'missing');self.assertNotIn('server_ttft_ms',v)

    def test_duplicate_event_is_not_silently_averaged(self):
        result,events=fixture()
        with self.assertRaisesRegex(ValueError,'duplicate'):
            measure(result,events+[deepcopy(events[1])])

    def test_native_ttft_cannot_be_mistaken_for_http_ttft(self):
        result,events=fixture();result['response']['metrics']['time_to_first_token_ms']=1000.
        v=measure(result,events)
        self.assertEqual(v['status'],'invalid');self.assertIn('native_metric_mismatch:time_to_first_token_ms',v['errors'])

    def test_token_count_disagreement_is_reported(self):
        result,events=fixture();result['response']['usage']['completion_tokens']=5
        self.assertIn('usage_count_mismatch',measure(result,events)['errors'])

    def test_timestamp_order_is_checked(self):
        result,events=fixture();events[1]['queued_monotonic']=11.5
        self.assertIn('server_timestamps_out_of_order',measure(result,events)['errors'])

    def test_prometheus_counter_delta_excludes_created_and_pos(self):
        before='vllm:spec_decode_num_draft_tokens_total{engine="0"} 100\nvllm:spec_decode_num_accepted_tokens_total{engine="0"} 40\n'
        after=before.replace('100','110').replace('40','46')+'vllm:spec_decode_num_draft_tokens_created{engine="0"} 999\nvllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 200\n'
        v=counters(before,after)
        self.assertEqual(v['draft_tokens'],10.);self.assertEqual(v['accepted_tokens'],6.);self.assertEqual(v['acceptance'],.6)

class ServerAttribution(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):trace._ACTIVE.clear()

    async def test_middleware_pairs_received_and_completed_without_changing_body(self):
        observed=[];sent=[]
        async def app(scope,receive,send):
            await send(dict(type='http.response.start',status=200,headers=[]))
            await send(dict(type='http.response.body',body=b'{"ok":true}'))
        async def send(message):sent.append(message)
        with patch.object(trace,'install'),patch.object(trace,'emit',side_effect=observed.append),patch.object(trace.time,'monotonic',side_effect=[10.,12.]):
            m=trace.TimingMiddleware(app)
            await m(dict(type='http',path='/v1/chat/completions',headers=[(b'x-request-id',b'rsm-test')]),None,send)
        self.assertEqual([v['kind'] for v in observed],['http_received','http_completed'])
        self.assertEqual(observed[1]['monotonic'],12.)
        self.assertEqual(sent[1]['body'],b'{"ok":true}');self.assertEqual(trace._ACTIVE,{})

    async def test_other_requests_are_uninstrumented(self):
        async def app(scope,receive,send):pass
        with patch.object(trace,'install'),patch.object(trace,'emit') as emit:
            await trace.TimingMiddleware(app)(dict(type='http',path='/v1/chat/completions',headers=[]),None,None)
            emit.assert_not_called()

    def test_engine_external_id_matches_registered_request_exactly(self):
        stats=SimpleNamespace(arrival_time=100.,first_token_latency=.8,queued_ts=10.2,scheduled_ts=10.5,
            first_token_ts=11.,last_token_ts=12.,num_generation_tokens=6,num_preemptions=0)
        req=SimpleNamespace(stats=stats,external_req_id='chatcmpl-rsm-test',prompt_len=8,max_tokens_param=30,num_cached_tokens=0,lora_name=None)
        trace._ACTIVE['rsm-test']=10.
        with patch.object(trace,'emit') as emit:
            trace.engine_record(req,'stop')
            self.assertEqual(emit.call_args.args[0]['request_id'],'rsm-test')
            req.external_req_id='chatcmpl-rsm-other'
            trace.engine_record(req,'stop')
            self.assertEqual(emit.call_args.args[0]['kind'],'unmatched_engine_request')

if __name__=="__main__":unittest.main()
