"""External location is tool context, not a symptom-extraction slot."""
import unittest
from unittest.mock import patch
from agent.runtime_utils import _nearby_hospital_lines
from memory.memory import ConversationMemory


class ToolLocationTests(unittest.TestCase):
    def test_location_survives_empty_fact_projection_and_reaches_hospital_tool(self):
        memory=ConversationMemory();memory.enable_fact_mode()
        memory.update_tool_location('上海')
        inp=memory.fact_input('头痛。')
        memory.apply_fact_delta(inp,{'facts':[]})
        with patch('tools.registry.default_registry.get') as get:
            tool=get.return_value;tool.return_value={'status':'error'}
            lines=_nearby_hospital_lines(memory.get_case_state())
            tool.assert_called_once_with(location='上海')
        self.assertTrue(any('不要等待查询结果' in line for line in lines))

    def test_explicit_fact_city_overrides_external_setting(self):
        with patch('tools.registry.default_registry.get') as get:
            get.return_value.return_value={'status':'error'}
            _nearby_hospital_lines({'city':'杭州','tool_location':{'city':'上海','source':'user_setting'}})
            get.return_value.assert_called_once_with(location='杭州')

    def test_denied_external_city_is_not_resurrected_but_coordinates_remain_available(self):
        state={'tool_location':{'city':'上海','source':'user_setting'},'fact_view':{'facts':[
            dict(field='city',value='上海',polarity='negated',subject='patient',temporality='current')]}}
        with patch('tools.registry.default_registry.get') as get:
            _nearby_hospital_lines(state)
            get.return_value.assert_not_called()
            state['user_coords']={'latitude':31.2,'longitude':121.4}
            _nearby_hospital_lines(state)
            get.return_value.assert_called_once_with(location='',latitude=31.2,longitude=121.4)

    def test_external_setting_is_not_fact_evidence_and_persists_across_restore(self):
        memory=ConversationMemory();memory.enable_fact_mode()
        memory.update_tool_location('上海',source='evaluation_fixture')
        memory.commit_fact_ledger(memory.case_state['fact_ledger'])
        state=memory.get_case_state()
        self.assertFalse(any(f['field']=='city' for f in state['fact_view']['facts']))
        restored=ConversationMemory(session_store=memory.session_store,session_id=memory.session_id)
        self.assertEqual(restored.get_case_state()['tool_location'],state['tool_location'])

    def test_invalid_settings_are_rejected(self):
        memory=ConversationMemory()
        for city,source in [(None,'user_setting'),('上海','model_inference'),('x'*101,'user_setting')]:
            with self.assertRaises(ValueError):memory.update_tool_location(city,source=source)


if __name__=='__main__':unittest.main()
