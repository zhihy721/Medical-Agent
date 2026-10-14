import unittest

from tools.symptom_tool import _extract_duration
from memory.memory import ConversationMemory
from memory.profile_store import InMemoryProfileStore
from memory.session_store import InMemorySessionStore


class DurationTests(unittest.TestCase):
    def test_chinese_durations_and_correction(self):
        for text, expected in [("已经有两周了", "两周"), ("持续半个月", "半个月"),
                               ("十四天没有好", "十四天"), ("不是两周，实际两天", "两天"),
                               ("并非14天，是3天", "3天"), ("不是两周", ""),
                               ("咳嗽三天，没有发烧", "三天"), ("不影响睡眠，咳嗽3天", "3天")]:
            with self.subTest(text=text):
                self.assertEqual(_extract_duration(text), expected)

    def test_duration_conflict_can_be_confirmed_and_resolved(self):
        m = ConversationMemory(profile_store=InMemoryProfileStore(), session_store=InMemorySessionStore(), user_id="duration-test", session_id="duration-test")
        m.update_case({"duration": "两周"})
        m.update_case({"duration": "两天"})
        self.assertIn("duration", m.get_case_state()["contradiction_fields"])
        m.resolve_contradiction("duration")
        self.assertNotIn("duration", m.get_case_state()["contradiction_fields"])
        self.assertEqual(m.get_case_state()["duration"], "两天")


if __name__ == "__main__":
    unittest.main()
