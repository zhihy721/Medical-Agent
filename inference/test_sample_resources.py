import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from inference.sample_resources import parse_gpu_csv, parse_meminfo, summarize, run


class ResourceSamplerTests(unittest.TestCase):
    def test_benchmark_stop_marker_finishes_normally_without_hiding_samples(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);meminfo=root/'meminfo';meminfo.write_text('unused')
            marker=root/'stop'
            def sample():
                marker.touch()
                return dict(time_utc='2026-10-01T00:00:00+00:00',
                    gpu=[dict(index=0,memory_used_mib=19000)],
                    wsl=dict(memavailable_bytes=600,swapfree_bytes=100))
            with patch('inference.sample_resources.MEMINFO',meminfo),patch('inference.sample_resources.sample_once',side_effect=sample):
                report=run(root/'resources',seconds=1800,interval=1,stop_file=marker)
                self.assertEqual((report['status'],report['samples']),('completed',1))
                self.assertEqual(report['stop_reason'],'owned_benchmark_stop_marker')
                self.assertEqual(report['gpu_peak_used_mib'],{'0':19000})
                with self.assertRaisesRegex(ValueError,'stop file already exists'):
                    run(root/'not_created',seconds=1,interval=1,stop_file=marker)
                self.assertFalse((root/'not_created').exists())

    def test_parse_gpu_and_memory(self):
        gpu = parse_gpu_csv('0, 19162, 24455, 87\n')
        self.assertEqual(gpu[0]['memory_used_mib'], 19162)
        memory = parse_meminfo('MemTotal: 7693976 kB\nMemAvailable: 6900000 kB\n'
                               'SwapTotal: 2097152 kB\nSwapFree: 2097152 kB\n')
        self.assertEqual(memory['memavailable_bytes'], 6900000 * 1024)

    def test_bad_metrics_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            parse_gpu_csv('0, N/A, 24455, 0')
        with self.assertRaises(ValueError):
            parse_gpu_csv('0, 24500, 24455, 0')
        with self.assertRaises(ValueError):
            parse_meminfo('MemTotal: 100 kB\nMemAvailable: 90 kB\n')

    def test_summary_keeps_failures_visible(self):
        rows = [dict(status='ok', gpu=[dict(index=0,memory_used_mib=19000)],
                     wsl=dict(memavailable_bytes=600,swapfree_bytes=100)),
                dict(status='failed', error='TimeoutExpired'),
                dict(status='ok', gpu=[dict(index=0,memory_used_mib=19300)],
                     wsl=dict(memavailable_bytes=400,swapfree_bytes=80))]
        summary = summarize(rows)
        self.assertEqual(summary['gpu_peak_used_mib'], {'0':19300})
        self.assertEqual((summary['succeeded'],summary['failed']), (2,1))
        self.assertEqual(summary['wsl_min_mem_available_bytes'],400)


if __name__ == '__main__':
    unittest.main()
