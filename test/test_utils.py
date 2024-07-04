import sys
if not '.' in sys.path:
    sys.path.append('.')
import unittest
import torch
import os

import utils

class Test_utils(unittest.TestCase):
    def test_plot_metrics_v2(self):
        print('''---test_plot_metrics_v2---''')
        metrics = {
            'rope':'./results/llama_rope/logs/version_0/metrics.csv',
            'abs':'./results/llama_abs/logs/version_0/metrics.csv'
        }
        utils.plot_metrics_v2(metrics,save_dir=None)
        print()
if __name__=="__main__":
    unittest.main()