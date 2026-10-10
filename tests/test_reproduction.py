"""Validate the public experiment inputs and risk-pooling semantics."""
import hashlib
import io
import json
import os
import shlex
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
SUITES = json.loads((ROOT / 'configs/suites.json').read_text(encoding='utf-8'))


def commands(mode, suite):
    text = (ROOT / 'scripts' / mode / (suite + '.sh')).read_text(encoding='utf-8')
    return [shlex.split(line) for line in text.splitlines() if line.startswith('python ')]

from harl.envs.multi_lt_transship.eppen_correlation import correlation_from_sequences, pooled_demand_variance, load_training_correlation
from harl.envs.multi_lt_transship.transship_vec_env import VecTransshipEnv
import test_baseline_vec as heuristic_entry


class ReproductionInputs(unittest.TestCase):
    def test_complete_command_coverage(self):
        for suite, counts in {'backlog': (120, 216), 'backlog_additional': (105, 189), 'lost': (135, 324),
                              'beverage': (40, 72), 'ririshun': (40, 72)}.items():
            for mode, expected in zip(('rl', 'heuristic'), counts):
                with self.subTest(suite=suite, mode=mode):
                    jobs = list(commands(mode, suite))
                    self.assertEqual(len(jobs), expected)
                    self.assertEqual(len({tuple(job) for job in jobs}), expected)
                    for job in jobs:
                        self.assertTrue((ROOT / job[job.index('--load_config') + 1]).is_file())

    def test_real_data_forecast_weights(self):
        weights = {'SKU006': '0.1', 'SKU019': '0.1', 'SKU022': '0.1', 'SKU023': '0.9',
                   'SKU029': '0.1', 'SKU045': '0.1', 'SKU046': '0.1', 'SKU062': '0.1',
                   '1936c558': '0.3', '26f47370': '0.1', '3b6fda08': '0.1', '3e0092db': '0.1',
                   'a4830d20': '0.6', 'b01ff11b': '0.1', 'b0bdce95': '0.1', 'bf67f87f': '0.1'}
        for suite in ('beverage', 'ririshun'):
            for job in commands('heuristic', suite):
                config = json.loads((ROOT / job[job.index('--load_config') + 1]).read_text(encoding='utf-8'))
                sku = config['env_args']['dataset_name']
                forecast = job[job.index('--estimate_demand_method') + 1]
                self.assertEqual(job[job.index('--alpha') + 1], weights[sku] if forecast == 'ema' else '0.3')
                strategy = job[job.index('--run_function') + 1]
                self.assertEqual(job[job.index('--eppen_use_demand_correlation') + 1], str(strategy == 'eppen'))

    def test_data_integrity_and_sku_scope(self):
        manifest = json.loads((ROOT / 'data_manifest.json').read_text(encoding='utf-8'))
        for path, digest in manifest.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        real_names = {entry['name'] for suite in ('beverage', 'ririshun') for entry in SUITES[suite]}
        data_root = ROOT / 'harl/envs/multi_lt_transship'
        self.assertEqual({p.name for p in (data_root / 'gen_source').iterdir()}, real_names)
        for split in ('eval_data', 'test_data'):
            self.assertEqual({p.name for p in (data_root / split).iterdir() if p.name in real_names}, real_names)

    def test_config_and_sequence_compatibility(self):
        checked = set()
        data_root = ROOT / 'harl/envs/multi_lt_transship'
        for suite, entries in SUITES.items():
            for entry in entries:
                config = json.loads((ROOT / entry['config']).read_text(encoding='utf-8'))
                env = config['env_args']
                with self.subTest(config=entry['config']):
                    self.assertEqual(env['episode_length'], config['algo_args']['train']['episode_length'])
                    real = suite in ('beverage', 'ririshun')
                    self.assertEqual(env['eppen_use_demand_correlation'], real)
                    self.assertEqual(env['backlog_tf'], suite != 'lost')
                    if suite == 'backlog_additional':
                        self.assertEqual([env[key] for key in ('P', 'C', 'K')], [0, 0, 0])
                        self.assertEqual(env['episode_length'], 200)
                    if real:
                        self.assertEqual(env['eval_dataset_mode'], 'all')
                        corr, length = load_training_correlation(env, env['num_agents'])
                        self.assertGreaterEqual(length, env['episode_length'])
                        self.assertEqual(corr.shape, (env['num_agents'], env['num_agents']))
                    for split in ('eval_data', 'test_data'):
                        for agent in range(env['num_agents']):
                            folder = data_root / split / env['dataset_name'] / f'agent_{agent}'
                            key = (str(folder), env['episode_length'])
                            if key in checked:
                                continue
                            checked.add(key)
                            arrays = [np.loadtxt(folder / name, delimiter=',', ndmin=2)
                                      for name in ('demand.csv', 'LT.csv', 'shippingloss.csv')]
                            self.assertEqual(arrays[0].shape, arrays[1].shape)
                            self.assertEqual(arrays[0].shape, arrays[2].shape)
                            self.assertGreaterEqual(arrays[0].shape[0], env['episode_length'])
                            self.assertGreater(arrays[0].shape[1], 0)
                            for values in arrays:
                                self.assertTrue(np.isfinite(values).all())
                                self.assertTrue((values >= 0).all())
                            self.assertTrue((arrays[2] <= 1).all())

    def test_eppen_correlated_and_independent_variance(self):
        sigma = np.array([[2., 3.], [4., 1.]])
        np.testing.assert_allclose(pooled_demand_variance(sigma, np.eye(2))[:, 0], [13., 17.])
        np.testing.assert_allclose(pooled_demand_variance(sigma, np.ones((2, 2)))[:, 0], [25., 25.])
        sequences = [[1, 2, 3, 4], [2, 4, 6, 8], [7, 7, 7, 7]]
        corr = correlation_from_sequences(sequences)
        np.testing.assert_allclose(corr, [[1, 1, 0], [1, 1, 0], [0, 0, 1]], atol=1e-12)

    def test_every_environment_loads_all_splits(self):
        for suite, entries in SUITES.items():
            for entry in entries:
                with self.subTest(config=entry['config']):
                    args = json.loads((ROOT / entry['config']).read_text(encoding='utf-8'))['env_args']
                    args['num_envs'] = 2
                    env = VecTransshipEnv(args)
                    for split in ('train', 'eval', 'test'):
                        env.reset(usage=split)
                        self.assertEqual(env.demand.shape[1:], (args['num_agents'], args['episode_length']))
                        if suite in ('beverage', 'ririshun') and split != 'train':
                            np.testing.assert_array_equal(env.evaluation_episode_indices, np.arange(env.num_envs))
                    env.close()

    def test_heuristic_cli_preserves_policy_semantics(self):
        for suite in ('backlog', 'beverage'):
            config = SUITES[suite][0]['config']
            with patch.object(sys, 'argv', ['test_baseline_vec.py', '--load_config', config, '--alpha', '0.3']), \
                 patch.object(heuristic_entry, 'BaselineExperiment') as runner, patch('sys.stdout', new_callable=io.StringIO):
                heuristic_entry.main()
                args = runner.call_args.kwargs['args']
                self.assertEqual(args['transship_type'], 'force_to_transship')
                self.assertEqual(args['action_type'], 'heu_discrete')
                self.assertEqual(args['product_allocation_method']['allocate_func'], 'anup')
                self.assertEqual(args['num_envs'], 20)
                self.assertEqual(args['eppen_use_demand_correlation'], suite == 'beverage')


if __name__ == '__main__':
    unittest.main()
