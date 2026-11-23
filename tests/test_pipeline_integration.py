"""Offline synthetic corpus and frozen-feature integration coverage."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from robofate.config import PAPER, TASKS
from robofate.data import Corpus, assign_families, collate_contexts, to_device
from robofate.model import project_history_with_grad
from robofate.prepare import prepare_record
from test_contracts import context, record
from test_native_integration import tiny_model, native_context


def history_context(slots, length):
    result = native_context(history=True, length=length)
    result['input_ids'][0, 1:1 + slots * 16] = 6
    result['history_mask'] = result['input_ids'] == 6
    result['token_type_ids'][result['history_mask']] = 1
    if slots:
        result['history_pixel_values'] = torch.randn(slots, 3, 32, 32)
    return result


def cache_features(model, raw):
    cached = copy.deepcopy(raw)
    with torch.no_grad():
        for key in ('pixel_values', 'history_pixel_values'):
            if cached.get(key) is not None:
                cached[key] = model.dm.model.visual(pixel_values=cached[key], return_dict=True).last_hidden_state
    cached['visual_features'] = True
    return cached


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_ragged_history_prefix_matches_frozen_cache(self):
        model = tiny_model().eval()
        raw = [history_context(slots, length) for slots, length in ((0, 2), (1, 5), (3, 3), (2, 4))]
        cached = [cache_features(model, value) for value in raw]
        raw_batch, cached_batch = collate_contexts(raw), collate_contexts(cached)
        self.assertEqual(int(cached_batch['history_mask'].sum()), 6 * 16)
        self.assertEqual(cached_batch['history_pixel_values'].shape[0], 6)
        with torch.no_grad():
            a, length_a = model.prefix(raw_batch)
            b, length_b = model.prefix(cached_batch)
        self.assertEqual(length_a, length_b)
        for left, right in zip(a.layers, b.layers):
            torch.testing.assert_close(left.keys, right.keys, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(left.values, right.values, atol=2e-6, rtol=2e-5)

    def test_cached_current_summaries_match_raw_images(self):
        model = tiny_model().eval()
        raw = [native_context(length=i+2) for i in range(4)]
        cached = [cache_features(model, value) for value in raw]
        with torch.no_grad():
            a = model.summarize(collate_contexts(raw))
            b = model.summarize(collate_contexts(cached))
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_history_projection_remains_trainable(self):
        model = tiny_model().train()
        feature = cache_features(model, history_context(2, 3))['history_pixel_values']
        projected = project_history_with_grad(model.dm.model.vlm.model, feature)
        projected.square().mean().backward()
        projection = model.dm.model.vlm.model.multi_modal_projector.mm_input_projection_weight
        self.assertGreater(float(projection.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.dm.model.visual.parameters()))

    def test_mixed_history_velocity_matches_individual_samples(self):
        model = tiny_model().eval()
        contexts = [history_context(slots, length) for slots, length in ((0, 2), (1, 5), (3, 3), (2, 4))]
        batch = collate_contexts(contexts)
        x, time = torch.randn(4, 50, 32), torch.full((4,), 0.5)
        with torch.no_grad():
            cache, length = model.prefix(batch)
            together = model.velocity(batch, cache, length, x, time, moae=False)
            separate = []
            for i, item in enumerate(contexts):
                cache, length = model.prefix(item)
                separate.append(model.velocity(item, cache, length, x[i:i+1], time[i:i+1], moae=False))
        torch.testing.assert_close(together, torch.cat(separate), atol=2e-6, rtol=2e-5)


class PreparedCorpusTests(unittest.TestCase):
    def test_loaded_batches_use_four_rank_slots_and_homogeneous_decisions(self):
        with tempfile.TemporaryDirectory(prefix='robofate_synthetic_') as temporary:
            root = Path(temporary)
            rows = [record(TASKS[0], kind, 0) for kind in ('original', 'dual_fate', 'native_success')]
            assignment = assign_families(rows)
            torch.save(context(), root/'context.pt')
            for r in rows:
                r.update(split=assignment[r['family_id']], admitted=True, prepared=r['kind']+'.pt')
                payload = {'source_id':r['source_id'], 'intervention_history':['context.pt']*8,
                           'decision_histories':[['context.pt']*8],
                           'windows':[{'start':start, 'suffix_offset':start, 'context':'context.pt',
                                       'action':torch.full((50,32), float(start)),
                                       'mask':torch.ones(50,32,dtype=torch.bool)} for start in range(0,5000,50)]}
                torch.save(payload, root/r['prepared'])
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'format':'robofate-prepared-v1','split_seed':191126,'records':rows}))
            corpus = Corpus(manifest, require_counts=False)
            original, decisions = corpus.batch(1, 0, PAPER, 191126)
            self.assertEqual(len(original), 4)
            self.assertFalse(decisions)
            self.assertTrue(all('_original_' in s['source_id'] for s in original))
            positive, decisions = corpus.batch(2, 0, PAPER, 191126)
            self.assertEqual([d['label'] for d in decisions], [1.]*4)
            self.assertTrue(all(len(s['history']) == 8 for s in positive))
            _, negative_decisions = corpus.batch(4, 15, PAPER, 191126)
            self.assertEqual([d['label'] for d in negative_decisions], [0.]*4)
            same, _ = corpus.batch(2, 0, PAPER, 191126)
            other_rank, _ = corpus.batch(2, 1, PAPER, 191126)
            self.assertEqual([s['start'] for s in positive], [s['start'] for s in same])
            self.assertNotEqual([s['start'] for s in positive], [s['start'] for s in other_rank])
            moved = to_device((positive, decisions), torch.device('cpu'))
            self.assertEqual(moved[0][0]['history'][0]['input_ids'].device.type, 'cpu')
            corpus.context.cache_clear()
            corpus.source.cache_clear()

    def test_complete_teacher_suffix_and_intervention_history(self):
        class SyntheticProcessor:
            q01, q99 = np.zeros(14, np.float32), np.ones(14, np.float32)
            def encode(self, raw, root, index, with_history):
                return context(value=index+1)
        with tempfile.TemporaryDirectory(prefix='robofate_synthetic_') as temporary:
            root = Path(temporary)
            output = root/'prepared'
            output.mkdir()
            raw = {'instruction':'Synthetic recorded command fixture.', 'action_interface':'dm05_absolute_joint_command',
                   'native_success':True, 'teacher_start':175, 'first_adverse_action':225, 'action_budget':400,
                   'actions':[{'index':i,'applied_joint_command':[0.1]*14} for i in range(287)],
                   'observations':[{'applied_actions':i} for i in range(287)]}
            (root/'episode.json').write_text(json.dumps(raw))
            r = record(TASKS[0], 'dual_fate', 0)
            r.update(episode='episode.json', boundary=175)
            path = prepare_record(r, root, output, SyntheticProcessor())
            payload = torch.load(output/path, weights_only=True)
            self.assertEqual([w['start'] for w in payload['windows']], [175,225,275])
            self.assertEqual([int(w['mask'].sum()) for w in payload['windows']], [700,700,168])
            refs = payload['intervention_history']
            self.assertEqual(len(refs), 8)
            values = [int(torch.load(output/ref, weights_only=True)['input_ids'][0,0]) for ref in refs]
            self.assertEqual(values, list(range(1,177,25)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
