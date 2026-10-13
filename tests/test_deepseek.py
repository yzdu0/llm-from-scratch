from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from llm.attention import LatentAttention
from llm.experts import MixtureOfExperts
from llm.generate import load_model
from llm.layers import RMSNorm, apply_rope
from llm.model import LanguageModel, ModelConfig
from llm.tokenizer import BPETokenizer
from llm.train import save_checkpoint


class DeepSeekTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)
        self.config = ModelConfig(context_size=8, n_layers=2, n_heads=2, n_embd=16,
                                  dropout=0, attention="mla", normalization="rmsnorm",
                                  n_experts=4, mtp_weight=0.1)

    def test_rmsnorm_matches_definition_and_preserves_dtype(self):
        norm = RMSNorm(4)
        x = torch.tensor([[1., 2., 3., 4.]])
        torch.testing.assert_close(norm(x), x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt())
        self.assertEqual(norm(x.to(torch.bfloat16)).dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(norm(torch.zeros_like(x))).all())

    def test_absorbed_mla_matches_expanded_keys_values_and_gradients(self):
        attention = LatentAttention(self.config).eval()
        x = torch.randn(2, 5, 16, requires_grad=True)
        actual, _ = attention(x)
        q = attention.q_up(attention.q_norm(attention.q_down(x)))
        q = q.view(2, 5, 2, -1).transpose(1, 2)
        q_content, q_rope = q.split((attention.head_dim, attention.rope_dim), dim=-1)
        latent, k_rope = attention.kv_down(x).split((attention.rank, attention.rope_dim), dim=-1)
        expanded = attention.kv_up(attention.kv_norm(latent)).view(2, 5, 2, -1).transpose(1, 2)
        k_content, values = expanded.chunk(2, dim=-1)
        q_rope = apply_rope(q_rope, attention.rope_cos[:5], attention.rope_sin[:5])
        k_rope = apply_rope(k_rope[:, None], attention.rope_cos[:5], attention.rope_sin[:5])
        q = torch.cat((q_content, q_rope), dim=-1)
        k = torch.cat((k_content, k_rope.expand(-1, 2, -1, -1)), dim=-1)
        scores = (q @ k.transpose(-2, -1)) / (attention.head_dim + attention.rope_dim) ** 0.5
        weights = scores.masked_fill(~attention.mask[:5, :5], float("-inf")).softmax(-1)
        expected = attention.projection((weights @ values).transpose(1, 2).reshape(2, 5, 16))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        parameters = (x, attention.kv_up.weight)
        actual_gradients = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected.sum(), parameters)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)

    def test_cached_chunks_match_full_logits(self):
        tokens = torch.randint(0, 256, (2, 7))
        for attention in ("mha", "mla"):
            with self.subTest(attention=attention), torch.no_grad():
                model = LanguageModel(replace(self.config, attention=attention)).eval()
                expected, _ = model(tokens)
                cache, outputs = None, []
                for start, end in ((0, 3), (3, 5), (5, 7)):
                    logits, _, cache = model(tokens[:, start:end], cache=cache, use_cache=True)
                    outputs.append(logits)
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-6, rtol=1e-5)
                if attention == "mla":
                    rank = model.blocks[0].attention.rank
                    rope_dim = model.blocks[0].attention.rope_dim
                    self.assertEqual(cache[0][0].shape, (2, 7, rank))
                    self.assertEqual(cache[0][1].shape, (2, 7, rope_dim))
                    self.assertLess(sum(t.numel() for t in cache[0]), 2 * 2 * 7 * 16)

    def test_cached_generation_matches_recomputation_through_window_rollover(self):
        configs = [self.config, replace(self.config, attention="mha"),
                   ModelConfig(context_size=8, n_layers=1, n_heads=2, n_embd=16,
                               dropout=0, position_encoding="learned", feed_forward="gelu")]
        for config in configs:
            with self.subTest(attention=config.attention, positions=config.position_encoding):
                model = LanguageModel(config)
                for length in (3, 12):
                    prompt = torch.randint(0, 256, (1, length))
                    cached = model.generate(prompt, 12, temperature=0)
                    reference = model.generate(prompt, 12, temperature=0, use_cache=False)
                    torch.testing.assert_close(cached, reference)
                self.assertTrue(model.training)

    def test_mla_and_experts_are_causal(self):
        model = LanguageModel(self.config).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        changed = torch.tensor([[1, 2, 3, 90, 91]])
        torch.testing.assert_close(model(tokens)[0][:, :3], model(changed)[0][:, :3])

    def test_sparse_experts_match_reference_and_only_selected_experts_get_gradients(self):
        moe = MixtureOfExperts(self.config)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.routing_bias.copy_(torch.tensor([1., 0.5, -0.5, -1.]))
        x = torch.randn(2, 3, 16)
        selected, weights = moe.route(x.reshape(-1, 16))
        self.assertEqual(set(selected.flatten().tolist()), {0, 1})
        torch.testing.assert_close(weights, torch.full_like(weights, 0.5))
        gates = torch.zeros(6, 4).scatter(1, selected, weights)
        expected = moe.shared(x.reshape(-1, 16))
        for i, expert in enumerate(moe.experts):
            expected = expected + expert(x.reshape(-1, 16)) * gates[:, i:i + 1]
        actual = moe(x)
        torch.testing.assert_close(actual, expected.view_as(x))
        self.assertEqual(moe.counts.sum().item(), 12)
        actual.square().sum().backward()
        self.assertIsNotNone(moe.shared.gate.weight.grad)
        self.assertIsNotNone(moe.router.weight.grad)
        self.assertIsNotNone(moe.experts[0].gate.weight.grad)
        self.assertIsNone(moe.experts[2].gate.weight.grad)

    def test_router_bias_updates_after_step_and_evaluation_is_read_only(self):
        model = LanguageModel(self.config)
        moe = model.blocks[1].mlp
        moe.counts.copy_(torch.tensor([9., 3., 0., 0.]))
        before = moe.routing_bias.clone()
        model.update_router_bias()
        torch.testing.assert_close(moe.routing_bias - before, torch.tensor([-0.001, 0., 0.001, 0.001]))
        self.assertEqual(moe.counts.sum().item(), 0)
        before = moe.routing_bias.clone()
        model.eval()
        model(torch.tensor([[1, 2, 3]]))
        model.update_router_bias()
        torch.testing.assert_close(moe.routing_bias, before)
        self.assertEqual(moe.counts.sum().item(), 0)

    def test_sequential_mtp_alignment_and_gradients(self):
        model = LanguageModel(self.config)
        captured = []
        hook = model.blocks[-1].register_forward_hook(lambda module, args, output: captured.append(output[0]))
        tokens = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5]])
        logits, loss = model(tokens, targets)
        hook.remove()
        hidden = model.mtp(captured[0][:, :-1], model.token_embedding(targets[:, :-1]))
        expected = F.cross_entropy(logits.reshape(-1, 256), targets.flatten())
        expected += 0.1 * F.cross_entropy(model.head(hidden).reshape(-1, 256), targets[:, 1:].flatten())
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(model.mtp.merge.weight.grad)
        self.assertIsNotNone(model.mtp.block.attention.kv_up.weight.grad)
        model.eval()
        with patch.object(model.mtp, "forward", side_effect=AssertionError("MTP used during evaluation")):
            evaluated, loss = model(tokens, targets)
        torch.testing.assert_close(loss, F.cross_entropy(evaluated.reshape(-1, 256), targets.flatten()))
        self.assertEqual(set(model.loss_components), {"next_token"})

    def test_deepseek_checkpoint_roundtrip_includes_router_bias(self):
        tokenizer = BPETokenizer.train("To be, or not to be.", vocab_size=256)
        model = LanguageModel(self.config).eval()
        model.blocks[1].mlp.routing_bias.fill_(0.25)
        prompt = torch.tensor([[1, 2, 3]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, tokenizer, 10, 4.2)
            restored, _ = load_model(path)
            torch.testing.assert_close(restored(prompt)[0], model(prompt)[0])
            self.assertEqual(restored.config.attention, "mla")
            torch.testing.assert_close(restored.blocks[1].mlp.routing_bias,
                                       model.blocks[1].mlp.routing_bias)

    def test_invalid_configuration_and_cache(self):
        for kwargs in ({"rope_dim": 3}, {"n_experts": 1, "experts_per_token": 2},
                       {"attention": "mla", "position_encoding": "learned"}, {"mtp_weight": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(**kwargs)
        model = LanguageModel(self.config)
        with self.assertRaises(ValueError):
            model(torch.tensor([[1]]), use_cache=True)
        model.eval()
        with self.assertRaises(ValueError):
            model(torch.tensor([[1]]), cache=[], use_cache=True)
