import pytest
import torch
from torchtune import generation
from tests.test_utils import base_test_model, fixed_init_model
from kvcompression.hooks.compressor import KVCompressor
from kvcompression.kv_cache.compression_strategy_protocol import DummyCompressionStrategy, RandomCompressionStrategy
from kvcompression.utils.generation import generate_with_compression

@pytest.mark.parametrize('strategy', ['noop', 'random'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_cpu_compressed_generation(model_base_config_fixture, strategy, device):
    def model():
        m = base_test_model(model_base_config_fixture, device=device)
        torch.manual_seed(0)
        fixed_init_model(m, min_val=-0.01, max_val=0.01, dtype=torch.float32, nonlinear=True)
        return m.eval()
    a, b = model(), model()
    torch.manual_seed(25)
    prompt = torch.randint(3,100,(1,30), device=device)
    compress = KVCompressor(model=a, compression_strategies=[DummyCompressionStrategy() if strategy == 'noop' else RandomCompressionStrategy()])
    with torch.no_grad(), torch.device(device), compress:
        a.setup_caches(batch_size=1, dtype=torch.float32)
        torch.manual_seed(42)
        actual = generate_with_compression(model=a, prompt=prompt, kv_compressor=compress, target_cache_size=2, max_generated_tokens=15, stop_tokens=[2])
    with torch.device(device):
        b.setup_caches(batch_size=1, dtype=torch.float32)
    torch.manual_seed(42)
    with torch.no_grad():
        expected, _ = generation.generate(model=b, prompt=prompt, max_generated_tokens=15, temperature=0.0, pad_id=2, stop_tokens=[2])
    if strategy == 'noop':
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert not torch.equal(actual, expected)
