from wavelet.configs.config import RLConfig, SFTConfig
from wavelet.configs.rl_config import RLConfig as LegacyRLConfig
from wavelet.configs.sft import SFTConfig as LegacySFTConfig


def test_canonical_config_imports_preserve_legacy_class_identity() -> None:
    assert RLConfig is LegacyRLConfig
    assert SFTConfig is LegacySFTConfig
