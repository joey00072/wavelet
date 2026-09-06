from wavelet.data import rl, sft
from wavelet.data.rl import PackedRLDataset, RLExample, serialize_rl_record
from wavelet.data.sft import SFTDataset


def test_sft_surface_preserves_legacy_objects() -> None:
    assert sft.SFTDataset is SFTDataset


def test_rl_surface_preserves_legacy_objects() -> None:
    assert rl.PackedRLDataset is PackedRLDataset
    assert rl.RLExample is RLExample
    assert rl.serialize_rl_record is serialize_rl_record
