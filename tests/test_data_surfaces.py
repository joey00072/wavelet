from wavelet.data import rl, sft
from wavelet.data.dataset import SFTDataset
from wavelet.data.rl_dataset import PackedRLDataset
from wavelet.data.rl_records import serialize_rl_record
from wavelet.data.rl_types import RLExample


def test_sft_surface_preserves_legacy_objects() -> None:
    assert sft.SFTDataset is SFTDataset


def test_rl_surface_preserves_legacy_objects() -> None:
    assert rl.PackedRLDataset is PackedRLDataset
    assert rl.RLExample is RLExample
    assert rl.serialize_rl_record is serialize_rl_record
