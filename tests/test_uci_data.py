from pathlib import Path

import pytest

from hmcl.data import build_dataset


def test_uci_concrete_protocol_loader() -> None:
    data_root = Path("data/uci")
    if not (data_root / "concrete" / "data" / "data.txt").exists():
        pytest.skip("local UCI benchmark data is not available")

    train = build_dataset(
        "concrete",
        source="uci",
        split="train",
        data_root=data_root,
        split_num=0,
    )
    test = build_dataset(
        "concrete",
        source="uci",
        split="test",
        data_root=data_root,
        split_num=0,
    )

    assert train.spec.input_dim == 8
    assert train.spec.target_dim == 1
    assert len(train) == 927
    assert len(test) == 103
    assert train[0]["x"].shape == (8,)
    assert train[0]["y"].shape == (1,)
