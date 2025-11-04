import math

from scripts.train import ModularDataset  # simple import since we put it in scripts/


def test_modular_dataset_split_deterministic():
    cfg = dict(N=97, op="add", train_fraction=0.1, seed=1337, split="hash")

    ds_train_1 = ModularDataset(role="train", **cfg)
    ds_val_1 = ModularDataset(role="val", **cfg)
    ds_test_1 = ModularDataset(role="test", **cfg)

    ds_train_2 = ModularDataset(role="train", **cfg)
    ds_val_2 = ModularDataset(role="val", **cfg)
    ds_test_2 = ModularDataset(role="test", **cfg)

    # sizes are stable
    assert len(ds_train_1) == len(ds_train_2)
    assert len(ds_val_1) == len(ds_val_2)
    assert len(ds_test_1) == len(ds_test_2)

    # disjointness: our implementation slices val from the "train mask" pool
    # train/val should be disjoint, and test disjoint from both
    set_train = set(ds_train_1.train_pairs)
    set_val = set(ds_val_1.val_pairs)
    set_test = set(ds_test_1.test_pairs)

    assert set_train.isdisjoint(set_val)
    assert set_train.isdisjoint(set_test)
    assert set_val.isdisjoint(set_test)

    # total coverage equals N^2
    N = cfg["N"]
    assert len(set_train) + len(set_val) + len(set_test) == N * N
