"""Reciprocal Rank Fusion behaviour."""
from src.retrieve import fuse_rrf


def test_agreement_between_retrievers_wins():
    fused = fuse_rrf([["a", "b", "c"], ["b", "c", "d"]], k=4, rrf_k=60)
    ids = [cid for cid, _ in fused]
    # b is rank 2 in both lists -> best consensus
    assert ids[0] == "b"
    assert set(ids[:3]) == {"b", "c", "a"}


def test_single_list_rank_one_wins():
    fused = fuse_rrf([["solo"], ["x", "y", "z"]], k=3, rrf_k=60)
    assert fused[0][0] == "solo"


def test_scores_are_rank_only():
    one = fuse_rrf([["a", "b"], ["a", "b"]], k=2, rrf_k=60)
    two = fuse_rrf([["a", "b"], ["a", "b"]], k=2, rrf_k=10)
    assert [cid for cid, _ in one] == [cid for cid, _ in two]
    # a is rank 1 in both lists, k=60 -> 2/61
    assert abs(one[0][1] - 2 / 61) < 1e-12


def test_respects_k_and_dedupes():
    fused = fuse_rrf([["a", "b", "c"], ["a", "d"]], k=2, rrf_k=60)
    assert len(fused) == 2
    assert len({cid for cid, _ in fused}) == 2


def test_rrf_k_damps_top_rank():
    tight = fuse_rrf([["a", "b"]], k=2, rrf_k=60)
    loose = fuse_rrf([["a", "b"]], k=2, rrf_k=10)
    assert (tight[0][1] / tight[1][1]) < (loose[0][1] / loose[1][1])
