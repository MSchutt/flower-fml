import pytest

from utils import get_partition_range


def test_partition_same_distribution():
    num_partitions = 10
    n_total = 1000
    idx = 0
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=False)

    assert x1 == 0
    assert x2 == 100

    idx = 2

    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=False)
    assert x1 == 200
    assert x2 == 300

    idx = 9
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=False)
    assert x1 == 900
    assert x2 == 1000


def test_partition_uneven_distribution():
    num_partitions = 10
    n_total = 1000

    idx = 0

    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 0
    assert x2 == 175

    idx = 1
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 175
    assert x2 == 200

    idx = 2
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 200
    assert x2 == 375

    idx = 3
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 375
    assert x2 == 400

    idx = 4
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 400
    assert x2 == 575

    idx = 5
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 575
    assert x2 == 600

    idx = 6
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 600
    assert x2 == 775

    idx = 7
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 775
    assert x2 == 800

    idx = 8
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 800
    assert x2 == 975

    idx = 9
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 975
    assert x2 == 1000



