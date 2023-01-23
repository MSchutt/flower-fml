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
    assert x2 == 190

    idx = 1
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 190
    assert x2 == 200

    idx = 2
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 200
    assert x2 == 390

    idx = 3
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 390
    assert x2 == 400

    idx = 4
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 400
    assert x2 == 590

    idx = 5
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 590
    assert x2 == 600

    idx = 6
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 600
    assert x2 == 790

    idx = 7
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 790
    assert x2 == 800

    idx = 8
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 800
    assert x2 == 990

    idx = 9
    x1, x2 = get_partition_range(n_total, num_partitions, idx, uneven=True)
    assert x1 == 990
    assert x2 == 1000


    n_total = 100
    num_partitions = 5
    x1, x2 = get_partition_range(n_total, num_partitions, 0, uneven=True)
    assert x1 == 0
    assert x2 == 38

    x1, x2 = get_partition_range(n_total, num_partitions, 1, uneven=True)
    assert x1 == 38
    assert x2 == 40

    x1, x2 = get_partition_range(n_total, num_partitions, 2, uneven=True)
    assert x1 == 40
    assert x2 == 78

    x1, x2 = get_partition_range(n_total, num_partitions, 3, uneven=True)
    assert x1 == 78
    assert x2 == 80

    x1, x2 = get_partition_range(n_total, num_partitions, 4, uneven=True)
    assert x1 == 80
    assert x2 == 118




