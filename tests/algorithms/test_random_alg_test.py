import pytest
from fcapsy import Context, Lattice
from fcapsy.algorithms.fcbo import fcbo
from fcapsy.algorithms.rice_siff import concept_subset
from fcapsy.similarity import jaccard


random_data = [Context.from_random(20, 10) for i in range(10)]


@pytest.mark.parametrize("context", random_data)
def test_random_alg_test(context):
    fcbo_result = set(fcbo(context))
    lattice_result = set(Lattice(context).concepts)
    subset_result = set(concept_subset(context, jaccard))

    assert fcbo_result == lattice_result
    assert subset_result.issubset(fcbo_result)
