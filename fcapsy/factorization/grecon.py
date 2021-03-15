from collections import namedtuple
import numpy as np

from fcapsy import Context
from fcapsy.algorithms.fcbo import fcbo

CoverTuple = namedtuple('CoverTuple', ['concept', 'cover_size', 'index'])


def _numpy_select_max_cover_concept(tuples, concepts, concept_matricies) -> tuple:
    max_cover_size = 0
    max_cover_concept = None
    index = -1

    tuples_intersection: np.array = np.zeros(tuples.shape, dtype=bool)

    for i, concept in enumerate(concepts):
        if concept.extent.any() and concept.intent.any():
            # tuples_intersection = context.tuples_for_concept(concept)

            tuples_intersection.fill(False)
            tuples_intersection[tuple(concept_matricies[i])] = True
            tuples_intersection &= tuples
            intersection_count = np.count_nonzero(tuples_intersection)

            if intersection_count > max_cover_size:
                max_cover_size = intersection_count
                max_cover_concept = concept
                index = i

    return CoverTuple(max_cover_concept, max_cover_size, index)


def numpy_grecon(context: Context) -> list:
    s = fcbo(context)
    # context.tuples()
    u = np.array(list(map(context.Attributes.bools, context.rows)), dtype=bool)
    context_matrix = np.zeros(u.shape, dtype=bool)  # context.tuples()
    concept_matricies = list(map(lambda c: np.meshgrid(list(c.extent.iter_set()), list(c.intent.iter_set())), s))

    f = []
    #f = np.zeros(context_matrix.shape, dtype=bool)
    concept_matrix = np.zeros(context_matrix.shape, dtype=bool)

    while np.any(u):
        max_tuple = _numpy_select_max_cover_concept(u, s, concept_matricies)

        if max_tuple.concept:
            concept = max_tuple.concept
            f.append(max_tuple.concept)
            concept_matrix[tuple(concept_matricies[max_tuple.index])] = True
            del s[max_tuple.index]


            # max_concept_tuples = context.tuples_for_concept(max_tuple.concept)
            max_concept_tuples = np.zeros(context_matrix.shape, dtype=bool)
            max_concept_tuples[tuple(concept_matricies[max_tuple.index])] = True

            del concept_matricies[max_tuple.index]

            # u = u & ~max_concept_tuples
            u &= ~max_concept_tuples
    return f
