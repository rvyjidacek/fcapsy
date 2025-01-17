# Typicality implementation
#
# Rosch, Eleanor, and Carolyn B. Mervis.
# "Family resemblances: Studies in the internal structure of categories."
# Cognitive psychology 7.4 (1975): 573-605.
#
# Belohlavek, Radim, and Tomas Mikula.
# "Typicality in conceptual structures within the framework of formal concept analysis."

import math

from itertools import compress
from fcapsy.decorators import metadata
from fcapsy.utils import iterator_mean


def _calculate_similarities(item, items_to_compare, similarity_function):
    return map(lambda other: similarity_function(item, other), items_to_compare)


@metadata(name='Average Typicality', short_name='typØ', latex='typ')
def typicality_avg(item, concept, context, similarity_function, axis=0):
    item = next(context.filter([item], axis=axis))

    item_set = concept.extent

    if axis == 1:
        item_set = concept.intent

    similarities = _calculate_similarities(
        item, context.filter(item_set, axis=axis), similarity_function)

    return iterator_mean(similarities)


@metadata(name='Average Typicality without Core', short_name='typØc', latex='typ^\\mathrm{c}')
def typicality_avg_without_core(item, concept, context, similarity_function, axis=0):
    item = next(context.filter([item], axis=axis))

    item_set = concept.extent
    core_set = concept.intent

    if axis == 1:
        item_set = concept.intent
        core_set = concept.extent

    item = item.difference(core_set)

    others = [row.difference(core_set)
              for row in context.filter(item_set, axis=axis)]

    similarities = _calculate_similarities(
        item, others, similarity_function)

    return iterator_mean(similarities)


@metadata(name='Minimal Typicality', short_name='typ∧', latex='typ^\\mathrm{min}')
def typicality_min(item, concept, context, similarity_function, axis=0):
    item = next(context.filter([item], axis=axis))

    item_set = concept.extent

    if axis == 1:
        item_set = concept.intent

    similarities = _calculate_similarities(
        item, context.filter(item_set, axis=axis), similarity_function)

    return min(similarities)


def _calculate_weights(objects):
    objects = map(lambda x: x.bools(), objects)
    return [sum(y) for y in zip(*objects)]


@metadata(name='Rosch Typicality', short_name='typ_\\mathrm{RM}')
def typicality_rosch(item, concept, context):
    item = next(context.filter([item]))

    weights = _calculate_weights(context.filter(concept.extent))

    return sum(compress(weights, item.bools()))


@metadata(name='Rosch Logarithm Typicality', short_name='typ_\\mathrm{RM}^\\mathrm{ln}')
def typicality_rosch_ln(item, concept, context):
    item = next(context.filter([item]))

    weights = _calculate_weights(context.filter(concept.extent))
    weights = map(lambda x: math.log(x) if x != 0 else -math.inf, weights)

    return sum(compress(weights, item.bools()))
