import csv
import random
from typing import Type, Tuple, Iterator
from itertools import compress
import itertools

from bitsets.bases import BitSet
from bitsets import bitset


class Context:
    def __init__(self, matrix: list, objects_labels: list, attributes_labels: list, name: str = None):
        self.Objects = bitset('Objects', objects_labels)
        self.Attributes = bitset('Attributes', attributes_labels)
        # self.CartesianProduct = bitset("CartesianProduct", range(0, len(objects_labels) * len(attributes_labels)))

        self.rows = tuple(map(self.Attributes.frombools, matrix))
        self.columns = tuple(map(self.Objects.frombools, zip(*matrix)))
        self.name = name

    def __repr__(self):
        if self.name:
            return "Context({}, {}x{})".format(self.name, len(self.rows), len(self.columns))
        return "Context({}x{})".format(len(self.rows), len(self.columns))

    @classmethod
    def from_random(cls, number_of_objects, number_of_attributes):
        matrix = [random.choices([0, 1], k=number_of_attributes)
                  for i in range(number_of_objects)]

        return cls(matrix, tuple(range(number_of_objects)), tuple(range(number_of_attributes)))

    @classmethod
    def from_pandas(cls, dataframe, name: str = None):
        return cls(dataframe.values, tuple(dataframe.index), tuple(dataframe.columns), name=name)

    @classmethod
    def from_csv(cls, filename: str, objects_labels: list = [],
                 attribute_labels: list = [], delimiter: str = ',', name: str = None):
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file, delimiter=delimiter)

            bools = []
            labels = []

            for idx, row in enumerate(csv_reader):
                if idx == 0 and not attribute_labels:
                    attribute_labels = tuple(row[1:])
                else:
                    if not objects_labels:
                        labels.append(row.pop(0))

                    bools.append(tuple(map(int, row)))

            if not objects_labels:
                objects_labels = labels

        return cls(bools, tuple(objects_labels), tuple(attribute_labels), name=name)

    @classmethod
    def from_fimi(cls, filename: str, objects_labels: list = None, attribute_labels: list = None, name: str = None):
        with open(filename, 'r') as file:
            max_attribute = 0
            rows = []

            for line in file:
                # remove '\n' from line
                line = line.strip()
                row_attributes = []

                for value in line.split():
                    attribute = int(value)
                    row_attributes.append(attribute)
                    max_attribute = max(attribute, max_attribute)

                rows.append(row_attributes)

            bools = [[True if i in row else False for i in range(max_attribute + 1)]
                     for row in rows]

        if objects_labels is None:
            objects_labels = tuple(map(str, range(len(bools))))

        if attribute_labels is None:
            attribute_labels = tuple(map(str, range(len(bools[0]))))

        return cls(bools, objects_labels, attribute_labels, name=name)

    def to_bools(self) -> Iterator[tuple]:
        return map(self.Attributes.bools, self.rows)

    @property
    def density(self) -> float:
        return sum([sum(row.bools()) for row in self.rows]) / \
            (self.shape[0] * self.shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.rows), len(self.columns))

    def filter(self, items: list = None, axis: int = 0) -> Iterator[Type[BitSet]]:
        if axis == 0:
            target = self.rows
            data_class = self.Objects
        elif axis == 1:
            target = self.columns
            data_class = self.Attributes
        else:
            ValueError("Axis should be 0 or 1.")

        return compress(target, data_class(items).bools())

    def __arrow_operator(self, input_set: Type[BitSet], data: tuple, ResultClass) -> Type[BitSet]:
        """Experimental implementation based on:
        https://stackoverflow.com/q/63917579/3456664"""

        result = ResultClass.supremum
        i = 0

        while i < len(data):
            if input_set:
                trailing_zeros = (input_set & -input_set).bit_length() - 1
                if trailing_zeros:
                    input_set >>= trailing_zeros
                    i += trailing_zeros
                else:
                    result &= data[i]
                    input_set >>= 1
                    i += 1
            else:
                break

        return ResultClass.fromint(result)

    def up(self, objects: Type[BitSet]) -> Type[BitSet]:
        return self.__arrow_operator(objects, self.rows, self.Attributes)

    def down(self, attributes: Type[BitSet]) -> Type[BitSet]:
        return self.__arrow_operator(attributes, self.columns, self.Objects)

    # def index_of_tuple(self, t: tuple) -> int:
    #     return (t[0] * len(self.columns)) + t[1]
    #
    # def tuple_for_index(self, index: int):
    #     row = int(index / len(self.columns))
    #     col = int(index - (row * len(self.columns)))
    #     return row, col
    #
    # def tuples(self) -> Type[BitSet]:
    #     tuples = self.CartesianProduct.infimum
    #
    #     for obj in range(len(self.rows)):
    #         for attribute, is_presented in enumerate(self.rows[obj].bools()):
    #             if is_presented:
    #                 tuples |= 2 ** self.index_of_tuple((obj, attribute))
    #     return self.CartesianProduct.fromint(tuples)
    #
    # def tuples_for_concept(self, concept) -> Type[BitSet]:
    #     cartesian_product = self.CartesianProduct.infimum
    #
    #     for obj, attr in itertools.product(concept.extent.iter_set(), concept.intent.iter_set()):
    #         cartesian_product |= 2 ** self.index_of_tuple((obj, attr))
    #
    #     return self.CartesianProduct.fromint(cartesian_product)