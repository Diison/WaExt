# -*- coding: utf-8 -*-

"""Get triples from the PPI5K dataset."""

from email.mime import base
import pathlib

from docdata import parse_docdata
from pykeen.triples import utils

from pykeen.datasets.base import LazyDataset

from pykeen.triples import CoreTriplesFactory, TriplesFactory

import torch


# LabeledTriples:  np.ndarray
# MappedTriples:  torch.LongTensor
# EntityMapping:  Mapping[str, int]
# RelationMapping:  Mapping[str, int]


__all__ = [
    "PPI5K_TRAIN_PATH",
    "PPI5K_TEST_PATH",
    "PPI5K_VALIDATE_PATH",
    "PPI5K",
]

HERE = pathlib.Path(__file__).resolve().parent

PPI5K_TRAIN_PATH = HERE.joinpath("train.tsv")
PPI5K_TEST_PATH = HERE.joinpath("test.tsv")
PPI5K_VALIDATE_PATH = HERE.joinpath("val.tsv")

@parse_docdata
class PPI5K(LazyDataset):

    def __init__(self, create_inverse_triples: bool = False, th=0, pct=1, **kwargs):
        self.create_inverse_triples = create_inverse_triples
        self.ents=[]
        self.rels=[]
        self.index_ents = {}  # {string: index}
        self.index_rels = {}  # {string: index}
        self.th = th  # {string: index}
        self.pct = pct  # {string: index}

        print("The toolkit is importing datasets.\n")
                
        self.train_quad = self.load_quadruples(PPI5K_TRAIN_PATH)
        self.test_quad = self.load_quadruples(PPI5K_TEST_PATH)
        self.val_quad = self.load_quadruples(PPI5K_VALIDATE_PATH)
        
        self.quads = {**self.train_quad, **self.test_quad, **self.val_quad}
                
        self._load()
        self._load_validation()

    def load_quadruples(self, filename, splitter='\t', line_end='\n'):
        '''Load the dataset'''
        quadruples = {}
        last_e = len(self.ents)-1
        last_r = len(self.rels)-1

        for line in open(filename):
            line_y = line
            line = line.rstrip(line_end).split(splitter)
            if self.index_ents.get(line[0]) == None:
                self.ents.append(line[0])
                last_e += 1
                self.index_ents[line[0]] = last_e
            if self.index_ents.get(line[2]) == None:
                self.ents.append(line[2])
                last_e += 1
                self.index_ents[line[2]] = last_e
            if self.index_rels.get(line[1]) == None:
                self.rels.append(line[1])
                last_r += 1
                self.index_rels[line[1]] = last_r
            h = self.index_ents[line[0]]
            r = self.index_rels[line[1]]
            t = self.index_ents[line[2]]
            w = float(line[3])
            # if (h, r, t) in quadruples.keys():
            #     print(line_y)
            quadruples[(h, r, t)]=w
            
        if self.pct>=1 or self.pct<=-1:
            pass
        elif self.pct>0:
            q=sorted(quadruples.items(), key = lambda kv:kv[1])
            q_pct=q[-int(len(q)*self.pct):]
            print("q_pct: ", q_pct[0])
            quadruples=dict(q_pct)
        elif self.pct<0:
            q=sorted(quadruples.items(), key = lambda kv:kv[1])
            q_pct=q[:-int(len(q)*self.pct)]
            quadruples=dict(q_pct)
            print("q_pct: ", q_pct[0])

        return quadruples

    def _load(self) -> None:
        self._training = TriplesFactory(mapped_triples = torch.LongTensor(tuple(self.train_quad.keys())), entity_to_id = self.index_ents, relation_to_id = self.index_rels, create_inverse_triples=self.create_inverse_triples)
        self._testing = TriplesFactory(mapped_triples = torch.LongTensor(tuple(self.test_quad.keys())), entity_to_id = self.index_ents, relation_to_id = self.index_rels, create_inverse_triples=False)

    def _load_validation(self) -> None:
        self._validation = TriplesFactory(mapped_triples = torch.LongTensor(tuple(self.val_quad.keys())), entity_to_id = self.index_ents, relation_to_id = self.index_rels, create_inverse_triples=False)



    def __repr__(self) -> str:  # noqa: D105
        return (
            f'{self.__class__.__name__}(training_path="{PPI5K_TRAIN_PATH}", testing_path="{PPI5K_TEST_PATH}",'
            f' validation_path="{PPI5K_VALIDATE_PATH}")'
        )

def load_ppi5k(th=0, pct=100, create_inverse_triples=False):
    return PPI5K(th=th, pct=pct, create_inverse_triples=create_inverse_triples)

if __name__ == "__main__":
    PPI5K().summarize()
