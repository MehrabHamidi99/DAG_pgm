import pickle
import networkx as nx
import os
import numpy as np


def dump(obj, exp_path, name, txt=False):
    if not txt:
        with open(os.path.join(exp_path, name + ".pkl"), "wb") as f:
            pickle.dump(obj, f)
    else:
        with open(os.path.join(exp_path, name + ".txt"), "w") as f:
            f.write(str(obj))


def get_intersection_edges(mat1,mat2):
    return mat1*mat2


