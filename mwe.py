import numpy as np
import dolfinx.mesh as mesh
from dolfinx.io import XDMFFile
from mpi4py import MPI

infile = XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r")
domain = infile.read_mesh(name="notched_sample")
infile.close()

def bottom(x):
    return np.isclose(x[1], -0.5)

bottom_facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, bottom)
