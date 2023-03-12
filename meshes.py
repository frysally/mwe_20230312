import gmsh
import dolfinx
from dolfinx.io.gmshio import extract_geometry, extract_topology_and_markers, ufl_mesh
from dolfinx.mesh import create_mesh
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio

def generate_mesh_with_crack(Lx=1, Ly=1, Lcrack=0.3, lc=0.015, refinement_ratio=10, dist_min=0.05, dist_max=0.2):
    # For further documentation see
    # - gmsh tutorials, e.g. see https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorial/python/t10.py 
    # - dolfinx-gmsh interface https://github.com/FEniCS/dolfinx/blob/master/python/demo/gmsh/demo_gmsh.py
    #
    gmsh.initialize()
    model = gmsh.model()
    model.add("Rectangle")
    model.setCurrent("Rectangle")

    p1 = model.geo.addPoint(0.0, lc/2, 0, lc)
    p2 = model.geo.addPoint(Lcrack, lc/2, 0, lc)
    p23 = model.geo.addPoint(Lcrack, 0, 0, lc)
    p3 = model.geo.addPoint(Lcrack, -lc/2, 0, lc)
    p4 = model.geo.addPoint(0.0, -lc/2, 0, lc)
    p5 = model.geo.addPoint(0.0, -Ly, 0, lc)
    p6 = model.geo.addPoint(Lx, -Ly, 0, lc)
    p7 = model.geo.addPoint(Lx, Ly, 0, lc)
    p8 = model.geo.addPoint(0.0, Ly, 0, lc)

    l1 = model.geo.addLine(p1, p2)
    l2 = model.geo.addCircleArc(p2, p23, p3, tag=-1, nx=0, ny=0, nz=-1)
    l3 = model.geo.addLine(p3, p4)
    l4 = model.geo.addLine(p4, p5)
    l5 = model.geo.addLine(p5, p6)
    l6 = model.geo.addLine(p6, p7)
    l7 = model.geo.addLine(p7, p8)
    l8 = model.geo.addLine(p8, p1)
    cloop1 = model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8])
    surface_1 = model.geo.addPlaneSurface([cloop1])

    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumbers(1, "NodesList", [p23])
    #model.mesh.field.setNumber(1, "NNodesByEdge", 100)
    #model.mesh.field.setNumbers(1, "EdgesList", [2])


    #
    # SizeMax -                     /------------------
    #                              /
    #                             /
    #                            /
    # SizeMin -o----------------/
    #          |                |    |
    #        Point         DistMin  DistMax

    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "InField", 1)
    model.mesh.field.setNumber(2, "LcMin", lc / refinement_ratio)
    model.mesh.field.setNumber(2, 'LcMax', lc)
    model.mesh.field.setNumber(2, 'DistMin', dist_min)
    model.mesh.field.setNumber(2, 'DistMax', dist_max)
    model.mesh.field.setAsBackgroundMesh(2)

    model.geo.synchronize()
    surface_entities = [model[1] for model in model.getEntities(2)]
    model.addPhysicalGroup(2, surface_entities, tag = 5)
    model.setPhysicalName(2, 2, "Rectangle surface")
    model.mesh.generate(2)
    # get mesh into fenics
    x = extract_geometry(model, name = "Rectangle")[:,0:2]
    gmsh_cell_id = model.mesh.getElementType("triangle", 1)
    topologies = extract_topology_and_markers(model, "Rectangle")
    cells = topologies[gmsh_cell_id]["topology"]
    gmsh_facet_id = model.mesh.getElementType("triangle", 1)
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh(gmsh_cell_id, 2), 
                       partitioner=dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet))

    # Create a DOLFINx mesh (same mesh on each rank)
    msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
    msh.name = "notched_sample"
    cell_markers.name = f"{msh.name}_cells"
    facet_markers.name = f"{msh.name}_facets"

    with XDMFFile(msh.comm, f"mesh.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_meshtags(cell_markers)
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
        file.write_meshtags(facet_markers)

    return mesh

if __name__ == '__main__':

    L = 1.0
    H = 0.5
    Lcrack = 0.3
    lc = 0.05 # Characteristic length of the mesh
    refinement_ratio = 10 # how much to refine at the tip zone
    dist_min = 0.1 # radius of tip zone
    dist_max = 0.2 # radius of the transition zone
    mesh = generate_mesh_with_crack(Lcrack=Lcrack,
                                    Ly=H, lc=lc, # characteristic length of the mesh
                                    refinement_ratio=refinement_ratio,
                                    dist_min = dist_min, # radius of tip zone,
                                    dist_max = dist_max # radius of the transition zone)
    
    )
