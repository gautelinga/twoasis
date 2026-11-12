"""What the dual mesh (twoasis/common/dual.py) adds to a backend's names.

Measures (dx, and those Measure makes) move an integral that mixes the flow mesh and
the phase field's mesh to the latter as it is formed (`dual.retarget`); on one mesh
they are UFL's own. A DirichletBC given facet markers of the flow mesh for a space on
the phase field's mesh acts through the markers' refinement there.
"""


def measure_class(ufl):
    """The Measure class over `ufl` (the UFL the backend's forms are written in)."""

    class Measure(ufl.Measure):
        """ufl.Measure whose products with an integrand go through dual.retarget, and
        whose reconstructions (ds(1), dx(domain=mesh), ...) are of this class too."""

        __slots__ = ()

        def reconstruct(self, *args, **kwargs):
            m = ufl.Measure.reconstruct(self, *args, **kwargs)
            return type(self)(m.integral_type(), domain=m.ufl_domain(),
                              subdomain_id=m.subdomain_id(), metadata=m.metadata(),
                              subdomain_data=m.subdomain_data())

        def __rmul__(self, integrand):
            from twoasis.common import dual
            if dual.current() is not None:
                moved = dual.retarget(self, integrand)
                if moved is not None:
                    self, integrand = moved
            return ufl.Measure.__rmul__(self, integrand)

    return Measure


def dirichlet_class(base, space_mesh):
    """The backend's DirichletBC `base`, its markers resolved for the space's mesh
    (`dual.on_space_mesh`)."""

    class DirichletBC(base):
        __doc__ = base.__doc__

        def __init__(self, *args, **kwargs):
            if len(args) >= 3:
                from twoasis.common.dual import on_space_mesh
                args = args[:2] + (on_space_mesh(space_mesh(args[0]), args[2]),) + args[3:]
            super().__init__(*args, **kwargs)

    return DirichletBC
