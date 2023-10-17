"""
Note by Amal Alghamdi: This code is copied from the project report: Depth 
Dependency in Electrical Impedance Tomography with the Complete 
Electrode Model by Anders Eltved and Nikolaj Vestbjerg Christensen (Appendix D.5). Some 
modifications are made.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
from dolfin import *
from mshr import *

class  EITFenics:
    def __init__(self, L=32, F=50):
        self.L = L
        self.F = F
        self._create_mesh()
        self._build_subdomains()
        self.V, self.dS = build_spaces(self.mesh, L, self.subdomains)

    def _create_mesh(self):
        R = 1  # radius of circle
        n = 300 # number of polygons to approximate circle
        self.mesh = generate_mesh(Circle(Point(0, 0), R, n), self.F)  # generate mesh
        print("Mesh created with %d elements" % self.mesh.num_entities(2))
    def _build_subdomains(self):
        L = self.L
        e_l = np.pi / L
        d_e = 2*np.pi / L - e_l

        # Define subdomain mesh
        self.subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 0)

        # Define subdomains
        def twopiarctan(x):
            val = np.arctan2(x[1], x[0])
            if val < 0:
                val = val+2*np.pi
            return val

        class e(SubDomain):
            def inside(self, x, on_boundary):
                theta = twopiarctan(x)
                # print "theta inside", theta
                if theta1 > theta2:
                    return on_boundary and ((theta >= 0
                                            and theta <= theta2) or (theta >= theta1
                                                                    and theta <= 2*np.pi))
                return on_boundary and theta >= theta1 \
                    and theta <= theta2
                # return  theta>=theta1 and theta<=theta2

        for i in range(1, L+1):
            shift_theta = np.pi/2 - np.pi/(2*L)
            # print "shift_theta", shift_theta
            # print L
            theta1 = np.mod((i - 1) * (e_l+d_e) + shift_theta, 2*np.pi)
            theta2 = np.mod(theta1+e_l, 2*np.pi)
            # print i
            # print theta1
            # print theta2
            e1 = e()  # create instance
            e1 .mark(self.subdomains, i)  # mark subdomains
            xdmf = XDMFFile("subdomains.xdmf")
            xdmf.write(self.subdomains)
    def _create_inclusion(self, phantom):
        high_conductivity = 1e1
        low_conductivity = 1e-2
        background_conductivity = 0.8
        # Define conductivity
        phantom_float = np.zeros(phantom.shape)
        phantom_float[phantom == 0] = background_conductivity
        phantom_float[np.isclose(phantom, 1, rtol=0.01)] = low_conductivity
        phantom_float[phantom == 2] = high_conductivity

        plt.figure()
        im = plt.imshow(phantom_float)
        plt.colorbar(im)  # norm= 'log'
        plt.savefig("phantom.png")

        self.inclusion = Inclusion(phantom_float, degree=0)
    def solve_forward(self, phantom, injection_patterns, num_inj_tested, z=1e-6):
        self._create_inclusion(phantom)
        L = self.L

        # Define vector of contact impedance
        # z = 10e-6  # 0.1 # Impedence
        Z = []
        for i in range(self.L):
            Z.append(z)

        # Define H1 room
        H1 = FunctionSpace(self.mesh, 'CG', 1)

        # Loop over current patterns
        num_inj = 76  # Number of injection pattern
        # num_inj_tested = 76

        B = build_b(self.inclusion, Z, self.V, self.dS, L)
        Q = np.zeros((L, num_inj))
        Diff = np.zeros((L-1, num_inj))
        q_list = []

        for i in range(num_inj)[:num_inj_tested]:
            print("injection pattern"+str(i))
            Q_i, q = solver(injection_patterns[:, i], B, self.V, self.dS, L)
            q_list.append(q)

            Q[:, i] = Q_i
            Diff[:, i] = np.diff(Q_i)

        Uel_sim = -Diff.flatten(order='F')
        return Uel_sim, Q, q_list

    def solve_P(self, y_list, sigma_perturb, background_sigma, z=1e-6):
        L = self.L

        # Define vector of contact impedance
        # z = 10e-6  # 0.1 # Impedence
        Z = []
        for i in range(self.L):
            Z.append(z)

        # Define H1 room
        H1 = FunctionSpace(self.mesh, 'CG', 1)

        B = build_b(background_sigma, Z, self.V, self.dS, L)

        v = TestFunction(self.V)

        w_list = []
        for y in y_list:

            f = -sigma_perturb * inner(nabla_grad(y[L]), nabla_grad(v[L])) * dx 

            rhs = assemble(f)

            # Compute solution
            w = Function(self.V)
            solve(B, w.vector(), rhs)
            w_list.append(w)

        return w_list


class Inclusion(UserExpression):
    def __init__(self, phantom, **kwargs):
        super().__init__(**kwargs)
        x_grid = np.linspace(-1, 1, 256)
        y_grid = np.linspace(-1, 1, 256)
        self._interpolater = RegularGridInterpolator(
            (x_grid, y_grid), phantom, method="nearest")

    def eval(self, values, x):
        values[0] = self._interpolater([x[0], x[1]])

def build_subdomains(L, mesh):
    def twopiarctan(x):
        val = np.arctan2(x[1], x[0])
        if val < 0:
            val = val+2*np.pi
        return val

    e_l = np.pi / L
    d_e = 2*np.pi / L - e_l

    # Define subdomain mesh
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

   # Define subdomains
    class e(SubDomain):
        def inside(self, x, on_boundary):
            theta = twopiarctan(x)
            # print "theta inside", theta
            if theta1 > theta2:
                return on_boundary and ((theta >= 0
                                         and theta <= theta2) or (theta >= theta1
                                                                  and theta <= 2*np.pi))
            return on_boundary and theta >= theta1 \
                and theta <= theta2
            # return  theta>=theta1 and theta<=theta2

    for i in range(1, L+1):
        shift_theta = np.pi/2 - np.pi/(2*L)
        # print "shift_theta", shift_theta
        # print L
        theta1 = np.mod((i - 1) * (e_l+d_e) + shift_theta, 2*np.pi)
        theta2 = np.mod(theta1+e_l, 2*np.pi)
        # print i
        # print theta1
        # print theta2
        e1 = e()  # create instance
        e1 .mark(subdomains, i)  # mark subdomain

    return subdomains

def build_spaces(mesh, L, subdomains):
    R = FunctionSpace(mesh, "R", 0)
    H1 = FunctionSpace(mesh, "CG", 1)

    spacelist = None

    for i in range(1, L+1):

        if i == 1:
            spacelist = R.ufl_element()
        else:
            spacelist *= R.ufl_element()

    spacelist *= H1.ufl_element()
    spacelist *= R.ufl_element()

    # Create function space
    V = FunctionSpace(mesh, spacelist)

    # Define new measures associated with the boundaries
    dS = Measure('ds', domain=mesh)[subdomains]

    return V, dS


def build_b(sigma, Z, V, dS, L):

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    B = sigma * inner(nabla_grad(u[L]), nabla_grad(v[L])) * dx

    for i in range(L):
        B += 1/Z[i] * (u[L]-u[i])*(v[L]-v[i]) * dS(i + 1)
        #TODO: check if this is correct for P operator
        B += (v[L+1]*u[i] / assemble(1*dS(i+1))) * dS(i+1)
        B += (u[L+1]*v[i] / assemble(1*dS(i+1))) * dS(i+1)

    return assemble(B)


def solver(I, B, V, dS, L):  # sigma ,L, I , Z ,mesh, subdomains )
   # def 2 pi function

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    f = 0*dS(1)

    for i in range(L):
        f += (I[i] * v[i] / assemble(1*dS(i+1))) * dS(i+1)

    rhs = assemble(f)

    # Compute solution
    q = Function(V)
    solve(B, q.vector(), rhs)

    Q = q.vector().get_local()[:L]

    return Q, q
