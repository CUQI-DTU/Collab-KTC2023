
#%% 
"""
Note by Amal Alghamdi: This code is copied from the project report: Depth 
Dependency in Electrical Impedance Tomography with the Complete 
Electrode Model by Anders Eltved and Nikolaj Vestbjerg Christensen (Appendix D.5). Some 
modifications are made to make it run against FEniCS 2016.2.

Created on Mon Mar 9 05:17:48 2015
@author : ec0di
"""
import numpy as np 
from dolfin import *
from mshr import *
import dolfin as dl
import matplotlib.pyplot as plt

Z=0.1; 
I=[2,-2]
x0 =0; y0 =0; x1 =1; y1=1 
L=2
mesh = RectangleMesh(Point(x0 , y0) , Point(x1 , y1), 10, 10) 

class sigma_fun(Expression):
    def eval(self ,values ,x): 
        values [0]=1

sigma = sigma_fun(degree=2)

class Left (SubDomain) :
    def inside(self , x, on_boundary): 
        return near(x[0] , 0.0)
class Right(SubDomain):
    def inside(self , x, on_boundary): 
        return near(x[0] , 1.0)

left = Left() 
right = Right()

boundaries = FacetFunction("size_t", mesh) 
boundaries.set_all (0)
left.mark(boundaries , 1)
right.mark(boundaries , 2)

dS = Measure('ds', domain=mesh)[boundaries]

R = FunctionSpace (mesh , "R" ,0)
H1 = FunctionSpace(mesh,"CG",1)

#R = FiniteElement("R", mesh.ufl_cell(), 0)
#H1 = FiniteElement("CG", mesh.ufl_cell(), 1)

mixedspaces=R.ufl_element()*R.ufl_element()*H1.ufl_element()*R.ufl_element()

V = FunctionSpace(mesh, mixedspaces)
u = TrialFunction(V)
v = TestFunction(V)

f = 0*dS(1)
B = sigma*inner(nabla_grad(u[L]) , nabla_grad(v[L]))*dx
for i in range(L):
    B += 1/Z*(u[L]-u[i])*(v[L]-v[i])*dS(i+1)
    B += (v[L+1]*u[i]/assemble(1*dS(i+1)))*dS(i+1) 
    B += (u[L+1]*v[i]/assemble(1*dS(i+1)))*dS(i+1)
    f += (I[i]*v[i]/assemble(1*dS(i+1)))*dS(i+1)
          
q = Function(V)
solve(B==f, q)

Q=q.split(deepcopy=True)
U=[]

for i in range(2):
    U.append(Q[ i ])

u = Q[1]#Q[2]

for i in range(2):
    print("U"+str(i+1)+": "+str(U[i].compute_vertex_values()[0]))

#%%
im = plot(q[2])
plt.colorbar(im)
interactive()
plt.show()
plt.savefig("squaredomainCEM.png")
