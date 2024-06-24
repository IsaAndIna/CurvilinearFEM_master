import numpy as np
import matplotlib.pyplot as plt
import torch 

from scipy.sparse import coo_matrix

from scipy.sparse.linalg import spsolve


### Solver of cubic equation
### https://github.com/shril/CubicEquationSolver/blob/master/CubicEquationSolver.py
import math


def solve(a, b, c, d):

    if (a == 0 and b == 0):                     # Case for handling Liner Equation
        return np.array([(-d * 1.0) / c])                 # Returning linear root as numpy array.

    elif (a == 0):                              # Case for handling Quadratic Equations

        D = c * c - 4.0 * b * d                       # Helper Temporary Variable
        if D >= 0:
            D = math.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
        else:
            D = math.sqrt(-D)
            x1 = (-c + D * 1j) / (2.0 * b)
            x2 = (-c - D * 1j) / (2.0 * b)
            
        return np.array([x1, x2])               # Returning Quadratic Roots as numpy array.

    f = findF(a, b, c)                          # Helper Temporary Variable
    g = findG(a, b, c, d)                       # Helper Temporary Variable
    h = findH(g, f)                             # Helper Temporary Variable

    if f == 0 and g == 0 and h == 0:            # All 3 Roots are Real and Equal
        if (d / a) >= 0:
            x = (d / (1.0 * a)) ** (1 / 3.0) * -1
        else:
            x = (-d / (1.0 * a)) ** (1 / 3.0)
        return np.array([x, x, x])              # Returning Equal Roots as numpy array.

    elif h <= 0:                                # All 3 roots are Real

        i = math.sqrt(((g ** 2.0) / 4.0) - h)   # Helper Temporary Variable
        j = i ** (1 / 3.0)                      # Helper Temporary Variable
        k = math.acos(-(g / (2 * i)))           # Helper Temporary Variable
        L = j * -1                              # Helper Temporary Variable
        M = math.cos(k / 3.0)                   # Helper Temporary Variable
        N = math.sqrt(3) * math.sin(k / 3.0)    # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

        x1 = 2 * j * math.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return np.array([x1, x2, x3])           # Returning Real Roots as numpy array.

    elif h > 0:                                 # One Real Root and two Complex Roots
        R = -(g / 2.0) + math.sqrt(h)           # Helper Temporary Variable
        if R >= 0:
            S = R ** (1 / 3.0)                  # Helper Temporary Variable
        else:
            S = (-R) ** (1 / 3.0) * -1          # Helper Temporary Variable
        T = -(g / 2.0) - math.sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))                # Helper Temporary Variable
        else:
            U = ((-T) ** (1 / 3.0)) * -1        # Helper Temporary Variable

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

        return np.array([x1, x2, x3])           # Returning One Real Root and two Complex Roots as numpy array.

# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)



class custom_function_1d:
    def __init__(self, function, prime, prime2, inverse):
        assert function(0)==0, "Function must be zero at zero"
        assert function(1)==1, "Function must be one at one"
        assert prime(0)>=0, "Prime must be positive at zero"
        assert prime(1)>=0, "Prime must be positive at one"
        assert inverse(0)==0, "Inverse must be zero at zero"
        assert inverse(1)==1, "Inverse must be one at one"
        for i in np.linspace(0,1,100):
            assert (function(inverse(i))-i)**2<0.001, "The inverse is not correct"
        self.function = function
        self.prime = prime
        self.prime2 = prime2
        self.inverse = inverse

class custom_function_2d:
    def __init__(self, function, prime, inverse,name : str=""):
        for st in [(0,0),(0,1),(1,0),(1,1)]:
            f=function(st)
            inv=inverse(st)
            assert type(f) is np.ndarray, "Function must turn list."
            assert len(f) ==2, "Function must turn list of size 2."
            assert type(inv) is np.ndarray, "The inverse must turn tuple."
            assert len(inv) ==2, "The inverse must turn tuple of size 2."
            for i in range(2):
                assert f[i]==st[i], f"Function must be {st} at {st}, but it's turning {f}."
                assert inv[i]==st[i], f"The inverse must be {st} at {st}, but it's turning {inv}."
        for st in [(0,0),(0,1),(1,0),(1,1)]:
            p=prime(st)
            assert type(p) is np.ndarray, "The prime must turn 2x2 matrix (list of list)"
            assert len(p)==2, "The prime must turn 2x2 matrix (list of list)"
            assert len(p[0])==2, "The prime must turn 2x2 matrix (list of list)"
        for i in np.linspace(0,1,5):
            for j in np.linspace(0,1,5):
                f_inv=inverse(function((i,j)))
                assert (f_inv[0]-i)**2<0.001, "The inverse is not correct"
                assert (f_inv[1]-j)**2<0.001, "The inverse is not correct"
        
        self.function = function
        self.prime = prime
        self.inverse = inverse
        self.name = name


def generate_alpha (name:str):
    if name=="identity":
        ### s=x, t=y
        alpha=custom_function_2d(lambda xy:  np.array(xy),
                        lambda xy:  np.array([[1,0],[0,1]]), 
                        lambda st:  np.array( st), "identity")
    
    elif name=="cube":
        ### s=x^3, t=y^3
        alpha=custom_function_2d(lambda xy:  np.array([xy[0]**3,xy[1]**3]),
                        lambda xy:  np.array([[3*xy[0]**2,0],[0,3*xy[1]**2]]), 
                        lambda st:  np.array( [st[0]**(1/3),st[1]**(1/3)]), "cube")
    
    elif name=="sqrt-cube":
        ### s=sqrt(x), t=y^3
        alpha=custom_function_2d(lambda xy:  np.array([np.sqrt(xy[0]),xy[1]**3]),
                        lambda xy:  np.array([[0.5/np.sqrt(xy[0]),0],[0,3*xy[1]**2]]), 
                        lambda st:  np.array( [st[0]**2,st[1]**(1/3)]), "sqrt-cube")
    
    elif name=="square":
        ### s=x^2, t=y^2
        alpha=custom_function_2d(lambda xy:  np.array([xy[0]**2,xy[1]**2]),
                        lambda xy:  np.array([[2*xy[0],0],[0,2*xy[1]]]), 
                        lambda st: np.sqrt(np.array( st)), "square")
    
    elif name=="sqrt":
        ### s=sqrt(x), t=sqrt(y)
        alpha=custom_function_2d(lambda xy:  np.sqrt(np.array(xy)),
                        lambda xy:  np.array([[0.5/np.sqrt(xy[0]),0],[0,0.5/np.sqrt(xy[1])]]), 
                        lambda st: np.array( st)**2, "sqrt")
    
    elif name=="sin":
        ### s=sin(πx/2), t=sin(πy/2)
        alpha=custom_function_2d(lambda xy:  np.sin(np.array(xy)*np.pi*0.5),
                            lambda xy: np.array([[np.cos(xy[0]*np.pi*0.5)*np.pi*0.5,0],
                                                 [0,np.cos(xy[1]*np.pi*0.5)*np.pi*0.5]]) , 
                            lambda st:  np.arcsin(np.array(st))/np.pi*2, "sin")
    
    elif name=="linearsquare":
        def xy2st(x,y):
            ### s=yxx+(1-y)x
            ### t=xyy+(1-x)y
            xy=x*y
            return np.array([xy*(x-1)+x,xy*(y-1)+y])

        def xy2st_derivative(x,y):
            ### ∂s/∂x, ∂t/∂x
            ### ∂s/∂y, ∂t/∂y
            xy=x*y
            return np.array([[2*xy+1-y,y*(y-1)],[x*(x-1),2*xy+1-x]])
        
        def st2xy(s,t):
            alpha=s+t
            beta=s-t
            ### alpha-2=(xy+1)(x+y-2) = (C+1)(A-2)
            ### beta = (xy+1)(x-y) = (C+1)B
            ### A^2-B^2 = 4C
            ### thus, 4C(C+1)^2 = (alpha +2C)^2 - beta^2
            ### thus, 4C^3+8C^2+4C = alpha^2+4alphaC+4C^2 - beta^2
            ### thus, 4C^3+4C^2+4(1-alpha)C + beta^2 - alpha^2 = 0
            ### thus, C^3+C^2+(1-alpha)C + (beta^2-alpha^2)*0.25 = 0
            a,b,c,d=1,1,1-alpha,(beta**2-alpha**2)*0.25
            Cs=solve(a,b,c,d)### solve a cubic equation to abtain the value of C (real number)
            C=Cs[0].real### take the first solution (always real)
            if C==-1:### There are three real solutions
                C=0### when Cs[0]==-1, the second and third solutions are zero.
            A=(alpha-2)/(C+1) +2
            B=beta/(C+1)
            return np.array([round((A+B)/2,8),round((A-B)/2,8)])
        
        alpha=custom_function_2d(lambda xy:  xy2st(xy[0],xy[1]),
                          lambda xy:  xy2st_derivative(xy[0],xy[1]),
                          lambda st: st2xy(st[0],st[1]), "linearsquare")
    else:
        raise ValueError("The alpha is not defined.")
    return alpha

def save_coor_shape_figure(alpha:custom_function_2d,N_divisions:int=5):
    d:float =1/N_divisions
    xy_collect=np.zeros((N_divisions+1,N_divisions+1,2))
    for i,si in enumerate(np.arange(0,1+d,d)):
        for j,ti in enumerate(np.arange(0,1+d,d)):
            xy_collect[i,j]=alpha.inverse([si,ti])
    
    for i in range(N_divisions):
        for j in range(N_divisions):
            # draw a line from point (i,j) to (i+1,j) and (i,j) to (i,j+1)
            plt.plot([xy_collect[i,j,0],xy_collect[i+1,j,0]],[xy_collect[i,j,1],xy_collect[i+1,j,1]],c='black',ls='--')
            plt.plot([xy_collect[i,j,0],xy_collect[i,j+1,0]],[xy_collect[i,j,1],xy_collect[i,j+1,1]],c='black')
    for i in range(N_divisions):
        plt.plot([xy_collect[i,N_divisions,0],xy_collect[i+1,N_divisions,0]],[xy_collect[i,N_divisions,1],xy_collect[i+1,N_divisions,1]],c='black',ls='--')
        plt.plot([xy_collect[N_divisions,i,0],xy_collect[N_divisions,i+1,0]],[xy_collect[N_divisions,i,1],xy_collect[N_divisions,i+1,1]],c='black')

    # set the title 
    plt.title(f'Coordinate system "{alpha.name}" on cartesian')
    #fix the aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    # set the limit [0,1]^2
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(f'coor_shape_{alpha.name}.png')
    plt.show()

def main_original(N_divisions:int=5,alpha=None):
    N_nodes=(N_divisions+1)**2
    f_analytical=lambda x,y: x**2+y**2

    if type(alpha)==str:
        alpha=generate_alpha(alpha)

    elif alpha is None:
        alpha=custom_function_2d(lambda xy:  np.array(xy),
                            lambda xy:  np.array([[1,0],[0,1]]), 
                            lambda st:  np.array( st), "identity")
    
    d: float=1/N_divisions
    # gaussian points to iterate over [0,1] 
    # gauss_points = np.array([0.11270167, 0.5, 0.88729833])
    # gauss_weights = np.array([5/9, 8/9, 5/9])*0.5
    gauss_points = (np.array([-np.sqrt(5+2*np.sqrt(10/7))/3,-np.sqrt(5-2*np.sqrt(10/7))/3, 0, 
                                np.sqrt(5-2*np.sqrt(10/7))/3,np.sqrt(5+2*np.sqrt(10/7))/3])+1.0)*0.5
    gauss_weights = np.array([(322-13*np.sqrt(70))/900, (322+13*np.sqrt(70))/900, 128/225, 
                            (322+13*np.sqrt(70))/900, (322-13*np.sqrt(70))/900])*0.5
    N_gauss_points = gauss_points.size

    # Shape functions : s,t\in[0,1]
    N_nodes_e=4
    N=lambda s,t: np.array([(1-s)*(1-t),  s*(1-t),(1-s)*t,  s*t])
    N_derivatives=lambda s,t: np.array( [ [t-1,s-1],[1-t,-s],[-t,1-s],[t,s]])


    K=np.zeros((N_nodes,N_nodes))
    K_row=np.zeros(N_nodes)

    f=np.zeros(N_nodes)
    K_temp = np.zeros((N_nodes_e, N_nodes_e))
    f_temp = np.zeros(N_nodes_e)
    for i in range(N_divisions): # i-th division on s axis
        s=np.array([i*d,(i+1)*d])
        for j in range(N_divisions): # j-th division on t axis
            t=np.array([j*d,(j+1)*d])
            indeces=[j*(N_divisions+1)+i,
                    j*(N_divisions+1)+i+1,
                    (j+1)*(N_divisions+1)+i,
                    (j+1)*(N_divisions+1)+i+1]
            
            for k in range(N_gauss_points):# k-th gauss point over t
                t_temp=(j+gauss_points[k])*d
                for l in range(N_gauss_points):# l-th gauss point over s
                    s_temp=(i+gauss_points[l])*d
                    J_temp=alpha.prime(alpha.inverse([s_temp, t_temp]) )
                    det_J_temp=np.abs(np.linalg.det(J_temp))
                    JJ=J_temp.T@J_temp
                    N_temp=N(gauss_points[l],gauss_points[k])
                    N_derivatives_temp = N_derivatives(gauss_points[l],gauss_points[k])
                    for m in range(N_nodes_e):
                        for n in range(N_nodes_e):
                            K_temp[m,n]=-N_derivatives_temp[m]@JJ@N_derivatives_temp[n]/4
                        f_temp[m]=N_temp[m]
                    K_temp*=1/det_J_temp
                    f_temp*=d*d/det_J_temp
                    
                    for m,index in enumerate(indeces):
                        for m2,index2 in enumerate(indeces):
                            K[index,index2] += K_temp[m,m2]*gauss_weights[k]*gauss_weights[l]
                        #K[index,indeces] += K_temp[m,:]*gauss_weights[k]*gauss_weights[l]
                    
                    for m,index in enumerate(indeces):
                        f[index] += f_temp[m]*gauss_weights[k]*gauss_weights[l]#*d*d*4##要：再調査
                    #f[indeces] += f_temp*gauss_weights[k]*gauss_weights[l]

    #Apply the Boundary condition.
    BC_dict={}
    for i in range(N_divisions+1):
        xy=alpha.inverse([i*d,0])
        BC_dict[i]=f_analytical(xy[0],xy[1])

        xy=alpha.inverse([0,i*d])
        BC_dict[i*(N_divisions+1)+0]=f_analytical(xy[0],xy[1])

        xy=alpha.inverse([1,i*d])
        BC_dict[(i+1)*(N_divisions+1)-1]=f_analytical(xy[0],xy[1])

        xy=alpha.inverse([i*d,1])
        BC_dict[i+N_divisions*(N_divisions+1)]=f_analytical(xy[0],xy[1])

    no_BC_list=[i for i in range(N_nodes) if i not in BC_dict.keys() ]
    A=np.zeros((len(no_BC_list),len(no_BC_list)))
    b=np.zeros(len(no_BC_list))
    for i,index_i in enumerate(no_BC_list):
        b[i]=f[index_i]
        for j,index_j in enumerate(no_BC_list):
            A[i,j]=K[index_i,index_j]
        for k,val in BC_dict.items():
            b[i]-=K[index_i,k]*val
    



    x=np.linalg.solve(A,b)
    x_numerical=np.zeros(N_nodes)
    for i,index in enumerate(no_BC_list):
        x_numerical[index]=x[i]
    for k,val in BC_dict.items():
        x_numerical[k]=val
    #t_vals=[ (i,j)  for j in np.linspace(0,1,N_divisions+1) for i in np.linspace(0,1,N_divisions+1)]
    #x_vals=[alpha.inverse(t_val) for t_val in t_vals]
    #x_analytical=np.array([f_analytical(a[0],a[1]) for a in x_vals])

    SE=0.0

    # 3 gaussian points to iterate over [0,1]
    # x_numerical=np.zeros(N_nodes) # Integration validation (Done. SE=0)
    for i in range(N_divisions): # i-th division on s axis
        s=np.array([i*d,(i+1)*d])
        for j in range(N_divisions): # j-th division on t axis
            t=np.array([j*d,(j+1)*d])
            indeces=[j*(N_divisions+1)+i,j*(N_divisions+1)+i+1,(j+1)*(N_divisions+1)+i,(j+1)*(N_divisions+1)+i+1]
            # xy=np.array([[alpha.inverse([s[k], t[l]])   for l in range(2)] for k in range(2)])
            SE_local=0.0
            for k in range(N_gauss_points):# k-th gauss point over t
                t_temp=(j+gauss_points[k])*d
                for l in range(N_gauss_points):# l-th gauss point over s
                    s_temp=(i+gauss_points[l])*d
                    
                    xy=alpha.inverse([s_temp, t_temp])
                    J_temp=alpha.prime(xy)
                    #print(f"{J_temp=}")
                    det_J_temp=np.abs(np.linalg.det(J_temp))
                    #print(f'{det_J_temp=}')
                    N_temp=N(gauss_points[l],gauss_points[k])
                    #print(xy)
                    error_temp=f_analytical(xy[0],xy[1])
                    for m in range(N_nodes_e):
                        #continue
                        error_temp-=x_numerical[indeces[m]]*N_temp[m]
                    # print(N_temp.sum()) # ==1
                    SE_local+=gauss_weights[k]*gauss_weights[l]*error_temp**2/det_J_temp*d*d
                    
            SE+=SE_local
            #print(SE)
    RMSE=np.sqrt(SE/1.0)

    print(f'{RMSE=:0.8f}')
    #print(f"When numerical=0.0, RMSE={np.sqrt(28/45):0.8f} ")
    # print(f"{x_vals[0:5]=}\n  {x_vals[-6:-1]=}\n{x_numerical[0:5]=}\n  {x_numerical[-6:-1]=}")
    # print(f"{f_analytical(x_vals[0:5])=}\n  {f_analytical(x_vals[-6:-1])=}\n{RMSE=:0.6f}")


    with open("output_torch.txt", "a") as file:
        # write the N, alpha.name, RMSE in csv format
        file.write(f"{N_nodes},{alpha.name},{RMSE:0.8f}\n")

class custom_function_2d_torch:
    def __init__(self, 
                 function, # (s,t)=f(x,y) e.g. lambda x:[torch.tensor(x[0]**3,dtype=torch.float32,requires_grad=True),
                           #                       torch.tensor(x[1]**2,dtype=torch.float32,requires_grad=True)]
                 #prime, # [[ds/dx,ds/dy],[dt/dx,dt/dy]]
                 inverse, # (x,y)=f_inv(s,t)
                 #Christoffel_symbols, # Chrstoffel symbols ([[[Γ^s_ss,Γ^t_ss],[Γ^s_st,Γ^t_st]],[[Γ^s_ts,Γ^t_ts],[Γ^s_tt,Γ^t_tt]]])
                 name : str=""):
        
        
        for st in [(0,0),(0,1),(1,0),(1,1)]:
            st_temp=torch.tensor(st,dtype=torch.float32)
            f=function(st_temp)
            inv=inverse(st_temp)
            assert type(f) is list, "Function must turn list."
            assert len(f) ==2, "Function must turn list of size 2."
            assert type(inv) is list, "The inverse must turn tuple."
            assert len(inv) ==2, "The inverse must turn tuple of size 2."
        for i in np.linspace(0,1,5):
            for j in np.linspace(0,1,5):

                f_inv=inverse(torch.tensor(function(torch.tensor((i,j)))))
                assert (f_inv[0]-i)**2<0.00001, "The inverse is not correct"
                assert (f_inv[1]-j)**2<0.00001, "The inverse is not correct"
        

        self.function = function
        self.inverse = inverse
        self.name = name

    
    def renew(self,st):
        self.st=st
        st_temp=torch.tensor(st,dtype=torch.float32,requires_grad=True)
        n=len(st)
        xy_temp=self.inverse(st_temp)
        m=len(xy_temp)

        xy=torch.zeros(n,dtype=torch.float32)
        self.xy=np.zeros(n)
        for i in range(n):
            xy[i]=xy_temp[i]
            self.xy[i]=xy_temp[i].detach().numpy()

        # collect the jacobian and christoffel symbols
        jacobian=np.zeros((n,m))
        gamma=np.zeros((n,m,m))
        for i in range(n):
            # collect jacobian (1st order derivatives)
            xy[i].backward(retain_graph=True)
            jacobian[i,:]=st_temp.grad.data
            st_temp.grad.data.zero_()
            
            # collect christoffel symbols (2nd order derivatives)
            g = torch.autograd.grad(xy[i], st_temp, create_graph=True)
            for j in range(m):
                v_temp=[0]*m
                v_temp[j]=1
                g[0].backward(torch.FloatTensor(v_temp),retain_graph=True)
                gamma[i,j,:]=st_temp.grad.data
                st_temp.grad.data.zero_()
        self.basis_vector_st=jacobian
        self.jacobian=np.linalg.inv(jacobian)
        self.gamma=gamma
    
    def prime(self,xy):
        self.xy=xy
        xy_temp=torch.tensor(xy,dtype=torch.float32,requires_grad=True)
        st_temp=self.function(xy_temp)
        n=len(xy)
        m=len(st_temp)

        st=torch.zeros(n,dtype=torch.float32)
        for i in range(n):
            st[i]=st_temp[i]

        # collect the jacobian and christoffel symbols
        jacobian=np.zeros((n,m))
        for i in range(n):
            # collect jacobian (1st order derivatives)
            st[i].backward(retain_graph=True)
            jacobian[i,:]=xy_temp.grad.data
            xy_temp.grad.data.zero_()
            
        self.jacobian=jacobian
        return jacobian

def main_vector_torch(N_divisions:int=5,alpha=None):
    if alpha is None:
        alpha=custom_function_2d_torch(lambda st: [st[0],st[1]],
                            lambda xy: [xy[0],xy[1]], "identity")
    #Definition of mechanical properties of the continuum
    Youngs_modulus=200*10**9#Pa
    Poisson_Ratio=0.3#0.3# No units
    Rho=7850#kg/m^3
    Lame_mu=Youngs_modulus*0.5/(1+Poisson_Ratio)#Pa
    Lame_lambda=Youngs_modulus*Poisson_Ratio/(1+Poisson_Ratio)/(1-2*Poisson_Ratio)#Pa

    force=Rho*np.array([0,-9.81])

    # Definition of mesh on xy coordinates.
    W, H = 1.0, 1.0
    Nx_divisions = N_divisions
    Ny_divisions = N_divisions

    Nx = Nx_divisions + 1
    Ny = Ny_divisions + 1
    d_x = W/Nx_divisions
    d_y = H/Ny_divisions

    N_nodes=Nx*Ny
    # N_nodes_dummy=Nx_divisions*Ny
    N_ele=Ny_divisions*Nx_divisions

    xy_coor=np.empty((N_nodes,2),float)
    cny=np.empty((N_ele,4),int)


    for i_y in range(Ny):
        xy_coor[i_y*Nx:(i_y+1)*Nx,1]=i_y*d_y

    for i_x in range(Nx):
        xy_coor[i_x::Nx,0]=i_x*d_x

    for i_y in range(Ny_divisions):
        for i_x in range(Nx_divisions):
            i_ele=i_x+i_y*Nx_divisions
            cny[i_ele,0]=i_x   + i_y * Nx
            cny[i_ele,1]=i_x+1 + i_y * Nx
            cny[i_ele,2]=i_x   + (i_y+1) * Nx
            cny[i_ele,3]=i_x+1 + (i_y+1) * Nx

            
    cny_dummy=cny.copy()

    for i_y in range(Ny_divisions):
        cny_dummy[(i_y+1)*Nx_divisions-1,1]=i_y * Nx
        cny_dummy[(i_y+1)*Nx_divisions-1,3]=(i_y+1) * Nx

    # Transformation of the mesh to the alpha coordinates
    st_coor=np.zeros((N_nodes,2))
    for i in range(N_nodes):
        st_coor[i]=alpha.function(torch.Tensor(xy_coor[i]))
    #st_coor=np.array(alpha.function(xy_coor.T)).T

    # Gauss points and weights
    # https://en.wikipedia.org/wiki/Gaussian_quadrature
    # gauss_points = (np.array([-np.sqrt(5+2*np.sqrt(10/7))/3,-np.sqrt(5-2*np.sqrt(10/7))/3, 0, 
    #                               np.sqrt(5-2*np.sqrt(10/7))/3,np.sqrt(5+2*np.sqrt(10/7))/3])+1.0)*0.5
    # gauss_weights = np.array([(322-13*np.sqrt(70))/900, (322+13*np.sqrt(70))/900, 128/225, 
    #                           (322+13*np.sqrt(70))/900, (322-13*np.sqrt(70))/900])*0.5
    gauss_points = (np.array([-np.sqrt(3/5), 0,np.sqrt(3/5)])+1.0)*0.5
    gauss_weights = np.array([5/9, 8/9, 5/9])*0.5
    N_gauss_points = gauss_points.size

    # Shape functions : s,t\in[0,1]
    N_nodes_e=4
    N=lambda zeta,eta: np.array([(1-zeta)*(1-eta),  zeta*(1-eta),(1-zeta)*eta,  zeta*eta])
    N_derivatives=lambda zeta,eta: np.array( [ [eta-1,zeta-1],[1-eta,-zeta],[-eta,1-zeta],[eta,zeta]])
    
    
    K=np.zeros((N_nodes*2,N_nodes*2))
    f_vector=np.zeros(N_nodes*2)

    K_row=np.empty((N_ele*(N_nodes_e*2)**2),dtype=int)
    K_col=np.empty((N_ele*(N_nodes_e*2)**2),dtype=int)
    K_values=np.empty((N_ele*(N_nodes_e*2)**2),dtype=float)

    for i_y in range(Ny_divisions):
        for i_x in range(Nx_divisions):
            i_ele=i_y*Nx_divisions+i_x

            # Element 
            node_ids= cny[i_ele]
            node_dummy_ids= cny_dummy[i_ele]
            
            # Element coordinates
            node_coordinates = st_coor[node_ids]

            
            K_e = np.zeros((N_nodes_e*2,N_nodes_e*2))
            f_e = np.zeros(N_nodes_e*2)
            for i_zeta_gauss in range(N_gauss_points):
                for i_eta_gauss in range(N_gauss_points):
                    zeta_gauss = gauss_points[i_zeta_gauss]
                    eta_gauss = gauss_points[i_eta_gauss]

                    N_st=N(zeta_gauss,eta_gauss)

                    s=(node_coordinates[:,0]*N_st).sum()
                    t=(node_coordinates[:,1]*N_st).sum()

                    #prepare collection of variables
                    K_temp = np.zeros((N_nodes_e*2,N_nodes_e*2))
                    F_temp = np.zeros(N_nodes_e*2)

                    alpha.renew((s,t))

                    # Jacobian # 
                    J=alpha.jacobian # J^st_xy
                    basis_vector_st = alpha.basis_vector_st
                    inv_det_J=np.linalg.det(basis_vector_st)
                    
                    #Christoffel symbols
                    Gamma = alpha.gamma

                    #N_local
                    N_derivatives_st=N_derivatives(zeta_gauss,eta_gauss)
                    J2=np.linalg.inv(N_derivatives_st.T@node_coordinates)
                    
                    #First, gather the epsilons.
                    epsilon_v = np.zeros((N_nodes_e*2,2,2))
                    for i_node in range(N_nodes_e):
                        for j_index in range(2):# j_index = 0(s), 1(t)
                            #                 N_i * Gamma^l_jk (k=s,t, l=x,y)
                            epsilon_v_temp = N_st[i_node] * Gamma[j_index,:,:]
                            #         J^(zeta,eta)_(s,t) [ ∂N_i/∂zeta, ∂N_i/∂eta].T @ [(e_j)x, (e_j)y]
                            epsilon_v_temp += J2@N_derivatives_st[i_node,:,None] @ basis_vector_st[j_index,None,:]
                            epsilon_v_temp = J@epsilon_v_temp
                            epsilon_v[i_node*2+j_index,:,:] = (epsilon_v_temp+epsilon_v_temp.T)*0.5
                    
                        
                    sigma_u = np.zeros((N_nodes_e*2,2,2))
                    for i_node in range(N_nodes_e):
                        for j_index in range(2):#j_index=0(s),1(t)
                            sigma_u[i_node*2+j_index,:,:] = \
                                Lame_lambda*np.trace(epsilon_v[i_node*2+j_index,:,:])*np.eye(2)+\
                                2*Lame_mu*epsilon_v[i_node*2+j_index,:,:]
                    
                    for i_node in range(N_nodes_e*2):
                        for i_node_2 in range(N_nodes_e*2):
                            K_temp[i_node,i_node_2] = np.sum(sigma_u[i_node,:,:]*epsilon_v[i_node_2,:,:])
                    K_e += K_temp*gauss_weights[i_zeta_gauss]*gauss_weights[i_eta_gauss]*d_x*d_y#*inv_det_J
                
                    force_st = basis_vector_st @ force# 2x2, 2x1 -> 2x1 
                    for i_node in range(N_nodes_e):
                        F_temp[i_node*2:i_node*2+2] = N_st[i_node]*force_st
                        # for j_index in range(2):
                        #     F_temp[i_node*2+j_index] = N_st[i_node]*force_st[j_index]
                    f_e += F_temp*gauss_weights[i_zeta_gauss]*gauss_weights[i_eta_gauss]*d_x*d_y#*inv_det_J
                    
            indeces=np.zeros(N_nodes_e*2,dtype=int)
            indeces[0:N_nodes_e*2:2]=node_ids*2
            indeces[1:N_nodes_e*2:2]=node_ids*2+1
            indeces[0:N_nodes_e*2:2]=node_dummy_ids*2
            indeces[1:N_nodes_e*2:2]=node_dummy_ids*2+1

            f_vector[indeces]+=f_e

            i_ele_start=(i_x+i_y*Nx_divisions)*(N_nodes_e*2)**2
            for i in range(N_nodes_e*2):
                for j in range(N_nodes_e*2):
                    K_values[i_ele_start+i*(N_nodes_e*2)+j] = K_e[i,j]
                    K_row[i_ele_start+i*(N_nodes_e*2)+j]    = indeces[i]
                    K_col[i_ele_start+i*(N_nodes_e*2)+j]    = indeces[j]

    K=coo_matrix((K_values, (K_row, K_col)), shape=(N_nodes*2, N_nodes*2)).tocsr()
    #K,f_vector

    N_dofs=N_nodes*2

    #Apply boundary conditions
    #u_known_index=np.zeros(Nt*2,dtype=int)
    u_value=np.zeros(N_dofs,dtype=float)
    u_BC=np.zeros(N_dofs,dtype=int)

    for i in range(1,Ny):
        # Fix the left side (fix x displacement)
        u_value[i*Nx*2]=0
        u_BC[i*Nx*2]=-1

        #Fix the right side (fix x displacement)
        u_value[(i+1)*Nx*2-2]=0
        u_BC[(i+1)*Nx*2-2]=-1

    # Fix the bottom side (fix x and y displacement)
    for i in range(Nx):
        u_value[i*2]=0
        u_value[i*2+1]=0
        u_BC[i*2]=-1
        u_BC[i*2+1]=-1

    # Define cyclic boundary

    for i in range(1,Ny):
        #Fix the right side (fix y displacement as cyclic boundary)
        u_BC[(i+1)*Nx*2-1]=-1

    N_known=-np.sum(u_BC)
    N_unknown=N_dofs-N_known

    count=0
    for i in range(N_dofs):
        if u_BC[i]!=-1:
            u_BC[i]=count
            count+=1


    A_row=[]
    A_col=[]
    A_values=[]

    b=np.zeros(N_unknown)

    for ind in range(N_dofs):
        i=u_BC[ind]
        if i==-1:
            continue
        b[i]=f_vector[ind]
        for ptr in range(K.indptr[ind],K.indptr[ind+1]):
            i2=K.indices[ptr]
            if u_BC[i2]!=-1:
                if K.data[ptr]==0.0:
                    continue
                A_values.append(K.data[ptr])
                A_row.append(i)
                A_col.append(u_BC[i2])
            else:
                b[i]-=K.data[ptr]*u_value[i2]
        
    A=coo_matrix((A_values, (A_row, A_col)), shape=(N_unknown, N_unknown)).tocsr()
    u_unknown_value=spsolve(A,b)
    #u_value=np.zeros(Ns*Nt*2)
    #u_value[u_known_index]=u_known_value
    u_value[u_BC!=-1]=u_unknown_value

    # update boundary displacements
    for i in range(1,Ny):
        #Fix the right side (fix y displacement as cyclic boundary)
        u_value[(i+1)*Nx*2-1]=u_value[(i)*Nx*2+1]
    
    f_analytical=lambda x,y: -1/(Lame_lambda+2*Lame_mu)*force[1]*(0.5*y**2-H*y)

    SE=0.0
    for i_y in range(Ny_divisions):
        for i_x in range(Nx_divisions):
            i_ele=i_y*Nx_divisions+i_x

            # Element 
            node_ids= cny[i_ele]
            
            # Element coordinates
            node_coordinates = st_coor[node_ids]

            e_basis=np.zeros((N_nodes_e,2,2))
            for i,node_id in enumerate(node_ids):
                e_basis[i,:,:]=np.linalg.inv(alpha.prime(xy_coor[node_id,:]))
            

            indeces=np.zeros(N_nodes_e*2,dtype=int)
            indeces[0:N_nodes_e*2:2]=node_ids*2
            indeces[1:N_nodes_e*2:2]=node_ids*2+1

            u_values_local=u_value[indeces].reshape((4,2))

            SE_local=0.0
            for i_zeta_gauss in range(N_gauss_points):
                for i_eta_gauss in range(N_gauss_points):
                    zeta_gauss = gauss_points[i_zeta_gauss]
                    eta_gauss = gauss_points[i_eta_gauss]

                    N_st=N(zeta_gauss,eta_gauss)
                    u_local=np.einsum('ij,i,ijk->k',u_values_local,N_st,e_basis)
                    #print(u_local)

                    s=(node_coordinates[:,0]*N_st).sum()
                    t=(node_coordinates[:,1]*N_st).sum()
                    
                    xy=alpha.inverse([torch.tensor(s),torch.tensor(t)])
                    u_analytical_local=np.array([0,f_analytical(xy[0],xy[1])])
                    #print(u_analytical_local)
                    SE_local+=((u_local[0]-u_analytical_local[0])**2+(u_local[1]-u_analytical_local[1])**2)*gauss_weights[i_zeta_gauss]*gauss_weights[i_eta_gauss]
            SE+=SE_local*d_x*d_y

    RMSE=np.sqrt(SE/(W*H))
    
    u_analytical=np.zeros(N_dofs)
    u_analytical[1::2]=f_analytical(xy_coor[:,0],xy_coor[:,1])
    # adjust the u_value depending the local basis vector.
    e_basis_length=np.zeros((N_nodes,2))
    for i in range(Ny) :
        for j in range(Nx):
            #alpha.renew(xy_coor[i*Nx+j,:])
            J=alpha.prime(xy_coor[i*Nx+j,:])
            e_basis_length[i*Nx+j,:]=np.linalg.norm(np.linalg.inv(J),axis=0)

    u_value_xy=u_value*e_basis_length.flatten()

    du=u_value_xy-u_analytical
    
    # plot u_value and u_analytical
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].scatter(xy_coor[:,1],u_value_xy[1::2],label='FEM')
    ax[0].plot(xy_coor[:,1],u_analytical[1::2],label='Analytical')
    ax[0].set_xlabel('y')
    ax[0].set_ylabel('u')
    ax[0].legend()

    # plot u_value and u_analytical
    ax[1].plot(xy_coor[:,1],du[1::2],label='error')
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('u')
    ax[1].legend()
    plt.show()

    #create the mesh on xy and st plane
    fig, ax = plt.subplots(1,2,figsize=(10,5))

    for i in range(N_ele):
        node_ids=cny[i]
        #print(node_ids)
        for indeces in [(0,1),(1,3),(2,3),(0,2)]:
            #print(node_ids[indeces])
            ax[0].plot(xy_coor[node_ids[indeces,],0],xy_coor[node_ids[indeces,],1],c='black')
            ax[1].plot(st_coor[node_ids[indeces,],0],st_coor[node_ids[indeces,],1],c='black')
            


    # set the title 
    ax[0].set_title(f'Coordinate system "{alpha.name}" on cartesian')
    ax[1].set_title(f'Coordinate system "{alpha.name}" on st coordinates')
    #fix the aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    # set the limit [0,1]^2
    ax[0].set_xlim(0,1)
    ax[0].set_ylim(0,1)
    plt.tight_layout()
    plt.savefig(f'coor_shape_{alpha.name}.png')
    plt.show()

    with open("output_torch.txt", "a") as file:
        # write the N, alpha.name, RMSE in csv format
        file.write(f"{N_nodes},{N_divisions},{alpha.name},{RMSE:0.8f}\n")

def main_vector(N_divisions:int=5,alpha=None):
    #Definition of mechanical properties of the continuum
    Youngs_modulus=200*10**9#Pa
    Poisson_Ratio=0.3#0.3# No units
    Rho=7850#kg/m^3
    Lame_mu=Youngs_modulus*0.5/(1+Poisson_Ratio)#Pa
    Lame_lambda=Youngs_modulus*Poisson_Ratio/(1+Poisson_Ratio)/(1-2*Poisson_Ratio)#Pa

    force=Rho*np.array([0,-9.81])


    # Definition of mesh on xy coordinates.
    W, H = 1.0, 1.0
    Nx_divisions = 15
    Ny_divisions = 15

    Nx = Nx_divisions + 1
    Ny = Ny_divisions + 1
    d_x = W/Nx_divisions
    d_y = H/Ny_divisions

    N_nodes=Nx*Ny
    # N_nodes_dummy=Nx_divisions*Ny
    N_ele=Ny_divisions*Nx_divisions

    xy_coor=np.empty((N_nodes,2),float)
    cny=np.empty((N_ele,4),int)


    for i_y in range(Ny):
        xy_coor[i_y*Nx:(i_y+1)*Nx,1]=i_y*d_y

    for i_x in range(Nx):
        xy_coor[i_x::Nx,0]=i_x*d_x

    for i_y in range(Ny_divisions):
        for i_x in range(Nx_divisions):
            i_ele=i_x+i_y*Nx_divisions
            cny[i_ele,0]=i_x   + i_y * Nx
            cny[i_ele,1]=i_x+1 + i_y * Nx
            cny[i_ele,2]=i_x   + (i_y+1) * Nx
            cny[i_ele,3]=i_x+1 + (i_y+1) * Nx

            
    cny_dummy=cny.copy()

    for i_y in range(Ny_divisions):
        cny_dummy[(i_y+1)*Nx_divisions-1,1]=i_y * Nx
        cny_dummy[(i_y+1)*Nx_divisions-1,3]=(i_y+1) * Nx

    st_coor=alpha.function(xy_coor.T).T

    # Gauss points and weights
    # https://en.wikipedia.org/wiki/Gaussian_quadrature
    # gauss_points = (np.array([-np.sqrt(5+2*np.sqrt(10/7))/3,-np.sqrt(5-2*np.sqrt(10/7))/3, 0, 
    #                               np.sqrt(5-2*np.sqrt(10/7))/3,np.sqrt(5+2*np.sqrt(10/7))/3])+1.0)*0.5
    # gauss_weights = np.array([(322-13*np.sqrt(70))/900, (322+13*np.sqrt(70))/900, 128/225, 
    #                           (322+13*np.sqrt(70))/900, (322-13*np.sqrt(70))/900])*0.5
    gauss_points = (np.array([-np.sqrt(3/5), 0,np.sqrt(3/5)])+1.0)*0.5
    gauss_weights = np.array([5/9, 8/9, 5/9])*0.5
    N_gauss_points = gauss_points.size

    # Shape functions : s,t\in[0,1]
    N_nodes_e=4
    N=lambda zeta,eta: np.array([(1-zeta)*(1-eta),  zeta*(1-eta),(1-zeta)*eta,  zeta*eta])
    N_derivatives=lambda zeta,eta: np.array( [ [eta-1,zeta-1],[1-eta,-zeta],[-eta,1-zeta],[eta,zeta]])

K=np.zeros((N_nodes*2,N_nodes*2))
f_vector=np.zeros(N_nodes*2)

K_row=np.empty((N_ele*(N_nodes_e*2)**2),dtype=int)
K_col=np.empty((N_ele*(N_nodes_e*2)**2),dtype=int)
K_values=np.empty((N_ele*(N_nodes_e*2)**2),dtype=float)

for i_y in range(Ny_divisions):
    for i_x in range(Nx_divisions):
        i_ele=i_y*Nx_divisions+i_x

        # Element 
        node_ids= cny[i_ele]
        node_dummy_ids= cny_dummy[i_ele]
        
        # Element coordinates
        node_coordinates = st_coor[node_ids]

        
        K_e = np.zeros((N_nodes_e*2,N_nodes_e*2))
        f_e = np.zeros(N_nodes_e*2)
        for i_zeta_gauss in range(N_gauss_points):
            for i_eta_gauss in range(N_gauss_points):
                zeta_gauss = gauss_points[i_zeta_gauss]
                eta_gauss = gauss_points[i_eta_gauss]

                N_st=N(zeta_gauss,eta_gauss)

                s=(node_coordinates[:,0]*N_st).sum()
                t=(node_coordinates[:,1]*N_st).sum()

                #prepare collection of variables
                K_temp = np.zeros((N_nodes_e*2,N_nodes_e*2))
                F_temp = np.zeros(N_nodes_e*2)

                # Jacobian # 
                J = alpha.prime(alpha.inverse((s,t))) # J^st_xy
                basis_vector_st = np.linalg.inv(J) # e_s,e_t
                #print(basis_vector_st)
                inv_det_J=np.linalg.det(basis_vector_st)
                
                #Christoffel symbols
                Gamma = alpha.Christoffel_symbols((s,t))

                #N_local
                N_derivatives_st=N_derivatives(zeta_gauss,eta_gauss)
                J2=np.linalg.inv(N_derivatives_st.T@node_coordinates)
                
                #First, gather the epsilons.
                epsilon_v = np.zeros((N_nodes_e*2,2,2))
                for i_node in range(N_nodes_e):
                    for j_index in range(2):# j_index = 0(s), 1(t)
                        #                 N_i * Gamma^l_jk (k=s,t, l=x,y)
                        epsilon_v_temp = N_st[i_node] * Gamma[j_index,:,:]
                        #         J^(zeta,eta)_(s,t) [ ∂N_i/∂zeta, ∂N_i/∂eta].T @ [(e_j)x, (e_j)y]
                        epsilon_v_temp += J2@N_derivatives_st[i_node,:,None] @ basis_vector_st[j_index,None,:]
                        epsilon_v_temp = J@epsilon_v_temp
                        epsilon_v[i_node*2+j_index,:,:] = (epsilon_v_temp+epsilon_v_temp.T)*0.5
                
                      
                sigma_u = np.zeros((N_nodes_e*2,2,2))
                for i_node in range(N_nodes_e):
                    for j_index in range(2):#j_index=0(s),1(t)
                        sigma_u[i_node*2+j_index,:,:] = \
                            Lame_lambda*np.trace(epsilon_v[i_node*2+j_index,:,:])*np.eye(2)+\
                            2*Lame_mu*epsilon_v[i_node*2+j_index,:,:]
                
                for i_node in range(N_nodes_e*2):
                    for i_node_2 in range(N_nodes_e*2):
                        K_temp[i_node,i_node_2] = np.sum(sigma_u[i_node,:,:]*epsilon_v[i_node_2,:,:])
                K_e += K_temp*gauss_weights[i_zeta_gauss]*gauss_weights[i_eta_gauss]*d_x*d_y#*inv_det_J
             
                force_st = basis_vector_st @ force# 2x2, 2x1 -> 2x1 
                for i_node in range(N_nodes_e):
                    F_temp[i_node*2:i_node*2+2] = N_st[i_node]*force_st
                    # for j_index in range(2):
                    #     F_temp[i_node*2+j_index] = N_st[i_node]*force_st[j_index]
                f_e += F_temp*gauss_weights[i_zeta_gauss]*gauss_weights[i_eta_gauss]*d_x*d_y#*inv_det_J
                
        indeces=np.zeros(N_nodes_e*2,dtype=int)
        indeces[0:N_nodes_e*2:2]=node_ids*2
        indeces[1:N_nodes_e*2:2]=node_ids*2+1
        indeces[0:N_nodes_e*2:2]=node_dummy_ids*2
        indeces[1:N_nodes_e*2:2]=node_dummy_ids*2+1

        f_vector[indeces]+=f_e

        i_ele_start=(i_x+i_y*Nx_divisions)*(N_nodes_e*2)**2
        for i in range(N_nodes_e*2):
            for j in range(N_nodes_e*2):
                K_values[i_ele_start+i*(N_nodes_e*2)+j] = K_e[i,j]
                K_row[i_ele_start+i*(N_nodes_e*2)+j]    = indeces[i]
                K_col[i_ele_start+i*(N_nodes_e*2)+j]    = indeces[j]

K=coo_matrix((K_values, (K_row, K_col)), shape=(N_nodes*2, N_nodes*2)).tocsr()
#K,f_vector

class custom_function_2d:
    def __init__(self, 
                 function, # (s,t)=f(x,y)
                 prime, # [[ds/dx,ds/dy],[dt/dx,dt/dy]]
                 inverse, # (x,y)=f_inv(s,t)
                 Christoffel_symbols, # Chrstoffel symbols ([[[Γ^s_ss,Γ^t_ss],[Γ^s_st,Γ^t_st]],[[Γ^s_ts,Γ^t_ts],[Γ^s_tt,Γ^t_tt]]])
                 name : str=""):
        
        
        for st in [(0,0),(0,1),(1,0),(1,1)]:
            st_temp=torch.tensor(st,dtype=torch.float32)
            f=function(st_temp)
            inv=inverse(st_temp)
            assert type(f) is list, "Function must turn list."
            assert len(f) ==2, "Function must turn list of size 2."
            assert type(inv) is list, "The inverse must turn tuple."
            assert len(inv) ==2, "The inverse must turn tuple of size 2."
        for i in np.linspace(0,1,5):
            for j in np.linspace(0,1,5):

                f_inv=inverse(torch.tensor(function(torch.tensor((i,j)))))
                assert (f_inv[0]-i)**2<0.00001, "The inverse is not correct"
                assert (f_inv[1]-j)**2<0.00001, "The inverse is not correct"
        

        self.function = function
        self.inverse = inverse
        self.prime = prime
        self.Christoffel_symbols = Christoffel_symbols
        self.name = name


def main():
    line=input("Enter the number of divisions:")
    N_divisions=int(line)
    line=input("Enter the alpha name:")
    alpha_name=line
    line=input("Choose the mode (original, torch):")
    mode=line
    if mode=="original":
        main_original(N_divisions,alpha_name)
    elif mode=="torch":
        main_torch(N_divisions,alpha_name)
    else:
        raise ValueError("The mode is not defined.")
    
if __name__ == "__main__":
    main_torch()
    alpha=custom_function_2d(lambda xy:  np.array(xy),
                            lambda st:  np.array( st), "identity")

    alpha=custom_function_2d(lambda xy:  np.array([xy[0]**3,xy[1]**3]),
                            lambda xy:  np.array([[3*xy[0]**2,0],[0,3*xy[1]**2]]), 
                            lambda st:  np.array( [st[0]**(1/3),st[1]**(1/3)]), "cube")

    alpha=custom_function_2d(lambda xy:  np.array([np.sqrt(xy[0]),xy[1]**3]),
                            lambda xy:  np.array([[0.5/np.sqrt(xy[0]),0],[0,3*xy[1]**2]]), 
                            lambda st:  np.array( [st[0]**2,st[1]**(1/3)]), "sqrt-cube")

    # alpha=custom_function_2d(lambda xy:  np.array([np.sqrt(xy[0]**2+xy[1]**2),np.arctan2(xy[1],xy[0])*2/np.pi()]),
    #                           lambda xy:  np.array([[0.5/np.sqrt(xy[0]),0],[0,3*xy[1]**2]]), 
    #                           lambda st:  np.array( [st[0]**2,st[1]**(1/3)]))


    alpha=custom_function_2d(lambda xy:  np.array([xy[0]**2,xy[1]**2]),
                            lambda xy:  np.array([[2*xy[0],0],[0,2*xy[1]]]), 
                            lambda st: np.sqrt(np.array( st)), "square")

    # alpha=custom_function_2d(lambda xy:  np.sqrt(np.array(xy)),
    #                           lambda xy:  np.array([[0.5/np.sqrt(xy[0]),0],[0,0.5/np.sqrt(xy[1])]]), 
    #                           lambda st: np.array( st)**2)

