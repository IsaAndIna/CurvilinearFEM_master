import numpy as np
import matplotlib.pyplot as plt



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

def main(N_divisions:int=5,alpha=None):
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


    with open("output.txt", "a") as file:
        # write the N, alpha.name, RMSE in csv format
        file.write(f"{N_nodes},{alpha.name},{RMSE:0.8f}\n")



if __name__ == "__main__":
    main()
    alpha=custom_function_2d(lambda xy:  np.array(xy),
                            lambda xy:  np.array([[1,0],[0,1]]), 
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

