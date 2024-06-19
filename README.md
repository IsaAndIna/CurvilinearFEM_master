# CurvilinearFEM_master
This git repository is aimed to manage the documents used for my Master research, including source codes, research notes and records.

# Formulation of Finite Element Method on Curvilinear Coordinate System for Linear Elasticity Problem.
In this section, we discuss the mathematical formulation of the finite element method on curvilinear coordinate system. Especially, we will apply it in a case of Linear Elasticity problem.

## Definition of curvilinear coordinate system
Suppose a cartesian coordinate system with $n$ axis, described with $n$ basis vectors $\underline{e_{x_i}}  $, where $i \in \mathcal{N} = \lbrace 1,2,..., n \rbrace $. Then, a vector $\underline{x}$ on this coordinate system is described as 

$$\underline{x} = \sum_{i\in{\mathcal{N}}}  x_i\underline{e_{x_i}} \\ 
= \begin{pmatrix}x_1 & x_2 & ... & x_n\end{pmatrix}^\top .$$

Let $\underline{\psi}: \mathbb{R}^n \rightarrow \mathbb{R}^n$ be a coordinate transformation function. 
Then a coordinate transformation from $\underline{x}$ to $\underline{y}$ can be described as $y_j= \psi_j  (x_1, x_2, ..., x_n)$ , where $y_i$ is the $i$ th coefficient of the linear combination of the  basis vectors, s.t. 
$\underline{y} = \Sigma_{j\in \mathcal{N}} \ y_j\underline{e_{y_j}} .$  
The $j$ th basis vector $\underline{e_{y_j}}$ of the $\underline{y}$ coordinate system can be derived by 

$$\underline{e_{y_j}}= \frac{\partial \underline{x}  }{\partial y_j} = 
\begin{pmatrix}\frac{\partial x_1}{\partial y_j} & \frac{\partial x_2 }{\partial y_j} & ... & \frac{\partial x_n  }{\partial y_j} \end{pmatrix}^\top
=\sum_{i\in \mathcal{N}} \frac{\partial x_i  }{\partial y_j} \underline{e_{x_i}} .$$

Note that this expression describes the basis vector of the $\underline{y}$ coordinate system, as a linear combination of the $n$ basis vectors of the original Cartesian coordinate system 
$\langle \underline{e_{x_i}} \ | \ i \in \mathcal{N}\rangle$. 
Let $\psi^{-1}: \mathbb{R}^n \rightarrow \mathbb{R}^n$ be the inverse coordinate transformation function of $\psi$ , then,

$$x_i= \psi_i^{-1} (y_1, y_2, ..., y_n)$$

$$\underline{e_{y_j}}= 
\begin{pmatrix}\frac{\partial  \psi_1^{-1}}{\partial y_j} & \frac{\partial  \psi_2^{-1} }{\partial y_j} & ... & \frac{\partial  \psi_n^{-1} }{\partial y_j} \end{pmatrix}^\top
=\sum  \frac{\partial  \psi_i^{-1}  }{\partial y_j} \underline{e_{x_i}} .$$

Under Cartesian coordinate system, the basis vectors are constant, in other words, don't change from point to point, while in general, they depend on the point. Thus, we can calculate the partial derivative of $\underline{e_{y_j}}$ as follows:

$$\frac{\partial \underline{e_{y_j}}}{\partial y_k} = 
\begin{pmatrix}\frac{\partial^2 x_1}{\partial y_k \partial y_j} & \frac{\partial^2 x_2}{\partial y_k \partial y_j} &  ... & \frac{\partial^2 x_n}{\partial y_k \partial y_j}   \end{pmatrix}^\top \\ = 
\begin{pmatrix}\frac{\partial^2 \psi_1^{-1}}{\partial y_k \partial y_j} & \frac{\partial^2 \psi_2^{-1}}{\partial y_k \partial y_j} &  ... & \frac{\partial^2 \psi_n^{-1}}{\partial y_k \partial y_j}   \end{pmatrix}^\top . $$

By defining a new symbol $\Gamma^{x_i}_{y_j,y_k}$ , also known as Christoffel symbol, by 
$\Gamma^{x_i}_{y_j,y_k} =\frac{\partial^2 x_i}{\partial y_k \partial y_j}=\frac{\partial^2 \psi_i^{-1}}{\partial y_k \partial y_j}$
, the equation above is transformed into the next form:

$$\frac{\partial \underline{e_{y_j}}}{\partial y_k} = \begin{pmatrix}\Gamma^{x_1}_{y_j,y_k} & \Gamma^{x_2}_{y_j,y_k} &  ... & \Gamma^{x_n}_{y_j,y_k}   \end{pmatrix}^\top$$ 

$$=\sum_{i\in \mathcal{N}} \Gamma^{x_i}_{y_j,y_k} \underline{e_{x_i}} .$$

On the other hand, the integral of an arbitraty function $\phi$ over a domain $\Omega_x$ is calculated as follows:

$$\int_{\Omega_x} \phi(x_1,x_2,...,x_n)dx_1dx_2...dx_n$$ 

$$= \int_{\Omega_y} \phi(y_1,y_2,...,y_n) \left|\frac{\partial (x_1,x_2,...,x_n)}{\partial (y_1,y_2,...,y_n) } \right| dy_1dy_2...dy_n ,$$

where 

$$\frac{\partial (x_1,x_2,...,x_n)}{\partial (y_1,y_2,...,y_n) } = \begin{bmatrix} \frac{\partial x_1}{ \partial y_1} & \frac{\partial x_2}{ \partial y_1} & \dots& \frac{\partial x_n}{ \partial y_1} \\
\frac{\partial x_1}{ \partial y_2} & \frac{\partial x_2}{ \partial y_2} & \dots& \frac{\partial x_n}{ \partial y_2} \\ 
\vdots & \vdots & \ddots& \vdots \\ 
\frac{\partial x_1}{ \partial y_n} & \frac{\partial x_2}{ \partial y_n} & \dots& \frac{\partial x_n}{ \partial y_n}   \end{bmatrix}$$

$$= \begin{bmatrix} \frac{\partial \psi_1^{-1}}{ \partial y_1} & \frac{\partial \psi_2^{-1}}{ \partial y_1} & \dots& \frac{\partial \psi_n^{-1}}{ \partial y_1} \\ 
\frac{\partial \psi_1^{-1}}{ \partial y_2} & \frac{\partial \psi_2^{-1}}{ \partial y_2} & \dots& \frac{\partial \psi_n^{-1}}{ \partial y_2} \\ 
\vdots & \vdots & \ddots& \vdots \\ 
\frac{\partial \psi_1^{-1}}{ \partial y_n} & \frac{\partial \psi_2^{-1}}{ \partial y_n} & \dots& \frac{\partial \psi_n^{-1}}{ \partial y_n}    \end{bmatrix}  = \mathbf{J}^x_y . $$

$\mathbf{J}^x_y $ is also known as the Jacobian matrix of $\psi$.


## Definition of problem
The problem to be solved is the Linear Elasticity problem, which can be discribed with the following equations.

Generalized Hooke's Law (Stress-Strain):

$$\underline{\underline{\sigma}} = \lambda \ tr( \underline{\underline{\epsilon}}) \mathbf{I}_n + 2\mu \underline{\underline{\epsilon}} $$

Strain-displacement: 

$$\underline{\underline{\epsilon}} = \nabla \underline{u}$$

Equilibrium : 

$$\nabla \cdot \underline{\underline{\sigma}}  + \underline{f}= \underline{0}$$

Note that 
$\underline{\underline{\sigma}} \in \mathbb{R}^{n\times n}$ 
is the stress tensor, $\underline{\underline{\epsilon}} \in \mathbb{R}^{n\times n}$ 
is the strain tensor
and 
$\underline{u} \in \mathbb{R}^{n}$ 
is the displacement vector.
Here-in-after, the vectors will be underlined, like $\underline{u}$, and tensor will have two underlines, like 
$\underline{\underline{\sigma}}$ . 

$\nabla$ 
is the Del symbol, which can represent the gradient, the divergence or the curl of a vector, depending on the operator. In this context, 
$\nabla \underline{v}$ 
is a $n \times n$ matrix with $ \frac{\partial v_j }{\partial x_i } $ as its $(i,j)$ component, and $\nabla \cdot A$ 
is a $n$ dimention vector with $\sum_{j\in \mathcal{N}} \frac{\partial A_{i,j} }{\partial x_j }$  as its $i$ th component. 

$\mathbf{I}_n \in \mathbb{R}^{n\times n}$ 
is an identity matrix of $n$ dimentions.

$\lambda $ and $\mu$ are the two Lamé constants, describing the mechanical property of the continuum. $\underline{f} \in \mathbb{R}^{n}$ is the external force working on the material. For example, when the density is $\rho$ and the gravity is $\underline{g} \in \mathbb{R}^{n}$, $\underline{f}=\rho \underline{g} $ .


<!-- $ = \Sigma_i \frac{\partial v_i}{\partial x_i} \underline{e_i} $, where $ \underline{e_i}  $ is the unit vector in direction of the $i$ th axis on the Cartesian coordinate system. -->

## Derivation of weak form of the problem
Let $\underline{v}\in \mathbb{R}^n $ be a test function. Then the weak form of the equilibrium equation above is described as:

$$ \int_{\Omega_x} \underline{v}^\top \left( \nabla \cdot \underline{\underline{\sigma}}  + \underline{f} \right) \ dx_1dx_2...dx_n = 0 . $$ 

Thus 

$$ \int_{\Omega_x} \underline{v}^\top \left( \nabla \cdot \underline{\underline{\sigma}}  \right) \ dx_1dx_2...dx_n = - \int_{\Omega_x} \underline{v}^\top f \ dx_1dx_2...dx_n . $$  

By integrating it by parts to the left side, we obtain:

$$ \int_{\Omega_x} \underline{v}^\top \left( \nabla \cdot \underline{\underline{\sigma}}  \right) \ dx_1dx_2...dx_n \\
= \int_{\partial\Omega_x} \underline{v}^\top \left(  \underline{\underline{\sigma}} \cdot \underline{n}  \right) \ ds - \int_{\Omega_x} \left( \nabla \underline{v}\right) : \underline{\underline{\sigma}}   \ dx_1dx_2...dx_n \\
=\int_{\partial\Omega_x} \underline{v}^\top  \underline{t} \ ds - \int_{\Omega_x} \left( \nabla \underline{v}\right) : \underline{\underline{\sigma}}   \ dx_1dx_2...dx_n, $$ 

where $\partial\Omega_x$ is the boundary of the region $\Omega_x$, $\underline{n} $ is the normal vector of, and $\underline{t}=\underline{\underline{\sigma}} \cdot \underline{n} $ is the external force applied to the boundary  $\partial\Omega_x$ .
Therefore, the weak form is 

$$  \int_{\Omega_x} \left( \nabla \underline{v}\right) : \underline{\underline{\sigma}} (\underline{u})   \ dx_1dx_2...dx_n \\ 
= \int_{\partial\Omega_x} \underline{v}^\top  \underline{t} \ ds + \int_{\Omega_x} \underline{v}^\top f \ dx_1dx_2...dx_n , $$

where « $:$ » operator is the sum of element-wise product.
Note that the left side is a function of the displacement $\underline{u}$ and the right side is independent of it. 

## Derivation of the linear equation 

By applying a test function $\underline{v}$ and discretizing the region appropriately, we can convert this weak form into a finite number of linear equations, s.t. 
$K\underline{\bar{u}} = \underline{\bar{f}} $, where $K \in \mathbb{R}^{N \times N}$, $\underline{\bar{u}},\underline{\bar{f}}\in \mathbb{R}^{N } $
, $N$ is the number of test functions applied.

Suppose the region $\Omega$ is discretized by a mesh of $N_{ele}$ elements and $N_{nod}$ nodes. Let $\underline{u}$ be described as a linear combination of shape functions $\underline{\bar{u}} $, s.t. 

$$\underline{\bar{u}} = \sum_{i=1}^{N_{nod}} \sum_{j \in \mathcal{N}} \bar{u_{i,j}} \phi_i \underline{e_{j}} .$$








