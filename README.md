# CurvilinearFEM
This git repository is aimed to manage the documents used for my master research on Finite Element Method on Curvilinear Coordinate System, including source codes, research notes and records.

Isaya Inafuku. June 22, 2024.

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

By defining a new symbol 
$\Gamma^{x_i}_{y_j,y_k}$
, also known as Christoffel symbol, by 
$\Gamma^{x_i}_{y_j,y_k} =\frac{\partial^2 x_i}{\partial y_k \partial y_j}=\frac{\partial^2 \psi_i^{-1}}{\partial y_k \partial y_j}$
, the equation above is transformed into the next form:

$$\frac{\partial \underline{e_{y_j}}}{\partial y_k} = \begin{pmatrix}\Gamma^{x_1}_{y_j,y_k} & \Gamma^{x_2}_{y_j,y_k} &  ... & \Gamma^{x_n}_{y_j,y_k}   \end{pmatrix}^\top$$ 

$$=\sum_{i\in \mathcal{N}} \Gamma^{x_i}_{y_j,y_k} \underline{e_{x_i}} .$$

On the other hand, the integral of an arbitraty function $\xi$ over a domain $\Omega_x$ is calculated as follows:

$$\int_{\Omega_x} \xi(x_1,x_2,...,x_n)dx_1dx_2...dx_n$$ 

$$= \int_{\Omega_y} \xi(y_1,y_2,...,y_n) \left|\frac{\partial (x_1,x_2,...,x_n)}{\partial (y_1,y_2,...,y_n) } \right| dy_1dy_2...dy_n ,$$

where 

$$\frac{\partial (x_1,x_2,...,x_n)}{\partial (y_1,y_2,...,y_n) } = \begin{bmatrix} \frac{\partial x_1}{ \partial y_1} & \frac{\partial x_2}{ \partial y_1} & \dots& \frac{\partial x_n}{ \partial y_1} \\
\frac{\partial x_1}{ \partial y_2} & \frac{\partial x_2}{ \partial y_2} & \dots& \frac{\partial x_n}{ \partial y_2} \\ 
\vdots & \vdots & \ddots& \vdots \\ 
\frac{\partial x_1}{ \partial y_n} & \frac{\partial x_2}{ \partial y_n} & \dots& \frac{\partial x_n}{ \partial y_n}   \end{bmatrix}$$

$$= \begin{bmatrix} \frac{\partial \psi_1^{-1}}{ \partial y_1} & \frac{\partial \psi_2^{-1}}{ \partial y_1} & \dots& \frac{\partial \psi_n^{-1}}{ \partial y_1} \\ 
\frac{\partial \psi_1^{-1}}{ \partial y_2} & \frac{\partial \psi_2^{-1}}{ \partial y_2} & \dots& \frac{\partial \psi_n^{-1}}{ \partial y_2} \\ 
\vdots & \vdots & \ddots& \vdots \\ 
\frac{\partial \psi_1^{-1}}{ \partial y_n} & \frac{\partial \psi_2^{-1}}{ \partial y_n} & \dots& \frac{\partial \psi_n^{-1}}{ \partial y_n}    \end{bmatrix}  = \mathbf{J}^x_y .$$

$\mathbf{J}^x_y$ is also known as the Jacobian matrix of $\psi$.


## Definition of problem
The problem to be solved is the Linear Elasticity problem, which can be discribed with the following equations.

Generalized Hooke's Law (Stress-Strain):

$$\underline{\underline{\sigma}} = \lambda \ tr( \underline{\underline{\epsilon}}) \mathbf{I}_n + 2\mu \underline{\underline{\epsilon}} $$

Strain-displacement: 

$$\underline{\underline{\epsilon}}(u) = \frac{1}{2} \left( \nabla \underline{u} + \left( \nabla \underline{u} \right)^\top \right)$$

Equilibrium : 

$$\nabla \cdot \underline{\underline{\sigma}}  + \underline{\textrm{f}}= \underline{0}$$

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
is a $n \times n$ matrix with $\frac{\partial v_j}{\partial x_i }$ as its $(i,j)$ component, and $\nabla \cdot A$ 
is a $n$ dimention vector with $\sum_{j\in \mathcal{N}} \frac{\partial A_{i,j} }{\partial x_j}$  as its $i$ th component. 

$\mathbf{I}_n \in \mathbb{R}^{n\times n}$ 
is an identity matrix of $n$ dimentions.

$\lambda$ and $\mu$ are the two Lamé constants, describing the mechanical property of the continuum. $\underline{\textrm{f}} \in \mathbb{R}^{n}$ is the external force working on the material. For example, when the density is $\rho$ and the gravity is $\underline{g} \in \mathbb{R}^{n}$, $\underline{\textrm{f}}=\rho \underline{g}$ .


<!-- $ = \Sigma_i \frac{\partial v_i}{\partial x_i} \underline{e_i} $, where $ \underline{e_i}  $ is the unit vector in direction of the $i$ th axis on the Cartesian coordinate system. -->

## Derivation of weak form of the problem
Let 
$\underline{v} \in \mathbb{R}^n$ 
be a test function. Then the weak form of the equilibrium equation above is described as:

$$\int_{\Omega_x} \underline{v}^\top \left( \nabla \cdot \underline{\underline{\sigma}}  + \underline{\textrm{f}} \right) \ dx_1dx_2...dx_n = 0 . $$ 

Thus 

$$ \int_{\Omega_x} \underline{v}^\top \left( \nabla \cdot \underline{\underline{\sigma}}  \right) \ dx_1dx_2...dx_n = - \int_{\Omega_x} \underline{v}^\top \underline{\textrm{f}} \ dx_1dx_2...dx_n . $$  

By integrating it by parts to the left side, we obtain:

$$\int_{\Omega_x} \underline{v}^\top \left( \nabla \cdot \underline{\underline{\sigma}}  \right) \ dx_1dx_2...dx_n \\
= \int_{\partial\Omega_x} \underline{v}^\top \left(  \underline{\underline{\sigma}} \cdot \underline{n}  \right) \ ds - \int_{\Omega_x} \left( \nabla \underline{v}\right) : \underline{\underline{\sigma}}   \ dx_1dx_2...dx_n \\
=\int_{\partial\Omega_x} \underline{v}^\top  \underline{t} \ ds - \int_{\Omega_x} \left( \nabla \underline{v}\right) : \underline{\underline{\sigma}}   \ dx_1dx_2...dx_n, $$ 

where $\partial\Omega_x$ is the boundary of the region $\Omega_x$, $\underline{n}$ is the normal vector of, $\underline{t}=\underline{\underline{\sigma}} \cdot \underline{n}$ is the external force applied to the boundary  $\partial\Omega_x$ , and « $:$ » operator is the sum of element-wise product. Considering that $\underline{\underline{\sigma}}$ is a symetric matrix, we can convert $\left( \nabla \underline{v}\right) : \underline{\underline{\sigma}}$ in the flloing way:

$$\left( \nabla \underline{v}\right) : \underline{\underline{\sigma}} = \frac{1}{2} \left( \nabla \underline{v} + \left( \nabla \underline{v} \right)^\top \right) : \underline{\underline{\sigma}} + \frac{1}{2} \left( \nabla \underline{v} - \left( \nabla \underline{v} \right)^\top \right) : \underline{\underline{\sigma}} = \underline{\underline{\epsilon}}(\underline{v}) : \underline{\underline{\sigma}} , $$


Note that $\nabla \underline{v} - \left( \nabla \underline{v} \right)^\top$ is an alternating sign matrix with its diagonal components 0, and its element-wise product sum with a symetric matrix returns 0.
Therefore, the weak form is 

$$ \int_{\Omega_x} \underline{\underline{\epsilon}} \left(  \underline{v}\right) : \underline{\underline{\sigma}} (\underline{u})   \ dx_1dx_2...dx_n = \int_{\partial\Omega_x} \underline{v}^\top  \underline{t} \ ds + \int_{\Omega_x} \underline{v}^\top \underline{\textrm{f}} \ dx_1dx_2...dx_n . $$

Note that the left side is a function of the displacement $\underline{u}$ and the right side is independent of it. 

## Derivation of the linear equation 

By applying a test function $\underline{v}$ and discretizing the region appropriately, we can convert this weak form into a finite number of linear equations, s.t. 
$K\underline{\hat{u}} = \underline{\hat{\textrm{f}}} $, where 
$K \in \mathbb{R}^{N \times N}$, $\underline{\hat{u}},\underline{\hat{\textrm{f}}}\in \mathbb{R}^{N}$
, $N$ is the number of test functions applied.

Suppose the region $\Omega$ is discretized by a mesh of $N_{ele}$ elements and $N_{nod}$ nodes. Let $\underline{u}$ be described as a linear combination of shape functions, s.t. 

$$\underline{\hat{u}} = \sum_{i=1}^{N_{nod}} \sum_{j \in \mathcal{N}} \hat{u}_{i,j} \phi_i \underline{e_{j}} ,$$

where $\phi_i$ is the shape function of the $i$ th node, $\underline{e_{j}}$ is the basis function in the direction of the $j$ th axis of an coordinate system, and $\hat{u}_{i,j}$ is an unknown scalar value of the displacement of the $i$ the node in the direction of $j$ the axis. Then the strain tensor is 

$$\underline{\underline{\epsilon}}(\hat{\underline{u}}) = \frac{1}{2} \left( \nabla \hat{\underline{u}} + \left( \nabla \hat{\underline{u}} \right)^\top \right)$$

$$=\sum_{i=1}^{N_{nod}} \sum_{j \in \mathcal{N}} \hat{u}_{i,j}  \frac{1}{2} \left( \nabla \left( \phi_i \underline{e_{j}} \right) + \left( \nabla \left( \phi_i \underline{e_{j}} \right) \right)^\top \right)$$

$$=\sum_{i=1}^{N_{nod}} \sum_{j \in \mathcal{N}} \hat{u}_{i,j} \ \underline{\underline{\epsilon}}(\phi_i \underline{e_{j}}) \ ,$$

and the stress tensor is 

$$\underline{\underline{\sigma}} (\underline{\underline{\epsilon}}(\hat{\underline{u}})) =  \lambda \, tr( \underline{\underline{\epsilon}}(\hat{\underline{u}}) ) \mathbf{I}_n + 2\mu \underline{\underline{\epsilon}}(\hat{\underline{u}})$$

$$=\sum_{i=1}^{N_{nod}} \sum_{j \in \mathcal{N}} \hat{u}_{i,j} \, \lambda \, tr(\underline{\underline{\epsilon}}(\phi_i \underline{e_{j}}) ) \mathbf{I}_n + 2\mu \, \underline{\underline{\epsilon}}(\phi_i \underline{e_{j}})= \sum_{i=1}^{N_{nod}} \sum_{j \in \mathcal{N}} \hat{u}_{i,j} \, \underline{\underline{\sigma}} (\underline{\underline{\epsilon}}(\phi_i \underline{e_{j}})) .$$

In the same way, let the test function $\underline{v}$ be a shape function, s.t.

$$\underline{v} = \phi_{i'} \underline{e_{j'}} .$$

Then, by substituting the stress tensor and the strain tensor, the weak form is described as:

$$\sum_{i=1}^{N_{nod}} \sum_{j \in \mathcal{N}} \hat{u}_{i,j} \int_{\Omega_x} \underline{\underline{\epsilon}}(\phi_{i'} \underline{e_{j'}}) : \underline{\underline{\sigma}} (\underline{\underline{\epsilon}}(\phi_i \underline{e_{j}}))   \ dx_1dx_2...dx_n \\ = \int_{\partial\Omega_x} (\phi_{i'} \underline{e_{j'}})^\top  \underline{t} \, ds + \int_{\Omega_x} (\phi_{i'} \underline{e_{j'}})^\top \underline{\textrm{f}} \, dx_1dx_2...dx_n . $$

## Numerical Integration
In case an arbitrary function $f(x)$ cannot be analytically integrated, its integration $\int f(x) dx$ is calculated by numerical integration, by approximating the integral in the following way:

$$\int_a^b f(x) dx  = \sum_{i=1}^n w_i f(x_i) ,$$

where $x_i$ is the integration point and  $w_i$ is the weight for this point. When $a=-1$ and $b=1$, a major method, Gauss–Legendre quadrature, chooses the $x_i$ and $w_i$ meeting the following equations:

$$P_n(x_i)=0, w_i= \frac{2}{(1-x_i^2) \left[ P'_n (x_i)\right]^2} .$$


## Calculation of displacement tensor
Let the shape function be defined on $\underline{y}$ coordinates ($\phi_i=\phi_i(y_1,y_2,...,y_n)$ and $\underline{e_{j}}=\underline{e_{y_j}} $).
As explained above, the displacement $\underline{\underline{\epsilon}}(\phi_{i} \underline{e_{y_j}})$ is calculated by:

$$\underline{\underline{\epsilon}}(\phi_{i} \underline{e_{y_j}}) = \frac{1}{2} \left( \nabla_{\underline{x}} \left( \phi_{i} \underline{e_{y_j}} \right)  + \left( \nabla_{\underline{x}} \left( \phi_{i} \underline{e_{y_j}} \right) \right)^\top \right).$$

Considering that $\phi_i$ is calculated on $\underline{y}$ coordinates, we substitute the $\nabla_{\underline{x}}$ operator by $\nabla_{\underline{x}} =J_{\underline{x}}^{\underline{y}} \nabla_{\underline{y}}$, where

$$J_{\underline{x}}^{\underline{y}} = \begin{bmatrix}\frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} &  ... & \frac{\partial y_n}{\partial x_1}  \\ \frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} &  ... & \frac{\partial y_n}{ \partial x_2}  \\ \vdots & \vdots & \ddots& \vdots \\ \frac{\partial y_1}{\partial x_n} & \frac{\partial y_2}{\partial x_n} &  ... & \frac{\partial y_n}{ \partial x_n}  \\ \end{bmatrix} $$

By applying derivation by part, we obtain $\nabla \left( \phi_{i} \underline{e_{y_j}} \right) =  \left(\nabla \phi_{i}  \right) \underline{e_{y_j}} + \phi_{i}  \left( \nabla  \underline{e_{y_j}} \right) $, where

$$\left(\nabla \phi_{i}  \right) \underline{e_{y_j}}= \begin{bmatrix} \frac{\partial \phi_{i}}{ \partial y_1} \\\frac{\partial \phi_{i}}{ \partial y_2}  \\ \vdots  \\ \frac{\partial \phi_{i}}{ \partial y_n}   \end{bmatrix} \underline{e_{y_j}} = \begin{bmatrix} \frac{\partial \phi_i}{ \partial y_1}[e_{y_j}]_{x_1} & \frac{\partial \phi_i}{ \partial y_1}[e_{y_j}]_{x_2} & \dots& \frac{\partial \phi_i}{ \partial y_1}[e_{y_j}]_{x_n} \\ 
\frac{\partial \phi_i}{ \partial y_2}[e_{y_j}]_{x_1} & \frac{\partial \phi_i}{ \partial y_2}[e_{y_j}]_{x_2} & \dots& \frac{\partial \phi_i}{ \partial y_2}[e_{y_j}]_{x_n} \\ 
\vdots & \vdots & \ddots& \vdots \\ 
\frac{\partial \phi_i}{ \partial y_n}[e_{y_j}]_{x_1} & \frac{\partial \phi_i}{ \partial y_n}[e_{y_j}]_{x_2} & \dots& \frac{\partial \phi_i}{ \partial y_n}[e_{y_j}]_{x_n}    \end{bmatrix} ,$$

$$\nabla  \underline{e_{y_j}} = \begin{bmatrix} \frac{\partial \underline{e_{y_j}}}{ \partial y_1} \\\frac{\partial \underline{e_{y_j}}}{ \partial y_2}  \\ \vdots  \\ \frac{\partial \underline{e_{y_j}}}{ \partial y_n}   \end{bmatrix} =
\begin{bmatrix}\Gamma^{x_1}_{y_j,y_1} & \Gamma^{x_2}_{y_j,y_1} &  ... & \Gamma^{x_n}_{y_j,y_1}  \\\Gamma^{x_1}_{y_j,y_2} & \Gamma^{x_2}_{y_j,y_2} &  ... & \Gamma^{x_n}_{y_j,y_2} \\ \vdots & \vdots & \ddots& \vdots \\\Gamma^{x_1}_{y_j,y_n} & \Gamma^{x_2}_{y_j,y_n} &  ... & \Gamma^{x_n}_{y_j,y_n}   \end{bmatrix} .$$

Note that when $\underline{y}$ coordinate system is cartesian, $\nabla  \underline{e_{y_j}}=\underline{\underline{0}}$ .


