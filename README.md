# CurvilinearFEM_master
This git repository is aimed to manage the documents used for my Master research, including source codes, research notes and records.

# Formulation of Finite Element Method on Curvilinear Coordinate System for Linear Elasticity Problem.
In this section, we discuss the mathematical formulation of the finite element method on curvilinear coordinate system. Especially, we will apply it in a case of Linear Elasticity problem.

## Definition of problem
The problem to be solved is the Linear Elasticity problem on 2D plane, which can be discribed with the following equations.

Generalized Hooke's Law (Stress-Strain):
$$ \underline{\underline{\sigma}} = \lambda \ tr( \underline{\underline{\epsilon}}) \mathbf{I}_n + 2\mu \underline{\underline{\epsilon}}$$
Strain-displacement: 
$$ \underline{\underline{\epsilon}} = \nabla \underline{u}$$
Equilibrium : 
$$\nabla \underline{\underline{\sigma}}  + \underline{b}= \underline{0}$$

Note that $\underline{\underline{\sigma}}$ is the stress tensor, $\underline{\underline{\epsilon}}$ is the strain tensor
and $\underline{u} $ is the displacement vector.
Here-in-after, the vectors will be underlined, like $\underline{u} $, and tensor will have two underlines, like $\underline{\underline{\sigma}}$ . 

$\nabla$ is the Del symbol, which can represent a gradient, divergence or curl of a vector, depending on the operator. In this context, $\nabla \underline{v}$ is a matrix with $ \frac{\partial v_j }{\partial x_i } $  as its $(i,j)$ component, and $\nabla A $ is a matrix with $ \frac{\partial A_{i,j} }{\partial x_i } $  as its $(i,j)$ component. 

$\lambda $ and $\mu$ are the two Lamé constants, describing the mechanical property of the continuum.

$\mathbf{I}_n$ is an identity matrix of n dimentions.

<!-- $ = \Sigma_i \frac{\partial v_i}{\partial x_i} \underline{e_i} $, where $ \underline{e_i}  $ is the unit vector in direction of the $i$ th axis on the Cartesian coordinate system. -->

## Definition of curvilinear coordinate system
Suppose a cartesian coordinate system with $n$ axis, described with $n$ basis vectors $ \underline{e_{x_i}}  $, where $i \in \mathcal{N} = \{1,2,..., n\} $. Then, a vector $\underline{x}$ on this coordinate system is described as $\underline{x} = \sum_{i\in{\mathcal{N}}}  x_i\underline{e_{x_i}} = 
\begin{pmatrix}x_1 & x_2 & ... & x_n\end{pmatrix}^\top $ . 

Let $\underline{\phi}: \mathbb{R}^n \rightarrow \mathbb{R}^n$ be a coordinate transformation function. Then a coordinate transformation from $\underline{x}$ to $\underline{y}$ can be described as $y_j= \underline{\psi_j } (x_1, x_2, ..., x_n)$ , where $y_i$ is the coefficient of the $i$ th basis vector of the linear combination of $ \underline{y} = \Sigma_{j\in \mathcal{N}} \ y_j\underline{e_{y_j}} $ . The basis vectors $\underline{e_{y_j}}$ can be derived by 
$$\underline{e_{y_j}}= \frac{\partial \underline{x}  }{\partial y_j} = 
\begin{pmatrix}\frac{\partial x_1}{\partial y_j} & \frac{\partial x_2 }{\partial y_j} & ... & \frac{\partial x_n  }{\partial y_j} \end{pmatrix}^\top
=\sum_{i\in \mathcal{N}} \frac{\partial x_i  }{\partial y_j} \underline{e_{x_i}} .$$
Note that this expression describes the basis vector in the direction of the $i$ th axis in the original Cartesian coordinate system {$\underline{e_{x_i}} \ | \ i \in \mathcal{N}$}.

Under Cartesian coordinate system, the basis vectors are constant, namely, don't change from point to point, while in general, they depend on the point. Thus we can calculate the partial deriviation of $\underline{e_{y_j}}$ as follows:
$$ \frac{\partial \underline{e_{y_j}}}{\partial y_k} = 
\begin{pmatrix}\frac{\partial^2 x_1}{\partial y_k \partial y_j} & \frac{\partial^2 x_2}{\partial y_k \partial y_j} &  ... & \frac{\partial^2 x_n}{\partial y_k \partial y_j}   \end{pmatrix}^\top . $$

By defining a new symbol by $\Gamma^{x_i}_{y_j,y_k} =\frac{\partial^2 x_i}{\partial y_k \partial y_j}$, also known as Christoffel symbol, this equation is transformed in to the next form:
$$\frac{\partial \underline{e_{y_j}}}{\partial y_k} = 
\begin{pmatrix}\Gamma^{x_1}_{y_j,y_k} & \Gamma^{x_2}_{y_j,y_k} &  ... & \Gamma^{x_n}_{y_j,y_k}   \end{pmatrix}^\top . $$

On the other hand, the integral of an arbitraty function $\phi$ over a domain $\Omega_x$ is calculated as follows:
$$ \int_{\Omega_x} \phi(x_1,x_2,...,x_n)dx_1dx_2...dx_n \\
= \int_{\Omega_y} \phi(y_1,y_2,...,y_n) \left|\frac{\partial (x_1,x_2,...,x_n)}{\partial (y_1,y_2,...,y_n) } \right| dy_1dy_2...dy_n ,$$
where 
$$\frac{\partial (x_1,x_2,...,x_n)}{\partial (y_1,y_2,...,y_n) } = \begin{bmatrix}
\frac{\partial x_1}{ \partial y_1} & \frac{\partial x_2}{ \partial y_1} & \dots& \frac{\partial x_n}{ \partial y_1} \\
\frac{\partial x_1}{ \partial y_2} & \frac{\partial x_2}{ \partial y_2} & \dots& \frac{\partial x_n}{ \partial y_2} \\
\vdots & \vdots & \ddots& \vdots \\
\frac{\partial x_1}{ \partial y_n} & \frac{\partial x_2}{ \partial y_n} & \dots& \frac{\partial x_n}{ \partial y_n}   
\end{bmatrix} = \mathbf{J}^x_y, $$
also known as the Jacobian matrix.

## Derivation of weak form of the problem

