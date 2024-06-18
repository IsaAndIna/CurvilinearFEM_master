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
Suppose a cartesian coordinate system with $n$ axis, described with $n$ basis vectors $ \underline{e_{x_i}}  $, where $i \in \{1,2,..., n\} $. Then, a vector $\underline{x}$ on this coordinate system is described as $\underline{x} = \Sigma_{i\in{\{1,2,...,n\}}} x_i\underline{e_{x_i}} = 
\begin{pmatrix}x_1 & x_2 & ... & x_n\end{pmatrix}^T$ . 

Let $\underline{\phi}: \mathbb{R}^n \rightarrow \mathbb{R}^n$ be a coordinate transformation function. Then a coordinate transformation from $\underline{x}$ to $\underline{y}$ can be described as $y_i= \underline{\psi_i } (x_1, x_2, ..., x_n)$ , where $y_i$ is the coefficient of the $i$ th basis vector of the linear combination of $ \underline{y} = \Sigma_{i\in{\{1,2,...,n\}}} y_i\underline{e_{y_i}} $ . The basis vectors $\underline{e_{y_i}}$ can be derived by $$\underline{e_{y_i}}= \frac{\partial \underline{x}  }{\partial y_i} = \begin{pmatrix}\frac{\partial x_1}{\partial y_i} & \frac{\partial x_2 }{\partial y_i} & ... & \frac{\partial x_n  }{\partial y_i} \end{pmatrix}^T .$$


## Derivation of weak form of the problem

