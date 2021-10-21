using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x
Dt = Differential(t)
Dx = Differential(x)
@variables u1(..), u2(..), u3(..)
@variables Dxu1(..),Dtu1(..),Dxu2(..),Dtu2(..)

eqs_ = [Dt(Dtu1(t,x)) ~ Dx(Dxu1(t,x)) + u3(t,x)*sin(pi*x),
        Dt(Dtu2(t,x)) ~ Dx(Dxu2(t,x)) + u3(t,x)*cos(pi*x),
        exp(-t) ~ u1(t,x)*sin(pi*x) + u2(t,x)*cos(pi*x)]

bcs_ = [u1(0.,x) ~ sin(pi*x),
       u2(0.,x) ~ cos(pi*x),
       Dt(u1(0,x)) ~ -sin(pi*x),
       Dt(u2(0,x)) ~ -cos(pi*x),
       #Dtu1(0,x) ~ -sin(pi*x),
      # Dtu2(0,x) ~ -cos(pi*x),
       u1(t,0.) ~ 0.,
       u2(t,0.) ~ exp(-t),
       u1(t,1.) ~ 0.,
       u2(t,1.) ~ -exp(-t)]

der_ = [Dt(u1(t,x)) ~ Dtu1(t,x),
        Dt(u2(t,x)) ~ Dtu2(t,x),
        Dx(u1(t,x)) ~ Dxu1(t,x),
        Dx(u2(t,x)) ~ Dxu2(t,x)]

bcs__ = [bcs_;der_]

input_ = length(domains)
n = 15
chain = [FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:7]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

grid_strategy = NeuralPDE.GridTraining(0.07)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             grid_strategy,
                                             init_params= initθ)

vars = [u1(t,x),u2(t,x),u3(t,x),Dxu1(t,x),Dtu1(t,x),Dxu2(t,x),Dtu2(t,x)]
@named pde_system = PDESystem(eqs_,bcs__,domains,[t,x],vars)
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents
bcs_inner_loss_functions = inner_loss_functions[1:7]
aprox_derivative_loss_functions = inner_loss_functions[9:end]

cb = function (p,l)
    println("loss: ", l )
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    println("der_losses: ", map(l_ -> l_(p), aprox_derivative_loss_functions))
    return false
end

res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb, maxiters=2000)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=10000)

phi = discretization.phi
