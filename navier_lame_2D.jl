using DifferentialEquations, ModelingToolkit, DiffEqOperators, DomainSets
using NeuralPDE, Flux, GalacticOptim, DiffEqFlux, Optim
using Quadrature, Cubature
#Youngs modulus and Poisson ratio
ym = 0.001
pr = 0.47
#Lame Parameters
lamda =(pr * ym)/((1+pr)*(1-(2*pr)))
mu = ym/(2*(1+pr))
#defining parameters, variables and equations
@parameters x1 x2
@variables u1(..) f1(..) u2(..) f2(..)

D1 = Differential(x1)
D11 = Differential(x1)^2
D2 = Differential(x2)
D22 = Differential(x2)^2

eq = [(-1)*(lamda+mu)*D11(u1(x1,x2)) + (lamda+mu)*D1(D2(u2(x1,x2))) + mu*D11(u1(x1,x2)) + mu*D22(u2(x1,x2)) ~ f1(x1,x2),
      (-1)*(lamda+mu)*D2(D1(u1(x1,x2))) + (lamda+mu)*D22(u2(x1,x2)) + mu*D11(u1(x1,x2)) + mu*D22(u2(x1,x2)) ~ f2(x1,x2)]

boundary = [u1(0,x2) ~ 0, u1(1,x2) ~ 1,
            u2(x1,0) ~ 0, u2(x1,1) ~ 1]

domains = [x1 ∈ Interval(0.0, 1.0),
           x2 ∈ Interval(0.0, 1.0)]

input_ = length(domains)
n = 15
chain = [FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:7]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

grid_strategy = NeuralPDE.GridTraining(0.01)
discretization = NeuralPDE.PhysicsInformedNN(chain,grid_strategy,init_params= initθ)
@named pdesys = PDESystem(eq, boundary, domains,[x1,x2],[u1(x1,x2),u2(x1,x2),f1(x1,x2),f2(x1,x2)])

prob = NeuralPDE.discretize(pdesys,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pdesys,discretization)

cb = function(p,l)
      println("Current Loss: $l")
      return false
end

res = GalacticOptim.solve(prob, ADAM(0.001); cb = cb, maxiters=2000)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=10000)

phi = discretization.phi
