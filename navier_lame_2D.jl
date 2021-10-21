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

eq = [(-1)*(lamda+mu)*D11(u1(x1,x2)) + (lamda+mu)*D1(D2(u2(x1,x2))) + mu*D11(u1(x1,x2)) + mu*D22(u2(x1,x2)) ~ f1,
      (-1)*(lamda+mu)*D2(D1(u1(x1,x2))) + (lamda+mu)*D22(u2(x1,x2)) + mu*D11(u1(x1,x2)) + mu*D22(u2(x1,x2)) ~ f2]

boundary = [u1(0,x2) ~ 0, u1(1,x2) ~ 1,
            u2(x1,0) ~ 0, u2(x1,1) ~ 1]

domains = [x1 ∈ Interval(0.0, 1.0),
           x2 ∈ Interval(0.0, 1.0)]

dim = 2
dx = 0.1

chain  = FastChain(FastDense(dim,16,Flux.σ), FastDense(16,16,Flux.σ), FastDense(16,1))

init0 = Float64.(DiffEqFlux.initial_params(chain))

discretization = PhysicsInformedNN(chain,GridTraining(dx))

@named pdesys = PDESystem(eq, boundary, domains,[x1,x2],[u1,u2,f1,f2])

#order = 2
#discretization = MOLFiniteDifference([x1=>dx,x2=>dx],0)
prob = discretize(pdesys,discretization)
#sol = solve(prob, Tsit5(), saveat=0.1)

cb = function(p,l)
      println("Current Loss: $l")
      return flase
end

opt = Optim.BFGS()
res = GalacticOptim.solve(prob,opt;cb = cb, maxiters=1000)
#res = GalacticOptim.solve(prob, Fminbox(GradientDescent());cb = cb, maxiters=4000)
#res = GalacticOptim.solve(prob, Fminbox{GradientDescent}();cb = cb, maxiters=4000)
prob = remake(prob,u0=res.minimizer)
phi = discretization.phi
