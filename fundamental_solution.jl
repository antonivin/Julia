using LinearAlgebra
using Plots
using PlotlyJS

n = 10
Ux = (Float64,1:10:100,1:10:100,1:10:100)
Uy = (Float64,1:10:100,1:10:100,1:10:100)
Uz = (Float64,1:10:100,1:10:100,1:10:100)
Fx = zeros(Float64,n,n,n)
Fy = zeros(Float64,n,n,n)
Fz = zeros(Float64,n,n,n)

Fx[5,5,5] = 25
#Fz[2:5,2:5,2:5] .= 2

pr = 0.47
ym = 0.001
c = (1+pr)/(8*pi*ym*(1-pr))

Xo = [0 0 0]
Xp = [20 40 60]
r = Xp-Xo
n = r/norm(r)

CFx = (3-(4*pr))*Fx
CFy = (3-(4*pr))*Fy
CFz = (3-(4*pr))*Fz

dp = Fx*n[1]+Fy*n[2]+Fz*n[3]

cx = (CFx + n[1]*dp)/norm(r)
cy = (CFy + n[2]*dp)/norm(r)
cz = (CFz + n[3]*dp)/norm(r)

Ux = c*cx
Uy = c*cy
Uz = c*cz

quiver(Ux)
quiver!(Ux,Uy,Uz,quiver=(Fx,Fy,Fz))

#surf(Ux[:,1,1], Uy[:,1,1], Uz[:,1,1], supp = [Fx[:,1,1] Fy[:,1,1] Fz[:,1,1]], w = "vectors filled head", lw = 2,
#            Axes(xlabel = :x, ylabel = :y, zlabel = :z, view = (55, 62)))
