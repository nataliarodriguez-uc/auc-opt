using LinearAlgebra, Test
include("prox.jl")

f_prox_obj(x, delta, gamma; x0=+Inf) = min.(1,max.(0,x.-delta)) + (x-x0)^2/(2gamma)
f_prox(t::T, delta::T, gamma::T) where {T<:Real} = begin
  p=typemax(T)
  if t<delta # 1
    p=[t]
  elseif delta<=t && t<delta+gamma # 2
    if (gamma<=2) || (gamma>2 && t<delta+sqrt(2gamma))
      p=[delta]
    elseif (gamma>2) && (delta+sqrt(2gamma)<t)
      p=[t]
    elseif (gamma>2) && (t==delta+sqrt(2gamma))
      p=[t; delta]
    end
  elseif t==delta+gamma # 3
    if gamma<2
      p=[delta]
    elseif gamma==2
      p=[t; delta]
    elseif gamma>2
      p=[t]
    end
  elseif t>delta+gamma # 4
    if (gamma<2) && (t<1+delta+gamma/2)
      p=[t-gamma]
    elseif (gamma<2) && (t==1+delta+gamma/2)
      p=[t-gamma; t]
    elseif (gamma<=2 && t>1+delta+gamma/2) || (gamma>2)
      p=[t]
    end
  end
  return p
end

lo = -2//1
hi = 2//1
n = 400
T = Rational
ts = collect(range(lo, stop=hi, length=n+1))
tgrid = collect(range(lo, stop=hi, length=2n+1))
deltas = collect(range(lo/2, stop=hi/2, length=11))
gammas = [1//100; 1//10; 1//4; 1//2; 1//1; 11//10; 125//100; 150//100; 200//100; 210//100; 250//100; 3//1; 5//1; 10//1]
@testset "q(t)=gamma*min(max(t-delta),1) prox test" begin
for delta in T.(deltas)
  for gamma in T.(gammas)
    for x0 in T.(ts)
      prox_q = ProxQ([gamma], [delta]);
      v = [x0];
      prox = [0//1];
      Jprox = [0//1];
      int = [0];
      prox_q(prox, Jprox, int, v)

      fps = [f_prox_obj(t, delta, gamma;x0=x0) for t in T.(tgrid)];
      tstarsidx = findall(abs.(fps.-fps[argmin(fps)]).<1e-8)
      tstars = tgrid[tstarsidx]
      fstars = fps[tstarsidx]
      fproxs = f_prox(x0, delta, gamma)
      println("delta=$T($delta); gamma=$T($gamma); x0=$T($x0)")
      println("fproxs=$fproxs; tstars=$tstars")
      println("prox=$prox; tstars=$tstars\n")
      @test all(abs.(sort(fproxs) .- sort(tstars)) .<= (hi-lo)/n)
      @test minimum(abs.(prox .- tstars)) <= (hi-lo)/n
      # if all(abs.(sort(fproxs) .- sort(tstars)) .<= (hi-lo)/n) && minimum(abs.(prox .- tstars)) <= (hi-lo)/n
      #   continue
      # else
      #   push!(bad, delta, gamma, x0)
      # end
    end
  end
end
end # testset fprox
