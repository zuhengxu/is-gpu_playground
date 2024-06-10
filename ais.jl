using Distributions
using Random
using LinearAlgebra
# target distirbution
# Gaussina target N(0, Id)

d
target = McDiagMvNormal(zeros(2), ones(2))

# transition kernel


# weight calculation


# resampling operator






# IS benchmark


# GPU faster on simulating dynamics (langevin/HMC)


# NEO IS (no rejection operation, just weight calculation)




# AIS with Langevin or HMC
# what if we dont code the weights calculation in broadcast way?


# let's benchmark multinomial resampling speed


# on SMC

