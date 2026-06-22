# Geometric Ricci Flow and Perelman Entropy

This repository contains the code for the paper "Geometric Ricci Flow and Perelman Entropy: A Unifying Framework for Cognitive and Economic Dynamics" by Marc Daghar.

## Overview

The framework implements:
1. Cognitive Ricci flow: ∂g/∂t = -2Ric(g) + λ∇_gL - μ(g-g₀)
2. Ollivier-Ricci curvature: κ_ij = 1 - W_1(μ_i, μ_j)/d(i,j)
3. Perelman entropy: W(g,f,τ) = ∫[τ(R_g+|∇f|²)+f-n](4πτ)^(-n/2)e^(-f)dμ_g
4. Entropy correspondence: S_total = α·W + β
5. Unified stability: d/dt(L_econ + W) ≤ 0

## Requirements

```bash
pip install -r requirements.txt
