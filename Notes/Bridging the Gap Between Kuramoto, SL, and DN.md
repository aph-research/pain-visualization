The key insight is that the DN model from the paper and the coupling kernel approach are describing the same thing at different levels:

1. The DN model describes how individual neural populations respond:
$$ activation = (aG₁ ⊗ z + b)/(cG₂ ⊗ z + d)$$
where G₁,G₂ are Gaussian kernels describing spatial influence.

1. The Kuramoto/SL coupling kernel approach describes how oscillators influence each other:
$$ Cᵢ = (1/Nᵢ)(Σₛ Σⱼ Kₛ sin(θⱼ - θᵢ)) $$
where Kₛ are coupling strengths at different distances.

The bridge is that **the Gaussian kernel in DN (G₁) can be used to define the coupling strengths (Kₛ) in the oscillator model**. So instead of discrete distance-based coupling constants, you'd have:

$$ Kₛ = G(s) = exp(-s²/2σ²) $$

where s is the distance between oscillators and σ controls the spatial spread.

For your SL model, this would mean:
1. Keep the basic SL dynamics for each oscillator
2. Use a Gaussian kernel to define the coupling strengths
3. Add the baseline term b from DN

The full equation might look like:
$$ dz/dt = (λ + iω - |z|²)z + [aG₁ ⊗ z + b] + I(t) $$
This bridges the gap by using DN's spatial organization principles to inform how the oscillators should couple, while maintaining the rich dynamics of the SL model.