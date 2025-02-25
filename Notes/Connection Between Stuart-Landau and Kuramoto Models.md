## 1. The Models

### Stuart-Landau (SL)
```
dz/dt = (λ + iω)z - |z|²z + K Σⱼ(zⱼ - z)
```
where z = re^(iθ) is a complex number

### Kuramoto
```
dθᵢ/dt = ωᵢ + K Σⱼ sin(θⱼ - θᵢ)
```
where θ is just the phase

## 2. Key Connection: The Kuramoto Model is a Special Case

We can derive the Kuramoto model from the Stuart-Landau model through several steps:

### Step 1: Write SL in Polar Form
Writing z = re^(iθ), the SL equation separates into:

```
dr/dt = λr - r³ + K Σⱼ(rⱼcos(θⱼ - θ) - r)
dθ/dt = ω + K Σⱼ(rⱼ/r)sin(θⱼ - θ)
```

### Step 2: Strong Amplitude Constraint
If we assume:
1. λ > 0 (above Hopf bifurcation)
2. Very strong amplitude stability (r ≈ √λ quickly)
3. All oscillators have same amplitude (rⱼ = r)

Then the amplitude equation becomes "slaved" to a fixed value r = √λ

### Step 3: Phase Equation
Under these conditions, the phase equation becomes:

```
dθ/dt = ω + K Σⱼsin(θⱼ - θ)
```

This is exactly the Kuramoto model!

## 3. Key Differences

1. Degrees of Freedom:
   - SL: Both amplitude and phase (2 per oscillator)
   - Kuramoto: Only phase (1 per oscillator)

2. Coupling:
   - SL: Can have amplitude effects
   - Kuramoto: Only phase interactions

3. Dynamics:
   - SL: Can show amplitude death, chaos
   - Kuramoto: Only phase synchronization

## 4. What SL Can Do That Kuramoto Cannot

1. Amplitude Death:
   - Oscillators can completely stop (r → 0)
   - Important in biological systems

2. Mixed Mode Oscillations:
   - Different oscillators can have different amplitudes
   - Allows for more complex patterns

3. Amplitude-Phase Coupling:
   - Changes in amplitude affect phase and vice versa
   - More realistic for many physical systems

4. Bifurcation Analysis:
   - Can study transition between oscillatory and non-oscillatory states
   - More complete picture of system dynamics

## 5. Implications for Our Implementation

In our flexible kernel implementation:

1. When amplitude effects are weak (λ >> 0, strong stability):
   - Behavior similar to Kuramoto
   - Coupling mainly affects phases

2. When amplitude effects are strong:
   - New patterns emerge
   - Can see localized amplitude changes
   - More complex response to "attacks"

3. Choice of Parameters:
   - High λ, weak coupling → Kuramoto-like
   - Low λ, strong coupling → Full SL dynamics

## 6. Experimental Tests

To see the connection:

1. Set high λ and weak coupling in SL model
2. Watch only the phase patterns
3. Compare with Kuramoto model
4. Should see similar synchronization patterns

To see SL-specific effects:

1. Lower λ near bifurcation point
2. Increase coupling strength
3. Look for amplitude variations
4. Try inducing amplitude death

This relationship helps us understand when simpler phase-only models are sufficient and when we need the full amplitude-phase dynamics of Stuart-Landau.