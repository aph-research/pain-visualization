# Stuart-Landau Oscillator Network with Flexible Coupling: Model and Implementation Details

## Theoretical Model

### Core Mathematical Framework

The Stuart-Landau oscillator is a canonical model for systems near a Hopf bifurcation, described by a complex-valued differential equation:

```
dz/dt = (λ + iω - |z|²)z + coupling + I(t)
```

where:
- z is a complex number representing both amplitude (|z|) and phase (arg(z))
- λ is the excitability parameter
- ω is the natural frequency
- |z|² provides amplitude saturation
- I(t) represents external input

### Coupling Architecture

The model implements a flexible coupling scheme where oscillators can interact at multiple spatial scales:

1. Distance-based kernels:
   - First neighbors (immediate adjacent)
   - Second neighbors (diagonal and two-step)
   - Third neighbors (extended range)

2. Each kernel has an independently adjustable coupling strength (K₁, K₂, K₃)

3. Coupling influence is computed through convolution with these kernels:
   ```
   coupling = Σᵢ Kᵢ * (kernel_i ⊗ z)
   ```
   where ⊗ represents convolution

### Cluster Headache Attack Simulation

The model includes a mechanism to simulate localized pathological activity:

1. Normal state:
   - Uniform λ across the grid
   - No external input (I(t) = 0)

2. Attack state (in central region):
   - Amplified excitability (5λ)
   - Constant external input (I = 1.0)
   - Added noise (random fluctuations)

## Implementation Details

### Grid Structure

- 50x50 grid of oscillators
- Each oscillator stores:
  - Real component (z.real)
  - Imaginary component (z.imag)
- Periodic boundary conditions

### Coupling Kernels

1. First neighbor kernel:
```
[0 1 0]
[1 0 1]
[0 1 0]
```

2. Second neighbor kernel:
```
[0 0 1 0 0]
[0 1 0 1 0]
[1 0 0 0 1]
[0 1 0 1 0]
[0 0 1 0 0]
```

3. Third neighbor kernel (7x7 matrix with similar pattern)

### Numerical Integration

The system uses Euler integration with a small time step (dt = 0.05):

```javascript
z(t + dt) = z(t) + dz/dt * dt
```

where dz/dt includes:
- Linear growth term (λz)
- Rotation (iωz)
- Saturation (-|z|²z)
- Coupling influences
- External input during attack

### Visualization

- Phase mapped to HSL color hue
- Amplitude mapped to brightness
- Real-time rendering at animation frame rate
- Interactive control through UI sliders

### Control Parameters

User-adjustable parameters:
1. Base System:
   - λ (excitability): 0-0.5
   - ω (frequency): 0-2

2. Coupling:
   - K₁ (first neighbors): -1 to 1
   - K₂ (second neighbors): -1 to 1
   - K₃ (third neighbors): -1 to 1

3. Attack Trigger:
   - Binary state (on/off)
   - Affects central 10x10 region

## Key Features and Behaviors

1. Normal Operation:
   - Oscillators tend to synchronize based on coupling patterns
   - Amplitude remains bounded due to saturation term
   - Pattern formation depends on coupling kernel configuration

2. During Attack:
   - Increased local activity in central region
   - Pattern propagation through coupling
   - Possible emergence of traveling waves or standing patterns

3. Parameter Sensitivity:
   - λ controls baseline activity
   - Coupling strengths determine pattern formation
   - ω affects rotation speed of phases

## Applications and Extensions

The model can be used to study:
1. Pattern formation in coupled oscillator systems
2. Propagation of localized excitation
3. Effects of multi-scale coupling on system dynamics
4. Pathological synchronization phenomena

Possible extensions include:
1. Additional coupling kernels
2. Heterogeneous natural frequencies
3. Multiple attack regions
4. More sophisticated integration schemes
5. Analysis tools for pattern characterization

# Derivation of the Stuart-Landau Equation

## 1. Starting Point: Hopf Bifurcation

The Stuart-Landau equation describes the normal form of a Hopf bifurcation. Let's derive it step by step.

### 1.1 Basic Concepts

A Hopf bifurcation occurs when:
- A system has a fixed point
- A pair of complex conjugate eigenvalues crosses the imaginary axis
- The system transitions from stable to unstable (or vice versa)

## 2. Two-Dimensional System

Let's start with a general 2D dynamical system near a fixed point:

```
dx/dt = f(x,y,μ)
dy/dt = g(x,y,μ)
```

where μ is our bifurcation parameter.

### 2.1 Linear Analysis

Near the fixed point (assumed to be at origin), we can write the Taylor expansion:

```
dx/dt = ax + by + higher order terms
dy/dt = cx + dy + higher order terms
```

The characteristic equation is:
```
|a-λ   b  |
|c    d-λ | = 0
```

For Hopf bifurcation, we need eigenvalues:
λ = α(μ) ± iω(μ)
where α(0) = 0 and ω(0) ≠ 0

## 3. Complex Form

### 3.1 Change to Complex Variable

Introduce z = x + iy. This transforms our 2D real system into a 1D complex system.

In terms of z, near the bifurcation point:
```
dz/dt = (α + iω)z + nonlinear terms
```

### 3.2 Nonlinear Terms

The most general form including cubic terms (lowest order nonlinear terms that can stabilize the system):
```
dz/dt = (α + iω)z + a₁z² + a₂|z|z + a₃z̄² + a₄|z|²z
```

### 3.3 Normal Form Reduction

Through a series of near-identity transformations, we can eliminate most nonlinear terms except |z|²z.

The key steps:
1. Remove z² terms through transformation z → z + hz²
2. Remove z̄² terms through z → z + kz̄²
3. Keep |z|²z term as it cannot be eliminated

## 4. Final Form

After all transformations, we get the Stuart-Landau equation:

```
dz/dt = (λ + iω)z - (1 + iγ)|z|²z
```

where:
- λ is the bifurcation parameter (previously α)
- ω is the natural frequency
- γ is the nonlinear frequency correction
- The coefficient of |z|²z can be normalized to 1 by rescaling

## 5. Physical Interpretation

### 5.1 In Polar Form

Writing z = reiθ, the equation separates into:

```
dr/dt = λr - r³
dθ/dt = ω - γr²
```

This shows:
1. Amplitude dynamics (r):
   - For λ < 0: decay to zero
   - For λ > 0: growth until saturation at r = √λ

2. Phase dynamics (θ):
   - Constant rotation at frequency ω
   - Amplitude-dependent frequency shift (-γr²)

### 5.2 Key Properties

1. For λ < 0:
   - Origin is stable
   - Any perturbation decays

2. For λ > 0:
   - Origin becomes unstable
   - System settles onto limit cycle with radius √λ
   - Oscillation frequency = ω - γλ

## 6. Connection to Our Implementation

In our numerical implementation:
1. We normalize γ = 0 for simplicity
2. The |z|²z term is implemented explicitly
3. We add coupling terms and external input I(t)

Final implemented form:
```
dz/dt = (λ + iω)z - |z|²z + coupling + I(t)
```

This gives us a system that can:
1. Spontaneously oscillate when λ > 0
2. Maintain stable amplitude through the nonlinear term
3. Interact with neighbors through coupling
4. Respond to external perturbations via I(t)

# Connection Between Stuart-Landau and Kuramoto Models

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

# Intuitive Understanding of Stuart-Landau Parameters

Let's break down the Stuart-Landau equation term by term:

```
dz/dt = (λ + iω)z - |z|²z + coupling + I(t)
```

## The Basic Parameters

### λ (Lambda): "Excitability"

Think of λ as the "eagerness to oscillate":
- When λ < 0: System wants to rest (decay to zero)
- When λ = 0: Critical point (knife edge)
- When λ > 0: System wants to oscillate

Real-world analogies:
- A playground swing: λ < 0 is like having friction (swing stops)
- A violin string: λ > 0 is like adding energy with the bow
- A neuron: λ represents how close it is to firing threshold

### ω (Omega): "Natural Frequency"

This is simply how fast the oscillator wants to spin:
- Low ω: Slow rotation (like a lazy carousel)
- High ω: Fast rotation (like a spinning top)
- In neurons: Might represent preferred firing frequency
- In chemical reactions: Natural rhythm of reaction cycle

## The Nonlinear Term: -|z|²z

This is the "amplitude control":
- Without it, oscillations would grow forever
- With it, amplitude stabilizes at r = √λ

Think of it like:
- A governor on an engine
- The limits of how far a swing can go
- Maximum amplitude of a violin string

## Complex Variable z = x + iy

Writing z = re^(iθ) gives:
- r: How big the oscillation is
- θ: What phase it's in

Like:
- A point going around in a circle
- r is distance from center
- θ is position on the circle

## Understanding the Dynamics

### When λ > 0:

1. Small amplitude (r << √λ):
   - (λ - r²) is positive
   - Amplitude grows

2. Large amplitude (r >> √λ):
   - (λ - r²) is negative
   - Amplitude shrinks

3. At r = √λ:
   - Perfect balance
   - Stable oscillation

### Phase Dynamics:

1. Without coupling:
   - Rotates at frequency ω
   - Like a perfect clock

2. With coupling:
   - Neighbors pull on phase
   - Can synchronize or form patterns

## The Extra Terms

### Coupling Term:

In our implementation:
```
K₁(neighbors at distance 1) +
K₂(neighbors at distance 2) +
K₃(neighbors at distance 3)
```

Like:
- Springs between oscillators
- Communication between neurons
- Heat diffusion in material

### External Input I(t):

- Direct forcing of the system
- In our "attack" simulation:
  * Represents strong local excitation
  * Like poking a pond with your finger
  * Or triggering a neuron directly

## Practical Effects in Our Simulation

### Visual Indicators:

1. High λ:
   - Bright colors (large amplitude)
   - Strong, stable patterns

2. Low λ:
   - Dim colors
   - Patterns might die out

3. High ω:
   - Fast color changes
   - Rapid phase rotation

4. Strong coupling:
   - Smooth patterns
   - Synchronized regions

5. During "attack":
   - Very bright center
   - Patterns emanating outward

## Common Parameter Combinations

1. "Orderly" Behavior:
   - Moderate λ (~0.1)
   - Low coupling
   - Creates stable, gentle patterns

2. "Chaotic" Behavior:
   - High λ
   - Strong coupling
   - Creates complex, changing patterns

3. "Dead" Zones:
   - Very low λ
   - Any coupling
   - Little to no activity

4. "Propagating Waves":
   - Moderate λ
   - Balanced coupling
   - Creates traveling patterns