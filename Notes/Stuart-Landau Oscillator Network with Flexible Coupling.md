## Theoretical Model

### Core Mathematical Framework

The Stuart-Landau oscillator is a canonical model for systems near a Hopf bifurcation, described by a complex-valued differential equation:

$$ dz/dt = (λ + iω - |z|²)z + coupling + I(t) $$

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
 
$$  coupling = Σᵢ Kᵢ * (kernel_i ⊗ z) $$

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

$$ z(t + dt) = z(t) + dz/dt * dt $$

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