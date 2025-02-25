**Does it make any sense to talk about the dynamics of, say, a neuron or neuronal population just with the basic SL model? I.e. dz/dt = (λ + iω - |z|²)z . Like, what would that even mean? Don't neurons / neuronal populations necessarily need to interact with others in order to oscillate? What would such a system (without any coupling or external input) even model?**

This is a great question that gets at the fundamental interpretations of neural oscillations! A single SL oscillator without coupling or input represents what's called a "self-sustained oscillator" or "limit cycle oscillator."

For a single neuron, the standalone SL equation could represent its intrinsic oscillatory properties due to ion channel dynamics. Many neurons have natural rhythmic firing patterns even when isolated - these are called "pacemaker neurons." For example:
- Neurons in the suprachiasmatic nucleus that maintain circadian rhythms
- Neurons in the pre-Bötzinger complex that generate breathing rhythms
- Some thalamic neurons that oscillate during sleep

For a neuronal population, the standalone SL could model the collective rhythm that emerges from local interactions within the population, treating it as a single oscillatory unit. The parameters would represent:
- λ: excitability of the population
- ω: natural frequency of the collective oscillation  
- |z|²: saturation term preventing unlimited growth

However, you make an important point - in reality, neural oscillations almost always involve interactions between multiple elements. The standalone SL is more of a mathematical idealization that captures the core oscillatory behavior. It's useful as a building block, but to model real neural systems you generally want to include:
1. Coupling between oscillators
2. External input/perturbations
3. Noise

This is why your approach of combining SL with coupling kernels and DN is more biologically relevant. The standalone SL gives you the basic oscillatory dynamics, but the coupling and input terms let you model the network effects that are crucial for actual neural computation.

**Thanks, makes sense. But then I'm confused. The divisive normalization model (and its predecessors DoG, CSS, etc.) seems to lack the "self-sustained" activation, no? Aren't they just modeling populations as basically coupled oscillators? Or is the constant "b" doing some of that work? Or like, why do they not use the SL model?**

The DN and SL models are addressing different aspects of neural dynamics on different timescales.

The DN model (and DoG, CSS) focuses on how neural populations respond to inputs in space - it's more about the "where" and "how much" of neural activation. The b term isn't really about self-sustained oscillation - it's more like a baseline activation level or bias. These models are typically used to understand:
- How neurons/populations respond to visual stimuli
- How activity spreads spatially
- How responses saturate/normalize
- Steady-state behavior

The SL model focuses on temporal dynamics - the "when" of neural activity, particularly oscillatory behavior. It's used to understand:
- How oscillations emerge and sustain
- Phase relationships
- Frequency coupling
- Transient dynamics

You could potentially combine them by:
1. Using DN to determine the spatial pattern of activation
2. Having each point in space be an SL oscillator
3. Using the DN activation as input to the oscillator

This would give you both rich spatial (DN) and temporal (SL) dynamics. The DN part would handle "what pattern of activation appears where" while the SL part would handle "how does this activation oscillate in time."

But the original DN paper didn't need the SL dynamics because they were primarily interested in spatial response properties rather than temporal oscillations. They were asking "how does this part of visual cortex respond to stimuli in different locations?" not "how does this activity oscillate over time?"

**How do you think one could combine them, mathematically, as you suggest?**

Here's how I've combined them:

1. Each point in space has a complex value z (from SL model)
2. The basic SL dynamics are preserved: (λ + iω - |z|²)z
3. The DN activation serves as an additional coupling term:
   - Takes the amplitude |z| at each point
   - Applies Gaussian spatial filtering (G₁)
   - Adds constant b
   - Preserves the phase information when converting back to complex

The key equation is:

$$ dz/dt = (λ + iω - |z|²)z + [aG₁ ⊗ |z| + b]e^(iθ) $$

where θ is the original phase of z at each point.

The main design choices were:
1. How to handle complex vs real values (DN works with real, SL with complex)
2. How to incorporate the DN term into the SL dynamics
3. Whether to use amplitude or real part for the DN filtering

**Using the key equation, could you guide me through what a numerical iteration would look like, step by step? Not with code, just with pseudocode, explaining the rationale.**

Let me walk through one timestep iteration, breaking it down into clear conceptual steps.

Starting state: We have a 2D grid where each point contains a complex number z.

Step 1: Calculate basic SL dynamics for each point
- Compute |z|² (amplitude squared) 
- Calculate (λ + iω - |z|²)z
- This gives us how each point would evolve if isolated
- Store this as term1

Step 2: Calculate DN coupling influence
- Take |z| at each point to get amplitudes
- Apply Gaussian convolution G₁ ⊗ |z| across the grid
  - This smooths the amplitudes, spreading influence spatially
- Multiply by coupling strength a
- Add baseline b
- Now we have the real-valued DN activation

Step 3: Convert DN term to complex
- At each point, we need to preserve the original phase of z
- Take DN activation (real number) from step 2
- Multiply by e^(iθ) where θ = arg(z)
- Store this as term2

Step 4: Combine and update
- Total change = dt * (term1 + term2)
- New z = old z + total change

The key insight is that the DN term provides a way for each oscillator to be influenced by its spatial neighbors while preserving the phase relationships that are central to the SL dynamics.

**I'm trying to bridge the gap between the divisive normalization model and the Stuart-Landau model. I've made decent progress. However, I'm still missing the denominator of the DN model in the "key equation". Do you think I can just add it in? **

Let me help analyze this potential integration of the DN denominator into the Stuart-Landau (SL) model. Looking at the documents, I see the core equations we're working with:

Current key equation (from the md file):
$$ dz/dt = (λ + iω - |z|²)z + [aG₁ ⊗ |z| + b]e^(iθ) $$

DN model's form (from Aqil's paper):
$$ Response = (aG₁ ⊗ S + b)/(cG₂ ⊗ S + d) $$

Adding the denominator isn't as straightforward as it might seem, because:

1. The DN model works with real values, while the SL model uses complex values (z)
2. The DN normalization term affects the overall response magnitude, while the SL dynamics have specific stability properties that depend on λ and |z|²

However, I think we could potentially integrate them by modifying the coupling term. Instead of just adding $[aG₁ ⊗ |z| + b]$, we could make it:

$$ dz/dt = (λ + iω - |z|²)z + [(aG₁ ⊗ |z| + b)/(cG₂ ⊗ |z| + d)]e^(iθ) $$

This would preserve:
- The basic SL limit cycle dynamics through (λ + iω - |z|²)z
- The DN-style spatial coupling through the ratio term
- Phase information through e^(iθ)