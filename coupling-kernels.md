## Coupling Kernels

We noticed that during 5-MeO-DMT experiences, “everything wanted to synchronize with everything else.” While people describe other psychedelics similarly, in retrospect, the phenomenon was quite different. On mushrooms, it is not *exactly* the case that everything wants to synchronize with everything else. It’s more that things alternate between wanting to synchronize or anti-synchronize (i.e., be in opposite phases). And not everything is interacting with everything! At least not as obviously as with 5-MeO-DMT.

The first hint of this kind of effect came from the [Tracer Tool](https://psychophysics.qri.org/) (*Gómez Emilsson 2020*), where some people reported that while DMT’s tracers were very colorful, 5-MeO-DMT tended to be monochrome, favoring black or white. DMT’s tracers typically feature fast alternating complementary colors, such as neon green and magenta, which give the impression of being polar opposites on a linearized color wheel.

The thought occurred to me at the time (see the section “Modulating Lateral Inhibition” in [Gomez-Emilsson 2020](https://qri.org/blog/5meo-vs-dmt)), that DMT might increase lateral inhibition and in turn increase the contrast between features across the predictive coding hierarchy. In contrast, 5-MeO-DMT might relax lateral inhibition, and thus reduce contrast across the entire hierarchy.

During the retreat, there was a general consensus that this explanation made a lot of sense and was congruent with people’s phenomenology.

A couple months after the retreat I started developing methods to interact with coupled oscillators in real time to figure out how to reproduce the characteristic effects we uncovered. Early this year, I came up with the idea of implementing coupled oscillators whose coupling constant across different distances can be modulated by the user—in other words, the idea was to develop a user experience that empowers an individual to modulate the geometric *coupling kernel* around each oscillator. And with a very broad, directional hypothesis, I did this with the aim of visualizing DMT and 5-MeO-DMT effects.

The first iteration of this concept can be found here (make sure to play some music along with it for maximum enjoyment!): [Coupling Kernels 2D Grid](https://qri.org/demo/coupled_kernels)

Here, the Coupling Kernel is applied:

**5-MeO-DMT Coupling Kernels Pattern**  
[Video: /images/distill/cessation-simulations/image12.webm](#)  
*Your browser does not support the video tag.*

**N,N-DMT Coupling Kernels Pattern**  
[Video: /images/distill/cessation-simulations/image10.webm](#)  
*Your browser does not support the video tag.*

| Neighbor Type                              | Coupling Constant | 5-MeO-DMT    | N,N-DMT         |
|--------------------------------------------|-------------------|--------------|-----------------|
| Immediate Neighbors (Distance = 1)         | K<sub>1</sub>     | 20           | -10             |
| Neighbors at Distance = 2                  | K<sub>2</sub>     | 20           | 10              |
| Neighbors at Distance = 3                  | K<sub>3</sub>     | 10           | -20 to +20      |
| Neighbors at Distance = 4                  | K<sub>4</sub>     | 0            | 5               |
| Small-World Neighbors (Random Connections) | K<sub>SW</sub>    | -20 to 20    | 0               |

### Details of the Implementation

This simulation models a grid of Kuramoto oscillators, each representing a cell with a specific natural frequency and phase that evolves over time. These oscillators interact with their neighbors through coupling constants, which vary based on the (Manhattan) distance between oscillators.

### Initialization and Connectivity

At the beginning of the simulation:

- Each cell is assigned a preferred frequency with spatial autocorrelation.
- Oscillators are connected to neighbors up to a Manhattan distance of 4.
- Each oscillator is also connected to two randomly chosen cells, introducing a “small-world” aspect to the network.

### User Interface

The simulation includes five interactive sliders:

- Four sliders modulate local coupling constants at different Manhattan distances (1 to 4).
- One “Small World” slider modulates the coupling constant for the random long-range connections.

### Mathematical Model

**Phase Update Equation**

For each oscillator \(i\), the phase \(\theta_i\) is updated at each time step \(\Delta t\) according to:

\[
\theta_i(t + \Delta t) = \theta_i(t) + [\omega_i + C_i] \Delta t
\]

Where:

- \(\theta_i(t)\): Current phase of oscillator \(i\).
- \(\omega_i\): Natural frequency of oscillator \(i\).
- \(C_i\): Cumulative influence on oscillator \(i\) from all of its neighbors.

### Coupling Term

The cumulative influence on oscillator \(i\) from all its neighbors is given by:

\[
C_i = \frac{1}{N_i} \left( \sum_{s=1}^{4} \sum_{j \in D_s} K_s \sin(\theta_j - \theta_i) + \sum_{k \in \text{SW}_i} K_{\text{SW}} \sin(\theta_k - \theta_i) \right)
\]

Where:

- \(D_s\): Set of neighbors at distance \(s\).
- \(K_s\): Coupling constant for neighbors at distance \(s\).
- \(\text{SW}_i\): Set of small-world neighbors for oscillator \(i\).
- \(N_i\): Total number of neighbors (including small-world connections).

### Visualization

Each oscillator’s phase \(\theta_i\) is mapped to a color using the CIELAB color space, providing a visual representation of phase patterns across the grid.

We are currently exploring the applications of the *Coupling Kernels* idea in a number of ways. For the purpose of this essay, we focused on how it could capture features of *cessations*. In particular, we combined it with the following idea (Dimensionality Interactions) to simulate *Standing Waves Across Multiple Fields* ([see below](http://qri.org/blog/cessation-simulations#standing-wave-pattern-across-multiple-fields-simplification-via-standing-wave-patterns)).

## Dimensionality Interactions: Metric Sharing

[![Dimensionality Interactions: Metric Sharing](https://img.youtube.com/vi/BRVH-LMCBkA/0.jpg)](https://www.youtube.com/embed/BRVH-LMCBkA)

The [QRI’s 4D Wave Simulator](https://qri.org/demo/4d_wave_control) can help us develop an intuition for how matrix multiplication results in changes to a lattice in 4D space (using color as a signal to encode information about a fourth spatial dimension).

One of the main outcomes of the 5-MeO-DMT retreat in Canada (2023) were the countless discussions about the way the visual and somatic fields map on to each other. This was not something I in particular focused on, but Cube Flipper, Asher, and Roger Thisdell, among others, all gave this a whole lot of thought and consideration. In general, it was agreed that changes in this mapping was one of the features of the experience, as if the relationship between one’s body schema and visual field was revealed to be one of the ingredients for constructing one’s world simulation all along.

The key insight here is that there seem to be waves that can travel in the visual field and waves that can travel in the somatic field, and that their difference is part of what constructs our world simulation. Tuning into states of the system where waves are seamlessly interfacing between these two spaces will in turn be closer to cessation (and generically higher in valence, as per STV).

In the context of the necessity of an interface between two spaces of different dimensionality (e.g. 3D somatic space and 2.5D visual space) and where the system exhibits resonant modes emergent from the wiring (and coupling constant) of coupled oscillators, the system as a whole would then be predicted to minimize its energy precisely where the waves are in a low energy configuration in both the spaces that are interfacing at once. That is, one ought to achieve a projection of one space onto the other and identify a coupling kernel for each of these spaces so that the resulting space behaves as if there was only one space. When this level of coherence is achieved (good projection + good kernel) the resulting waves cannot even notice there is a projective dynamic going on. There simply is no “internal distinction” that can be found, in an otherwise really complex system that tends to find a lot of differences between the spaces it is mapping to each other.

I think that one of the things that is elegant about this framework is that it predicts that there will be likely quite specific and unique cessation conditions. Namely, combinations of spaces of different dimensions having the right projection and the right coupling kernels for them to have waves emerge that behave in the exact same way in both spaces. Perhaps some of the more exotic cessations, such as those involving a toroidal space spinning and turning inside-out, correspond to some of the less intuitive “solutions” to this dynamic. Namely, where, for example, the somatic field finds a mapping to the visual field such that the waves in both do exactly the same thing and thus any sense of distinction is briefly lost.
