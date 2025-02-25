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

For each oscillator $θ_i$, the phase $θ_i$ is updated at each time step $\Delta t$ according to:

$$
θ_i(t + \Delta t) = θ_i(t) + [ω_i + C_i] \Delta t
$$

Where:
- $θ_i(t)$: Current phase of oscillator $i$
- $ω_i$: Natural frequency of oscillator $i$
- $C_i$: Cumulative influence on oscillator $i$ from all of its neighbors

### Coupling Term
The cumulative influence on oscillator $i$ from all its neighbors is given by:

$$
C_i = \frac{1}{N_i} \left( \sum_{s=1}^{4} \sum_{j \in D_s} K_s \sin(θ_j - θ_i) + \sum_{k \in \text{SW}_i} K_{\text{SW}} \sin(θ_k - θ_i) \right)
$$

Where:
- $D_s$: Set of neighbors at distance $s$
- $K_s$: Coupling constant for neighbors at distance $s$
- $\text{SW}_i$: Set of small-world neighbors for oscillator $i$
- $N_i$: Total number of neighbors (including small-world connections)

### Visualization

Each oscillator’s phase $\theta_i$ is mapped to a color using the CIELAB color space, providing a visual representation of phase patterns across the grid.

We are currently exploring the applications of the *Coupling Kernels* idea in a number of ways. For the purpose of this essay, we focused on how it could capture features of *cessations*. In particular, we combined it with the following idea (Dimensionality Interactions) to simulate *Standing Waves Across Multiple Fields* ([see below](http://qri.org/blog/cessation-simulations#standing-wave-pattern-across-multiple-fields-simplification-via-standing-wave-patterns)).