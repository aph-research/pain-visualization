const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Grid parameters
let N = 100;  // Reduced grid size for performance
let cellSize;
const dt = 0.2;  // Larger time step for performance

// Use RK4 integration for stability with larger time step
function rk4Step(x, y, dt, f) {
    const k1 = f(x, y);
    const k2 = f(x + dt/2 * k1.real, y + dt/2 * k1.imag);
    const k3 = f(x + dt/2 * k2.real, y + dt/2 * k2.imag);
    const k4 = f(x + dt * k3.real, y + dt * k3.imag);
    
    return {
        real: dt/6 * (k1.real + 2*k2.real + 2*k3.real + k4.real),
        imag: dt/6 * (k1.imag + 2*k2.imag + 2*k3.imag + k4.imag)
    };
}

// Initialize canvas size and grid
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Fixed number of cells for performance
    N = 100; // Reduced from 200 in Kuramoto for better performance
    // Calculate spacing to cover full screen
    const spacingX = canvas.width / N;
    const spacingY = canvas.height / N;
    cellSize = {x: spacingX, y: spacingY};
    
    // Reinitialize the grid when resizing
    initializeGrid();
}

// Control parameters
let lambda = 0.1;  // Base excitability
let vizMode = 'both'; // Visualization mode
let omega = 0.15;  // Natural frequency
let coupling1 = 0.2;
let coupling2 = 0.1;
let coupling3 = 0.05;
let smallWorldCoupling = 0;
let isAttackActive = false;

// Complex field z and small world connections
let z;
let smallWorldNeighbors;

function initializeGrid() {
    // Initialize the complex field z
    z = Array(N).fill().map(() => 
        Array(N).fill().map(() => ({
            real: 0.3 * (Math.random() - 0.5),
            imag: 0.3 * (Math.random() - 0.5)
        }))
    );

    // Initialize small world connections
    smallWorldNeighbors = Array(N).fill().map(() => 
        Array(N).fill().map(() => {
            // Each cell gets two random connections
            const connections = [];
            for (let i = 0; i < 2; i++) {
                connections.push({
                    row: Math.floor(Math.random() * N),
                    col: Math.floor(Math.random() * N)
                });
            }
            return connections;
        })
    );
}

// Define coupling kernels for different distances
const kernels = {
    distance1: [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ],
    distance2: [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0]
    ],
    distance3: [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ]
};

function createGridArray() {
    return Array.from({ length: N }, () =>
        Array.from({ length: N }, () => ({ real: 0, imag: 0 }))
    );
}

function update() {
    // We'll perform a full-grid RK4 update where we recompute the coupling field at each sub-step.
    // dt is our time step.
    
    // ---- k1 Step ----
    const coupling_k1 = computeFlexibleCoupling(z, [coupling1, coupling2, coupling3]);
    const k1 = createGridArray();
    const z_temp1 = createGridArray(); // will hold z + (dt/2)*k1
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            // Local parameters (attack simulation remains unchanged)
            let localLambda = lambda;
            let localInput = 0;
            if (isAttackActive &&
                i >= N/2 - N/10 && i < N/2 + N/10 &&
                j >= N/2 - N/10 && j < N/2 + N/10) {
                localLambda = lambda * 5;
                localInput = 1.0;
                if (Math.random() < 0.1) {
                    localInput += Math.random() * 0.5;
                }
            }
            const current = z[i][j];
            const ns = current.real * current.real + current.imag * current.imag;
            const f_real = (localLambda - ns) * current.real - omega * current.imag + coupling_k1[i][j].real + localInput;
            const f_imag = (localLambda - ns) * current.imag + omega * current.real + coupling_k1[i][j].imag;
            k1[i][j] = { real: f_real, imag: f_imag };
            z_temp1[i][j] = { real: current.real + (dt/2) * f_real, imag: current.imag + (dt/2) * f_imag };
        }
    }
    
    // ---- k2 Step ----
    const coupling_k2 = computeFlexibleCoupling(z_temp1, [coupling1, coupling2, coupling3]);
    const k2 = createGridArray();
    const z_temp2 = createGridArray(); // will hold z + (dt/2)*k2 (using original z)
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            let localLambda = lambda;
            let localInput = 0;
            if (isAttackActive &&
                i >= N/2 - N/10 && i < N/2 + N/10 &&
                j >= N/2 - N/10 && j < N/2 + N/10) {
                localLambda = lambda * 5;
                localInput = 1.0;
                if (Math.random() < 0.1) {
                    localInput += Math.random() * 0.5;
                }
            }
            const current = z_temp1[i][j];
            const ns = current.real * current.real + current.imag * current.imag;
            const f_real = (localLambda - ns) * current.real - omega * current.imag + coupling_k2[i][j].real + localInput;
            const f_imag = (localLambda - ns) * current.imag + omega * current.real + coupling_k2[i][j].imag;
            k2[i][j] = { real: f_real, imag: f_imag };
            // For k2, we use the original z for the update: z_temp2 = z + (dt/2)*k2
            const orig = z[i][j];
            z_temp2[i][j] = { real: orig.real + (dt/2) * f_real, imag: orig.imag + (dt/2) * f_imag };
        }
    }
    
    // ---- k3 Step ----
    const coupling_k3 = computeFlexibleCoupling(z_temp2, [coupling1, coupling2, coupling3]);
    const k3 = createGridArray();
    const z_temp3 = createGridArray(); // will hold z + dt*k3 (using original z)
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            let localLambda = lambda;
            let localInput = 0;
            if (isAttackActive &&
                i >= N/2 - N/10 && i < N/2 + N/10 &&
                j >= N/2 - N/10 && j < N/2 + N/10) {
                localLambda = lambda * 5;
                localInput = 1.0;
                if (Math.random() < 0.1) {
                    localInput += Math.random() * 0.5;
                }
            }
            const current = z_temp2[i][j];
            const ns = current.real * current.real + current.imag * current.imag;
            const f_real = (localLambda - ns) * current.real - omega * current.imag + coupling_k3[i][j].real + localInput;
            const f_imag = (localLambda - ns) * current.imag + omega * current.real + coupling_k3[i][j].imag;
            k3[i][j] = { real: f_real, imag: f_imag };
            const orig = z[i][j];
            z_temp3[i][j] = { real: orig.real + dt * f_real, imag: orig.imag + dt * f_imag };
        }
    }
    
    // ---- k4 Step ----
    const coupling_k4 = computeFlexibleCoupling(z_temp3, [coupling1, coupling2, coupling3]);
    const k4 = createGridArray();
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            let localLambda = lambda;
            let localInput = 0;
            if (isAttackActive &&
                i >= N/2 - N/10 && i < N/2 + N/10 &&
                j >= N/2 - N/10 && j < N/2 + N/10) {
                localLambda = lambda * 5;
                localInput = 1.0;
                if (Math.random() < 0.1) {
                    localInput += Math.random() * 0.5;
                }
            }
            const current = z_temp3[i][j];
            const ns = current.real * current.real + current.imag * current.imag;
            const f_real = (localLambda - ns) * current.real - omega * current.imag + coupling_k4[i][j].real + localInput;
            const f_imag = (localLambda - ns) * current.imag + omega * current.real + coupling_k4[i][j].imag;
            k4[i][j] = { real: f_real, imag: f_imag };
        }
    }
    
    // ---- Final RK4 Update ----
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            z[i][j].real += dt / 6 * (k1[i][j].real + 2 * k2[i][j].real + 2 * k3[i][j].real + k4[i][j].real);
            z[i][j].imag += dt / 6 * (k1[i][j].imag + 2 * k2[i][j].imag + 2 * k3[i][j].imag + k4[i][j].imag);
        }
    }
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const amplitude = Math.sqrt(
                z[i][j].real * z[i][j].real + 
                z[i][j].imag * z[i][j].imag
            );
            const phase = Math.atan2(z[i][j].imag, z[i][j].real);
            
            let fillStyle;
            switch(vizMode) {
                case 'both':
                    // Original visualization
                    const hue = ((phase / (2 * Math.PI)) + 0.5) * 360;
                    const saturation = 80;
                    const lightness = Math.min(amplitude * 50, 80);
                    fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                    break;
                    
                case 'phase':
                    // Phase only - full saturation and lightness
                    const phaseHue = ((phase / (2 * Math.PI)) + 0.5) * 360;
                    fillStyle = `hsl(${phaseHue}, 100%, 50%)`;
                    break;
                    
                case 'amplitude':
                    // Conservative range that accounts for attacks
                    const minAmp = 0;  // Include zero for complete suppression
                    const maxAmp = 1.0;  // Cover attack amplitudes plus some headroom
                    
                    const normalizedAmp = Math.min(amplitude / maxAmp, 1.0);
                    
                    if (amplitude < 0.01) {
                        fillStyle = 'rgb(50, 50, 50)'; // Very low = grey
                    } else {
                        const r = Math.floor(255 * Math.pow(normalizedAmp, 0.7));
                        const g = Math.floor(255 * 0.3 * (1 - normalizedAmp));
                        const b = Math.floor(255 * (normalizedAmp < 0.5 ? 
                            0.5 + normalizedAmp : 
                            1 - normalizedAmp));
                        fillStyle = `rgb(${r}, ${g}, ${b})`;
                    }
                    break;
            }
            
            ctx.fillStyle = fillStyle;
            ctx.fillRect(j * cellSize.x, i * cellSize.y, cellSize.x + 1, cellSize.y + 1); // +1 to avoid gaps
        }
    }
}

// Function to compute global dissonance of the current field configuration
function computeDissonance() {
    let totalDissonance = 0;
    let count = 0;
    // Loop over all grid cells
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            // Compute amplitude and phase of current oscillator
            const cur = z[i][j];
            const A_cur = Math.sqrt(cur.real * cur.real + cur.imag * cur.imag);
            const theta_cur = Math.atan2(cur.imag, cur.real);

            // Define neighbor offsets: up, down, left, right
            const neighbors = [
                { di: -1, dj: 0 },
                { di: 1, dj: 0 },
                { di: 0, dj: -1 },
                { di: 0, dj: 1 }
            ];

            let localDissonance = 0;
            let nCount = 0;
            // Loop over neighbors with periodic boundaries
            neighbors.forEach(offset => {
                const ni = (i + offset.di + N) % N;
                const nj = (j + offset.dj + N) % N;
                const neighbor = z[ni][nj];
                const A_neighbor = Math.sqrt(neighbor.real * neighbor.real + neighbor.imag * neighbor.imag);
                const theta_neighbor = Math.atan2(neighbor.imag, neighbor.real);
                // Dissonance measure for the pair: 1 - cos(phase difference)
                const phaseDiff = theta_cur - theta_neighbor;
                const pairDissonance = A_cur * A_neighbor * (1 - Math.cos(phaseDiff));
                localDissonance += pairDissonance;
                nCount++;
            });

            // Average dissonance for this oscillator
            if (nCount > 0) {
                totalDissonance += localDissonance / nCount;
                count++;
            }
        }
    }
    return count > 0 ? totalDissonance / count : 0;
}

// Add DN parameters
let a = 1.0;  // activation amplitude
let b = 0.0; // activation constant
let c = 1.0;  // normalization amplitude
let d = 40.0; // normalization constant

// Compute coupling with flexible kernels and DN terms
function computeFlexibleCouplingDN(z, couplingStrengths) {
    const result = Array(N).fill().map(() => 
        Array(N).fill().map(() => ({ real: 0, imag: 0 }))
    );
    
    // First compute amplitude field
    const amplitudes = Array(N).fill().map((_, i) => 
        Array(N).fill().map((_, j) => 
            Math.sqrt(z[i][j].real * z[i][j].real + z[i][j].imag * z[i][j].imag)
        )
    );
    
    // Compute activation (G₁) and normalization (G₂) terms using your discrete kernels
    const activationField = Array(N).fill().map(() => Array(N).fill(0));
    const normalizationField = Array(N).fill().map(() => Array(N).fill(0));
    
    // Apply kernels with different coupling strengths for activation (G₁)
    Object.entries(kernels).forEach(([distance, kernel], idx) => {
        const K = couplingStrengths[idx];
        const size = kernel.length;
        const offset = Math.floor(size / 2);
        
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                for (let ki = 0; ki < size; ki++) {
                    for (let kj = 0; kj < size; kj++) {
                        const ni = (i + ki - offset + N) % N;
                        const nj = (j + kj - offset + N) % N;
                        
                        if (kernel[ki][kj] !== 0) {
                            activationField[i][j] += K * kernel[ki][kj] * amplitudes[ni][nj];
                            // Use a different spread for normalization field (G₂)
                            normalizationField[i][j] += (K * 1.5) * kernel[ki][kj] * amplitudes[ni][nj];
                        }
                    }
                }
            }
        }
    });
    
    // Compute final coupling term including DN and small world
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            // Compute DN ratio
            const numerator = a * activationField[i][j] + b;
            const denominator = 1 // c * normalizationField[i][j] + d;
            const dnFactor = numerator / denominator;
            
            // Preserve phase from original oscillator
            const phase = Math.atan2(z[i][j].imag, z[i][j].real);
            result[i][j].real = dnFactor * Math.cos(phase);
            result[i][j].imag = dnFactor * Math.sin(phase);
            
            // Add small world coupling
            for (const neighbor of smallWorldNeighbors[i][j]) {
                const ni = neighbor.row;
                const nj = neighbor.col;
                result[i][j].real += smallWorldCoupling * (z[ni][nj].real - z[i][j].real);
                result[i][j].imag += smallWorldCoupling * (z[ni][nj].imag - z[i][j].imag);
            }
        }
    }
    
    return result;
}

function computeFlexibleCoupling(z, couplingStrengths) {
    // Initialize result and weight arrays for each cell
    const result = Array.from({ length: N }, () =>
        Array.from({ length: N }, () => ({ real: 0, imag: 0 }))
    );
    const totalWeight = Array.from({ length: N }, () =>
        Array.from({ length: N }, () => 0)
    );

    // Sum contributions from all kernels (without per-kernel normalization)
    Object.entries(kernels).forEach(([distance, kernel], idx) => {
        const K = couplingStrengths[idx];
        const size = kernel.length;
        const offset = Math.floor(size / 2);

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                for (let ki = 0; ki < size; ki++) {
                    for (let kj = 0; kj < size; kj++) {
                        const ni = (i + ki - offset + N) % N;
                        const nj = (j + kj - offset + N) % N;
                        if (kernel[ki][kj] !== 0) {
                            result[i][j].real += K * kernel[ki][kj] * z[ni][nj].real;
                            result[i][j].imag += K * kernel[ki][kj] * z[ni][nj].imag;
                            // Accumulate the absolute weight (you can choose a different scheme if desired)
                            totalWeight[i][j] += Math.abs(K * kernel[ki][kj]);
                        }
                    }
                }
            }
        }
    });

    // Global normalization: divide the summed contribution by the total weight
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            if (totalWeight[i][j] > 0) {
                result[i][j].real /= totalWeight[i][j];
                result[i][j].imag /= totalWeight[i][j];
            }
        }
    }

    // Add the small world coupling as before
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const numSmallWorld = smallWorldNeighbors[i][j].length;
            if (numSmallWorld > 0) {
                let swReal = 0, swImag = 0;
                for (const neighbor of smallWorldNeighbors[i][j]) {
                    const ni = neighbor.row;
                    const nj = neighbor.col;
                    swReal += (z[ni][nj].real - z[i][j].real);
                    swImag += (z[ni][nj].imag - z[i][j].imag);
                }
                result[i][j].real += smallWorldCoupling * (swReal / numSmallWorld);
                result[i][j].imag += smallWorldCoupling * (swImag / numSmallWorld);
            }
        }
    }

    return result;
}

// In your animation loop, update the dissonance display.
function animate() {
    update();
    draw();

    // Compute dissonance measure
    const dissonance = computeDissonance();
    // Update the HTML element with id "dissonanceValue"
    const dissonanceDisplay = document.getElementById('dissonanceValue');
    if (dissonanceDisplay) {
        dissonanceDisplay.textContent = dissonance.toFixed(2);
    }

    requestAnimationFrame(animate);
}

// Setup UI controls
document.getElementById('lambda').addEventListener('input', (e) => {
    lambda = parseFloat(e.target.value);
    document.getElementById('lambdaValue').textContent = lambda.toFixed(2);
});

document.getElementById('omega').addEventListener('input', (e) => {
    omega = parseFloat(e.target.value);
    document.getElementById('omegaValue').textContent = omega.toFixed(2);
});

document.getElementById('coupling1').addEventListener('input', (e) => {
    coupling1 = parseFloat(e.target.value);
    document.getElementById('coupling1Value').textContent = coupling1.toFixed(2);
});

document.getElementById('coupling2').addEventListener('input', (e) => {
    coupling2 = parseFloat(e.target.value);
    document.getElementById('coupling2Value').textContent = coupling2.toFixed(2);
});

document.getElementById('coupling3').addEventListener('input', (e) => {
    coupling3 = parseFloat(e.target.value);
    document.getElementById('coupling3Value').textContent = coupling3.toFixed(2);
});

document.getElementById('smallWorld').addEventListener('input', (e) => {
    smallWorldCoupling = parseFloat(e.target.value);
    document.getElementById('smallWorldValue').textContent = smallWorldCoupling.toFixed(2);
});

// Setup visualization mode control
document.getElementById('vizMode').addEventListener('change', (e) => {
    vizMode = e.target.value;
});

// DN Parameter Controls
/* document.getElementById('activationAmp').addEventListener('input', (e) => {
    a = parseFloat(e.target.value);
    document.getElementById('activationAmpValue').textContent = a.toFixed(1);
});

document.getElementById('activationConst').addEventListener('input', (e) => {
    b = parseFloat(e.target.value);
    document.getElementById('activationConstValue').textContent = b.toFixed(1);
});

document.getElementById('normAmp').addEventListener('input', (e) => {
    c = parseFloat(e.target.value);
    document.getElementById('normAmpValue').textContent = c.toFixed(1);
});

document.getElementById('normConst').addEventListener('input', (e) => {
    d = parseFloat(e.target.value);
    document.getElementById('normConstValue').textContent = d.toFixed(1);
}); */

document.getElementById('triggerAttack').addEventListener('click', () => {
    isAttackActive = !isAttackActive;
    const button = document.getElementById('triggerAttack');
    button.textContent = isAttackActive ? 'Stop Attack' : 'Trigger Attack';
    button.style.backgroundColor = isAttackActive ? '#f44336' : '#4CAF50';
    
    if (isAttackActive) {
        for (let i = N/2 - N/10; i < N/2 + N/10; i++) {
            for (let j = N/2 - N/10; j < N/2 + N/10; j++) {
                z[i][j].real += 0.5 * (Math.random() - 0.5);
                z[i][j].imag += 0.5 * (Math.random() - 0.5);
            }
        }
    }
});

// Handle window resizing
window.addEventListener('resize', resizeCanvas);

// Initial setup
resizeCanvas();
animate();