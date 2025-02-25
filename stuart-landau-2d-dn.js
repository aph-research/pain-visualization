const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const FIXED_MAX_AMP = 3.5;

// Grid parameters
let N = 100;
let cellSize;
const dt = 0.05;

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
let lambda = 0.01;  // Base excitability
let vizMode = 'both'; // Visualization mode
let omega = 1.0;  // Natural frequency
let coupling1 = 0.2;
let coupling2 = 0.0;
let coupling3 = 0.0;
let smallWorldCoupling = 0.0;
let isAttackActive = false;
let persistentLambdaMap = null; // Will store persistent lambda values 
let persistenceRate = 0.999;     // Controls how quickly persistent effects decay

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

function initializePersistenceMap() {
    persistentLambdaMap = Array(N).fill().map(() => Array(N).fill(0));
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

function getLocalParams(i, j) {
    // Default values
    let localLambda = lambda;
    let localInput = 0;
    
    // Check if this cell is in the attack region
    if (isAttackActive &&
        i >= N/2 - N/10 && i < N/2 + N/10 &&
        j >= N/2 - N/10 && j < N/2 + N/10) {
        
        // Active attack - store the heightened lambda value for persistence
        const attackLambda = lambda * 10;
        persistentLambdaMap[i][j] = Math.max(persistentLambdaMap[i][j], attackLambda - lambda);
        
        localLambda = attackLambda;
        localInput = 3.0;
        
        if (Math.random() < 0.1) {
            localInput += Math.random() * 0.5;
        }
    } else {
        // Not in active attack, but apply any persistent effects
        localLambda = lambda + persistentLambdaMap[i][j];
    }
    
    return { localLambda, localInput };
}

// This is the derivative function for the Stuart-Landau oscillator
function computeDerivative(z_real, z_imag, localLambda, localOmega, couplingTerm_real, couplingTerm_imag, localInput) {
    const ns = z_real * z_real + z_imag * z_imag;
    return {
        real: (localLambda - ns) * z_real - localOmega * z_imag + couplingTerm_real + localInput,
        imag: (localLambda - ns) * z_imag + localOmega * z_real + couplingTerm_imag
    };
}

function updatePersistence() {
    // Gradually decay the persistent effects
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            if (persistentLambdaMap[i][j] > 0.01) {
                persistentLambdaMap[i][j] *= persistenceRate;
            } else {
                persistentLambdaMap[i][j] = 0;
            }
        }
    }
}

function update() {
    // We'll perform a full-grid RK4 update where we recompute the coupling field at each sub-step.
    
    // ---- k1 Step ----
    const coupling_k1 = computeFlexibleCoupling(z, [coupling1, coupling2, coupling3]);
    const k1 = createGridArray();
    const z_temp1 = createGridArray(); // will hold z + (dt/2)*k1
    
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const { localLambda, localInput } = getLocalParams(i, j);
            const current = z[i][j];
            
            const derivative = computeDerivative(
                current.real, current.imag, 
                localLambda, omega, 
                coupling_k1[i][j].real, coupling_k1[i][j].imag, 
                localInput
            );
            
            k1[i][j] = { real: derivative.real, imag: derivative.imag };
            z_temp1[i][j] = { 
                real: current.real + (dt/2) * derivative.real, 
                imag: current.imag + (dt/2) * derivative.imag 
            };
        }
    }
    
    // ---- k2 Step ----
    const coupling_k2 = computeFlexibleCoupling(z_temp1, [coupling1, coupling2, coupling3]);
    const k2 = createGridArray();
    const z_temp2 = createGridArray(); // will hold z + (dt/2)*k2 (using original z)
    
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const { localLambda, localInput } = getLocalParams(i, j);
            const current = z_temp1[i][j];
            
            const derivative = computeDerivative(
                current.real, current.imag, 
                localLambda, omega, 
                coupling_k2[i][j].real, coupling_k2[i][j].imag, 
                localInput
            );
            
            k2[i][j] = { real: derivative.real, imag: derivative.imag };
            // For k2, we use the original z for the update: z_temp2 = z + (dt/2)*k2
            const orig = z[i][j];
            z_temp2[i][j] = { 
                real: orig.real + (dt/2) * derivative.real, 
                imag: orig.imag + (dt/2) * derivative.imag 
            };
        }
    }
    
    // ---- k3 Step ----
    const coupling_k3 = computeFlexibleCoupling(z_temp2, [coupling1, coupling2, coupling3]);
    const k3 = createGridArray();
    const z_temp3 = createGridArray(); // will hold z + dt*k3 (using original z)
    
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const { localLambda, localInput } = getLocalParams(i, j);
            const current = z_temp2[i][j];
            
            const derivative = computeDerivative(
                current.real, current.imag, 
                localLambda, omega, 
                coupling_k3[i][j].real, coupling_k3[i][j].imag, 
                localInput
            );
            
            k3[i][j] = { real: derivative.real, imag: derivative.imag };
            const orig = z[i][j];
            z_temp3[i][j] = { 
                real: orig.real + dt * derivative.real, 
                imag: orig.imag + dt * derivative.imag 
            };
        }
    }
    
    // ---- k4 Step ----
    const coupling_k4 = computeFlexibleCoupling(z_temp3, [coupling1, coupling2, coupling3]);
    const k4 = createGridArray();
    
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const { localLambda, localInput } = getLocalParams(i, j);
            const current = z_temp3[i][j];
            
            const derivative = computeDerivative(
                current.real, current.imag, 
                localLambda, omega, 
                coupling_k4[i][j].real, coupling_k4[i][j].imag, 
                localInput
            );
            
            k4[i][j] = { real: derivative.real, imag: derivative.imag };
        }
    }
    
    // ---- Final RK4 Update ----
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            z[i][j].real += dt / 6 * (k1[i][j].real + 2 * k2[i][j].real + 2 * k3[i][j].real + k4[i][j].real);
            z[i][j].imag += dt / 6 * (k1[i][j].imag + 2 * k2[i][j].imag + 2 * k3[i][j].imag + k4[i][j].imag);
        }
    }
    updatePersistence();
}

let selectedColormap = 'viridis';

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Still calculate statistics for display purposes
    let minAmp = Infinity;
    let maxAmp = -Infinity;
    let ampSum = 0;
    let ampCount = 0;
    
    // First pass to calculate statistics only (for the histogram and stats display)
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const amplitude = Math.sqrt(
                z[i][j].real * z[i][j].real + 
                z[i][j].imag * z[i][j].imag
            );
            
            if (amplitude > 0.01) { // Filter out very small values
                minAmp = Math.min(minAmp, amplitude);
                maxAmp = Math.max(maxAmp, amplitude);
                ampSum += amplitude;
                ampCount++;
            }
        }
    }
    
    // Safety check
    if (minAmp === Infinity) minAmp = 0;
    if (maxAmp === -Infinity) maxAmp = 1;
    
    // Calculate average
    const avgAmp = ampCount > 0 ? ampSum / ampCount : 0;
    
    // Visualization pass
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
                    // Combined phase & amplitude
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
                    if (amplitude < 0.01) {
                        fillStyle = 'rgb(50, 50, 50)'; // Very low = grey
                    } else {
                        // Use fixed maximum instead of calculated maxAmp
                        let normalizedAmp = amplitude / FIXED_MAX_AMP;
                        normalizedAmp = Math.max(0, Math.min(1, normalizedAmp)); // Clamp to [0,1]
                        
                        // Use the selected colormap
                        fillStyle = getColorFromMap(normalizedAmp, selectedColormap);
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
    
    // Update amplitude histogram every few frames for performance
    if (window.frameCount === undefined) window.frameCount = 0;
    window.frameCount++;
    
    if (window.frameCount % 10 === 0) { // Update every 10 frames
        updateAmplitudeHistogram();
    }
    
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

document.getElementById('colormap').addEventListener('change', (e) => {
    selectedColormap = e.target.value;
});

document.getElementById('triggerAttack').addEventListener('click', () => {
    isAttackActive = !isAttackActive;
    const button = document.getElementById('triggerAttack');
    button.textContent = isAttackActive ? 'Stop Attack' : 'Trigger Attack';
    button.style.backgroundColor = isAttackActive ? '#f44336' : '#4CAF50';
});

document.addEventListener('DOMContentLoaded', function() {
    setupAmplitudeMonitor();
});

function setupAmplitudeMonitor() {
    const histogramContainer = document.querySelector('.histogram-container');
    if (!histogramContainer) return; // Exit if container not found
    
    // Clear any existing bars
    histogramContainer.innerHTML = '';
    
    // Create histogram bars
    const NUM_BINS = 20;
    window.histogramBars = []; // Store reference globally
    
    for (let i = 0; i < NUM_BINS; i++) {
        const bar = document.createElement('div');
        bar.style.position = 'absolute';
        bar.style.bottom = '0';
        bar.style.width = `${100/NUM_BINS}%`;
        bar.style.left = `${i * (100/NUM_BINS)}%`;
        bar.style.height = '0px';
        bar.style.backgroundColor = getHistogramColor(i/NUM_BINS);
        bar.style.transition = 'height 0.3s ease';
        histogramContainer.appendChild(bar);
        window.histogramBars.push(bar);
    }
}

// Get color for histogram bars (simplified colormap)
function getHistogramColor(t) {
    return getColorFromMap(t, 'viridis');
}

// Update the histogram with current amplitude data
function updateAmplitudeHistogram() {
    if (!window.histogramBars || window.histogramBars.length === 0) return;
    
    // Collect all amplitude values
    const amplitudes = [];
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const amplitude = Math.sqrt(
                z[i][j].real * z[i][j].real + 
                z[i][j].imag * z[i][j].imag
            );
            if (amplitude > 0.001) { // Filter out near-zero values
                amplitudes.push(amplitude);
            }
        }
    }
    
    if (amplitudes.length === 0) return;
    
    // Calculate statistics
    let min = Math.min(...amplitudes);
    let max = Math.max(...amplitudes);
    let avg = amplitudes.reduce((sum, val) => sum + val, 0) / amplitudes.length;
    
    // Update stats display
    document.getElementById('minAmp').textContent = min.toFixed(3);
    document.getElementById('avgAmp').textContent = avg.toFixed(3);
    document.getElementById('maxAmp').textContent = max.toFixed(3);
    
    // Create histogram
    const NUM_BINS = window.histogramBars.length;
    const bins = Array(NUM_BINS).fill(0);
    const binSize = (max - min) / NUM_BINS;
    
    if (binSize === 0) return; // Avoid division by zero
    
    // Count values in each bin
    amplitudes.forEach(amp => {
        const binIndex = Math.min(NUM_BINS - 1, Math.floor((amp - min) / binSize));
        bins[binIndex]++;
    });
    
    // Find the maximum bin count for normalization
    const maxBinCount = Math.max(...bins);
    
    // Update histogram bars
    for (let i = 0; i < NUM_BINS; i++) {
        const heightPercentage = maxBinCount > 0 ? (bins[i] / maxBinCount) * 100 : 0;
        window.histogramBars[i].style.height = `${heightPercentage}%`;
    }
    
    // Also log to console for debugging
    console.log(`Amplitude stats - Min: ${min.toFixed(3)}, Max: ${max.toFixed(3)}, Avg: ${avg.toFixed(3)}`);
}

// Handle window resizing
window.addEventListener('resize', resizeCanvas);

// Initial setup
initializePersistenceMap();
resizeCanvas();
animate();