const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Grid parameters
let N = 100;  // Reduced grid size for performance
let cellSize;
const dt = 0.5;  // Larger time step for performance

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
let omega = 0.6;  // Natural frequency
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

// Compute coupling with flexible kernels and small world
function computeFlexibleCoupling(z, couplingStrengths) {
    const result = Array(N).fill().map(() => 
        Array(N).fill().map(() => ({ real: 0, imag: 0 }))
    );
    
    // Apply each kernel with its coupling strength
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
                        }
                    }
                }
                
                // Add small world coupling
                for (const neighbor of smallWorldNeighbors[i][j]) {
                    const ni = neighbor.row;
                    const nj = neighbor.col;
                    result[i][j].real += smallWorldCoupling * (z[ni][nj].real - z[i][j].real);
                    result[i][j].imag += smallWorldCoupling * (z[ni][nj].imag - z[i][j].imag);
                }
            }
        }
    });
    
    return result;
}

function update() {
    const couplingStrengths = [coupling1, coupling2, coupling3];
    const coupling = computeFlexibleCoupling(z, couplingStrengths);
    
    // Compute the local normalization field from the current amplitudes
    const localNorm = computeLocalNormalization();
    
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const current = z[i][j];
            const normSq = current.real * current.real + current.imag * current.imag;
            
            // Local parameters for attack simulation (same as before)
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
            
            // Create derivative function for RK4
            const derivs = (x, y) => {
                const ns = x * x + y * y;
                return {
                    real: (localLambda - ns) * x - omega * y + coupling[i][j].real + localInput,
                    imag: (localLambda - ns) * y + omega * x + coupling[i][j].imag
                };
            };

            // RK4 integration (same as before)
            const delta = rk4Step(current.real, current.imag, dt, derivs);
            
            // Compute normalization factor (prevent runaway by scaling derivative)
            // Here, normFactor = 1 + beta * localNorm[i][j]
            const normFactor = 1 + beta * localNorm[i][j];
            
            // Update state with divisive normalization applied:
            z[i][j].real += dt * delta.real / normFactor;
            z[i][j].imag += dt * delta.imag / normFactor;
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
                    // Much more aggressive amplitude scaling for high coupling regimes
                    const logScale = Math.log1p(amplitude * 3) / Math.log1p(3);
                    const normalizedAmp = Math.min(logScale, 1.0);
                    
                    // Enhanced color mapping with better midrange resolution
                    if (amplitude < 0.01) {
                        fillStyle = 'rgb(50, 50, 50)'; // Very low = grey
                    } else {
                        // Three-color gradient: grey -> blue -> red
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

// New DN parameter (divisive normalization strength)
let beta = 1.0;  // Adjust this value to change the normalization strength

// Define a function to create a Gaussian kernel
function gaussianKernel(size, sigma) {
    let kernel = [];
    let sum = 0;
    let half = Math.floor(size / 2);
    for (let i = -half; i <= half; i++) {
        kernel[i + half] = [];
        for (let j = -half; j <= half; j++) {
            let val = Math.exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel[i + half][j + half] = val;
            sum += val;
        }
    }
    // Normalize the kernel so that the sum is 1.
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            kernel[i][j] /= sum;
        }
    }
    return kernel;
}

// Pre-compute a Gaussian kernel for the DN calculation:
const kernelSize = 5;
const sigmaDN = 2;
const gaussKernel = gaussianKernel(kernelSize, sigmaDN);

// Compute local normalization: for each cell, convolve the amplitude field with the Gaussian kernel.
function computeLocalNormalization() {
    const localNorm = [];
    // Create a 2D amplitude array from z
    const amplitude = [];
    for (let i = 0; i < N; i++) {
        amplitude[i] = [];
        for (let j = 0; j < N; j++) {
            const a = Math.sqrt(z[i][j].real * z[i][j].real + z[i][j].imag * z[i][j].imag);
            amplitude[i][j] = a;
        }
    }
    // Convolve amplitude with gaussKernel using periodic boundaries
    const half = Math.floor(kernelSize / 2);
    for (let i = 0; i < N; i++) {
        localNorm[i] = [];
        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let ki = -half; ki <= half; ki++) {
                for (let kj = -half; kj <= half; kj++) {
                    const ni = (i + ki + N) % N;
                    const nj = (j + kj + N) % N;
                    sum += amplitude[ni][nj] * gaussKernel[ki + half][kj + half];
                }
            }
            localNorm[i][j] = sum;
        }
    }
    return localNorm;
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

document.getElementById('beta').addEventListener('input', (e) => {
    beta = parseFloat(e.target.value);
    document.getElementById('betaValue').textContent = beta.toFixed(1);
});

// Setup visualization mode control
document.getElementById('vizMode').addEventListener('change', (e) => {
    vizMode = e.target.value;
});

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