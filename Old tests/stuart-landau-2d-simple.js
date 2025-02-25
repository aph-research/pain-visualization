const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Grid parameters
const N = 50;  // Grid size
const dt = 0.05;  // Time step

// Initialize canvas size
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    cellSize = Math.min(canvas.width, canvas.height) / N;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Control parameters
let lambda = 0.1;  // Base excitability
let K = 0.2;      // Coupling strength
let omega = 0.6;  // Natural frequency
let isAttackActive = false;

// Initialize the complex field z
let z = Array(N).fill().map(() => 
    Array(N).fill().map(() => ({
        real: 0.1 * (Math.random() - 0.5),
        imag: 0.1 * (Math.random() - 0.5)
    }))
);

// Laplacian kernel for coupling
const kernel = [
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
];

// Compute Laplacian with periodic boundary conditions
function computeLaplacian(z) {
    const result = Array(N).fill().map(() => 
        Array(N).fill().map(() => ({ real: 0, imag: 0 }))
    );
    
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            for (let ki = -1; ki <= 1; ki++) {
                for (let kj = -1; kj <= 1; kj++) {
                    const ni = (i + ki + N) % N;
                    const nj = (j + kj + N) % N;
                    const k = kernel[ki + 1][kj + 1];
                    
                    result[i][j].real += k * z[ni][nj].real;
                    result[i][j].imag += k * z[ni][nj].imag;
                }
            }
        }
    }
    return result;
}

// Update function for Stuart-Landau dynamics
function update() {
    const laplacian = computeLaplacian(z);
    
    // Update each oscillator
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            // Get current state
            const current = z[i][j];
            const lap = laplacian[i][j];
            
            // Compute |z|^2
            const normSq = current.real * current.real + current.imag * current.imag;
            
            // Local parameters
            let localLambda = lambda;
            let localInput = 0;
            
            // Modify parameters in attack region if active
            if (isAttackActive && 
                i >= N/2 - 5 && i < N/2 + 5 && 
                j >= N/2 - 5 && j < N/2 + 5) {
                localLambda = lambda * 5;  // Increased excitability (amplified)
                localInput = 1.0;          // Stronger external input
                
                // Add some noise to make the effect more visible
                if (Math.random() < 0.1) {
                    localInput += Math.random() * 0.5;
                }
            }
            
            // Stuart-Landau dynamics
            const dreal = (localLambda - normSq) * current.real - 
                        omega * current.imag + 
                        K * lap.real + localInput;
            
            const dimag = (localLambda - normSq) * current.imag + 
                        omega * current.real + 
                        K * lap.imag;
            
            // Euler integration
            z[i][j].real += dt * dreal;
            z[i][j].imag += dt * dimag;
        }
    }
}

// Visualization function
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const amplitude = Math.sqrt(
                z[i][j].real * z[i][j].real + 
                z[i][j].imag * z[i][j].imag
            );
            const phase = Math.atan2(z[i][j].imag, z[i][j].real);
            
            // Map amplitude to brightness and phase to hue
            const hue = ((phase / (2 * Math.PI)) + 0.5) * 360;
            const saturation = 80;
            const lightness = Math.min(amplitude * 50, 80);
            
            ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
    }
}

// Animation loop
function animate() {
    update();
    draw();
    requestAnimationFrame(animate);
}

// Setup UI controls
document.getElementById('lambda').addEventListener('input', (e) => {
    lambda = parseFloat(e.target.value);
    document.getElementById('lambdaValue').textContent = lambda.toFixed(2);
});

document.getElementById('coupling').addEventListener('input', (e) => {
    K = parseFloat(e.target.value);
    document.getElementById('couplingValue').textContent = K.toFixed(2);
});

document.getElementById('omega').addEventListener('input', (e) => {
    omega = parseFloat(e.target.value);
    document.getElementById('omegaValue').textContent = omega.toFixed(2);
});

document.getElementById('triggerAttack').addEventListener('click', () => {
    isAttackActive = !isAttackActive;
    const button = document.getElementById('triggerAttack');
    button.textContent = isAttackActive ? 'Stop Attack' : 'Trigger Attack';
    button.style.backgroundColor = isAttackActive ? '#f44336' : '#4CAF50';
    console.log('Attack state:', isAttackActive);  // Debug logging
    
    // Reset central region to create more dramatic effect when attack starts
    if (isAttackActive) {
        for (let i = N/2 - 5; i < N/2 + 5; i++) {
            for (let j = N/2 - 5; j < N/2 + 5; j++) {
                z[i][j].real += 0.5 * (Math.random() - 0.5);
                z[i][j].imag += 0.5 * (Math.random() - 0.5);
            }
        }
    }
});

// Debug initialization
console.log('Script loaded');
console.log('Canvas dimensions:', canvas.width, canvas.height);
console.log('Cell size:', cellSize);

// Initialize with some non-black values
for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
        z[i][j].real = 0.3 * (Math.random() - 0.5);
        z[i][j].imag = 0.3 * (Math.random() - 0.5);
    }
}

// Start animation
animate();