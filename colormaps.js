// colormaps.js - Collection of color mapping functions for visualization

// Function to map a value in [0,1] to a color in various color schemes
function getColorFromMap(value, colormap = 'viridis') {
    // Ensure value is in [0,1]
    value = Math.max(0, Math.min(1, value));
    
    // Different color maps
    switch (colormap) {
        case 'viridis':
            return viridisColormap(value);
        case 'plasma':
            return plasmaColormap(value);
        case 'magma':
            return magmaColormap(value);
        case 'inferno':
            return infernoColormap(value);
        case 'turbo':
            return turboColormap(value);
        case 'coolwarm':
            return coolwarmColormap(value);
        case 'heatmap':
            return heatmapColormap(value);
        default:
            return viridisColormap(value);
    }
}

// Viridis colormap - good for general purpose visualization
function viridisColormap(t) {
    let r, g, b;
    
    if (t < 0.25) {
        // Dark purple to blue
        r = Math.floor(68 + t * 4 * (65 - 68));
        g = Math.floor(1 + t * 4 * (104 - 1));
        b = Math.floor(84 + t * 4 * (171 - 84));
    } else if (t < 0.5) {
        // Blue to teal
        const s = (t - 0.25) * 4;
        r = Math.floor(65 + s * (33 - 65));
        g = Math.floor(104 + s * (145 - 104));
        b = Math.floor(171 + s * (140 - 171));
    } else if (t < 0.75) {
        // Teal to green
        const s = (t - 0.5) * 4;
        r = Math.floor(33 + s * (94 - 33));
        g = Math.floor(145 + s * (170 - 145));
        b = Math.floor(140 + s * (65 - 140));
    } else {
        // Green to yellow
        const s = (t - 0.75) * 4;
        r = Math.floor(94 + s * (253 - 94));
        g = Math.floor(170 + s * (231 - 170));
        b = Math.floor(65 + s * (37 - 65));
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Plasma colormap - more vibrant, highlights medium values
function plasmaColormap(t) {
    let r, g, b;
    
    if (t < 0.25) {
        // Deep purple to purple
        r = Math.floor(13 + t * 4 * (84 - 13));
        g = Math.floor(8 + t * 4 * (39 - 8));
        b = Math.floor(135 + t * 4 * (159 - 135));
    } else if (t < 0.5) {
        // Purple to magenta
        const s = (t - 0.25) * 4;
        r = Math.floor(84 + s * (156 - 84));
        g = Math.floor(39 + s * (46 - 39));
        b = Math.floor(159 + s * (153 - 159));
    } else if (t < 0.75) {
        // Magenta to orange
        const s = (t - 0.5) * 4;
        r = Math.floor(156 + s * (236 - 156));
        g = Math.floor(46 + s * (112 - 46));
        b = Math.floor(153 + s * (50 - 153));
    } else {
        // Orange to yellow
        const s = (t - 0.75) * 4;
        r = Math.floor(236 + s * (253 - 236));
        g = Math.floor(112 + s * (231 - 112));
        b = Math.floor(50 + s * (37 - 50));
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Coolwarm colormap - good for diverging data (below/above average)
function coolwarmColormap(t) {
    // This maps values below 0.5 to cool colors (blues)
    // and values above 0.5 to warm colors (reds)
    let r, g, b;
    
    if (t < 0.5) {
        // Blue to white (0 to 0.5 mapped to darker-to-lighter blues)
        const s = t * 2; // Scale to [0,1]
        r = Math.floor(59 + s * (220 - 59));
        g = Math.floor(76 + s * (220 - 76));
        b = Math.floor(192 + s * (220 - 192));
    } else {
        // White to red (0.5 to 1 mapped to lighter-to-darker reds)
        const s = (t - 0.5) * 2; // Scale to [0,1]
        r = Math.floor(220 - s * (220 - 180));
        g = Math.floor(220 - s * (220 - 4));
        b = Math.floor(220 - s * (220 - 38));
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Magma colormap - dramatic, good for showing discontinuities
function magmaColormap(t) {
    let r, g, b;
    
    if (t < 0.25) {
        // Black to purple
        r = Math.floor(0 + t * 4 * (88 - 0));
        g = Math.floor(0 + t * 4 * (24 - 0));
        b = Math.floor(4 + t * 4 * (108 - 4));
    } else if (t < 0.5) {
        // Purple to magenta
        const s = (t - 0.25) * 4;
        r = Math.floor(88 + s * (196 - 88));
        g = Math.floor(24 + s * (30 - 24));
        b = Math.floor(108 + s * (114 - 108));
    } else if (t < 0.75) {
        // Magenta to orange
        const s = (t - 0.5) * 4;
        r = Math.floor(196 + s * (249 - 196));
        g = Math.floor(30 + s * (99 - 30));
        b = Math.floor(114 + s * (99 - 114));
    } else {
        // Orange to yellow
        const s = (t - 0.75) * 4;
        r = Math.floor(249 + s * (252 - 249));
        g = Math.floor(99 + s * (253 - 99));
        b = Math.floor(99 + s * (191 - 99));
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Inferno colormap - dramatic black to yellow
function infernoColormap(t) {
    let r, g, b;
    
    if (t < 0.25) {
        // Black to purple
        r = Math.floor(0 + t * 4 * (73 - 0));
        g = Math.floor(0 + t * 4 * (13 - 0));
        b = Math.floor(4 + t * 4 * (89 - 4));
    } else if (t < 0.5) {
        // Purple to red
        const s = (t - 0.25) * 4;
        r = Math.floor(73 + s * (183 - 73));
        g = Math.floor(13 + s * (55 - 13));
        b = Math.floor(89 + s * (82 - 89));
    } else if (t < 0.75) {
        // Red to orange
        const s = (t - 0.5) * 4;
        r = Math.floor(183 + s * (252 - 183));
        g = Math.floor(55 + s * (137 - 55));
        b = Math.floor(82 + s * (37 - 82));
    } else {
        // Orange to yellow
        const s = (t - 0.75) * 4;
        r = Math.floor(252 + s * (252 - 252));
        g = Math.floor(137 + s * (254 - 137));
        b = Math.floor(37 + s * (164 - 37));
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Turbo colormap - high perceptual variation, good for fine details
function turboColormap(t) {
    const r = Math.sin(t * Math.PI) * 0.5 + 0.5;
    const g = Math.sin(t * Math.PI + 2 * Math.PI/3) * 0.5 + 0.5;
    const b = Math.sin(t * Math.PI + 4 * Math.PI/3) * 0.5 + 0.5;
    
    return `rgb(${Math.floor(r*255)}, ${Math.floor(g*255)}, ${Math.floor(b*255)})`;
}

// Simple heatmap (black -> red -> yellow -> white)
function heatmapColormap(t) {
    let r, g, b;
    
    if (t < 0.33) {
        // Black to red
        r = Math.floor(t * 3 * 255);
        g = 0;
        b = 0;
    } else if (t < 0.66) {
        // Red to yellow
        r = 255;
        g = Math.floor((t - 0.33) * 3 * 255);
        b = 0;
    } else {
        // Yellow to white
        r = 255;
        g = 255;
        b = Math.floor((t - 0.66) * 3 * 255);
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Log-scaled mapping for better visualization of data with large dynamic range
function logScaleNormalize(value, min, max) {
    if (min >= max) return 0.5;
    
    // Apply log scaling if the range is large enough
    if (max / Math.max(0.0001, min) > 10) {
        // Use log10(1+x) to handle values close to zero
        const logMin = Math.log10(1 + min);
        const logMax = Math.log10(1 + max);
        const logValue = Math.log10(1 + value);
        
        return (logValue - logMin) / (logMax - logMin);
    } else {
        // Linear scaling for smaller ranges
        return (value - min) / (max - min);
    }
}

// Contrast enhancement for better visualization
function enhanceContrast(normalizedValue, amount = 2.0) {
    // Center-weighted contrast enhancement
    // amount: 1.0 = no change, >1.0 = more contrast
    const centered = normalizedValue - 0.5;
    const enhanced = Math.sign(centered) * Math.pow(Math.abs(centered), 1/amount) + 0.5;
    return Math.max(0, Math.min(1, enhanced));
}