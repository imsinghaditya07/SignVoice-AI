/**
 * SignVoice AI - Frontend Application Core
 * ---------------------------------------
 * Handles real-time camera processing, UI state management,
 * and high-performance communication with the AI inference backend.
 * 
 * Includes:
 * - Dynamic Slide Navigation
 * - Real-time Webcam to Base64 Pipeline
 * - Mediapipe Skeleton Overlay Rendering
 * - Text-to-Sign Animation Sequencer
 */

// ==========================================
// CONFIGURATION (CHANGE THIS WHEN DEPLOYING)
// ==========================================
// If deploying the frontend to Vercel, change this to your deployed Python Backend URL (e.g., your Render API Link)
const API_BASE_URL = 'http://localhost:5002';

// ==========================================
// Slide Navigation Logic
function navigate(slideId) {
    // Prevent default anchor behavior
    if (window.event) window.event.preventDefault();

    // 1. Remove active class from all slides
    const slides = document.querySelectorAll('.slide');
    slides.forEach(slide => {
        slide.classList.remove('active-slide');
    });

    // 2. Remove active class from all nav links
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        link.classList.remove('active');
    });

    // 3. Add active class to targeted slide and nav link
    document.getElementById(slideId).classList.add('active-slide');
    
    // Find the link that corresponds to this slide
    const targetLink = Array.from(navLinks).find(link => link.getAttribute('href') === `#${slideId}`);
    if(targetLink) {
        targetLink.classList.add('active');
    }
}

// Redirect from Home buttons directly to a specific Dashboard mode
function goToDashboard(mode) {
    navigate('dashboard');
    switchDashboardMode(mode);
}

// Dashboard Mode Switching Logic (S2T vs T2S)
function switchDashboardMode(mode) {
    // 1. Update Tabs
    document.getElementById('tab-s2t').classList.remove('active-tab');
    document.getElementById('tab-t2s').classList.remove('active-tab');
    document.getElementById(`tab-${mode}`).classList.add('active-tab');

    // 2. Update Panels
    document.getElementById('panel-s2t').classList.remove('active-panel');
    document.getElementById('panel-t2s').classList.remove('active-panel');
    document.getElementById(`panel-${mode}`).classList.add('active-panel');
}

// Theme (Dark/Light Mode) Toggle Logic
function toggleTheme() {
    const body = document.body;
    const themeIcon = document.getElementById('theme-icon');
    
    body.classList.toggle('dark-mode');
    
    // Switch icon depending on active mode
    if (body.classList.contains('dark-mode')) {
        themeIcon.setAttribute('name', 'sunny-outline');
    } else {
        themeIcon.setAttribute('name', 'moon-outline');
    }
}

// Check system preference on load
window.addEventListener('DOMContentLoaded', () => {
    // Default to light-mode as requested, but we can respect system settings if desired
    // For now, it initializes as light-mode based on the HTML class
    console.log("SignVoiceAI initialized.");
});

// Camera logic & AI Inference Loop
let captureInterval;

async function enableCamera() {
    const videoPlaceholder = document.getElementById('camera-placeholder');
    const videoElement = document.getElementById('video-feed');
    const outputBox = document.querySelector('.output-box p');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 400, height: 400, facingMode: "user" }
        });

        videoElement.srcObject = stream;
        videoElement.style.display = "block";
        videoPlaceholder.style.display = "none";
        
        outputBox.innerHTML = "Camera active. Syncing with AI model... Please make a sign.";
        outputBox.classList.remove('placeholder-text');

        // Create an invisible canvas to extract frames
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 400;
        const ctx = canvas.getContext('2d');

        // Start recursive synchronous inference loop (Replaces broken setInterval)
        let isRunning = true;
        
        async function captureLoop() {
            if (!isRunning) return;
            
            if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                const base64Image = canvas.toDataURL('image/jpeg', 0.6); // slight compression

                try {
                    const response = await fetch(`${API_BASE_URL}/predict`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64Image })
                    });
                    const data = await response.json();
                    
                    if (data.prediction && data.prediction !== '...') {
                        outputBox.innerHTML = `<strong>Letter:</strong> <span style="font-size:3rem; color:var(--accent-blue)">${data.prediction}</span>`;
                    }
                    
                    // Live Skeleton Preview Display
                    const skeletonFeed = document.getElementById('skeleton-feed');
                    if (data.skeleton) {
                        skeletonFeed.src = 'data:image/jpeg;base64,' + data.skeleton;
                        skeletonFeed.style.display = 'inline-block';
                    }
                } catch (apiError) {
                    console.warn("Backend syncing...");
                }
            }
            
            // Wait 300ms BEFORE requesting the next frame so we NEVER queue overload the server!
            setTimeout(captureLoop, 300);
        }
        
        // Fire it up!
        captureLoop();

    } catch (err) {
        console.error("Error accessing the camera: ", err);
        alert("Camera access denied. Please allow it in the browser popup.");
    }
}

// ==========================================
// TEXT TO SIGN LOGIC
// ==========================================
async function playTextToSign() {
    const textInput = document.getElementById('t2s-input').value.trim();
    if (!textInput) return alert("Please enter some text to translate!");

    const statusMsg = document.getElementById('t2s-status');
    const placeholder = document.getElementById('t2s-placeholder');
    const imageFeed = document.getElementById('t2s-image');
    
    // Filter input
    const cleanText = [...textInput].filter(c => c.match(/[a-zA-Z\s]/));
    if (cleanText.length === 0) return alert("Please enter only letters and spaces.");

    placeholder.style.display = 'none';
    imageFeed.style.display = 'block';

    for (let char of cleanText) {
        if (char === ' ') {
            imageFeed.style.display = 'none';
            placeholder.style.display = 'flex';
            statusMsg.innerHTML = '<strong style="font-size:2rem">[ SPACE ]</strong>';
            await new Promise(r => setTimeout(r, 600));
            continue;
        }

        imageFeed.style.display = 'none';
        placeholder.style.display = 'flex';
        statusMsg.innerHTML = `Loading <strong>${char.toUpperCase()}</strong>...`;

        try {
            const resp = await fetch(`${API_BASE_URL}/get_sign/${char}`);
            const data = await resp.json();

            if (data.status === 'success') {
                imageFeed.src = 'data:image/jpeg;base64,' + data.image;
                placeholder.style.display = 'none';
                imageFeed.style.display = 'block';
            } else {
                placeholder.style.display = 'flex';
                imageFeed.style.display = 'none';
                statusMsg.innerHTML = `<span style="color:var(--accent-blue);">No sign found for: <strong>${char.toUpperCase()}</strong></span>`;
            }
        } catch (e) {
            console.error("Failed to load sign", e);
        }

        // Wait 1000ms between letters to simulate the "Speed scale" setting
        await new Promise(r => setTimeout(r, 1000));
    }
    
    placeholder.style.display = 'flex';
    imageFeed.style.display = 'none';
    statusMsg.innerHTML = "Translation Finished!";
}
