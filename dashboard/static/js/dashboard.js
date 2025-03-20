let socket;
let currentActivity = null;
let activityStartTime = new Date();
let confidenceChart;
let activityColors = {
    'walking': '#6c757d',
    'standing': '#3498db',
    'offline': '#e74c3c'
    // 'noactivity': '#2ecc71',
    // 'running': '#e67e22',
    // 'jumping': '#e74c3c',
    // 'sitting': '#9b59b6'
};

let currentAnimation = null;
let hasReceivedData = false;

const animations = {
    'walking': '../static/animations/walking1.json',
    'standing': '../static/animations/standing.json',
    'waiting': '../static/animations/noactivity.json',
    'offline': '../static/animations/noactivity.json'
};

const DEBUG = true;

// Initialize dashboard
function initDashboard() {
    initWebSocket();
    initConfidenceChart();
    updateActivityDuration();
    showWaitingState();
}

// Show waiting state
function showWaitingState() {
    if (DEBUG) console.log('Showing waiting state');
    const activityElement = document.getElementById('currentActivity');
    activityElement.innerHTML = `
        <div class="text-2xl">Waiting for CSI data...</div>
    `;
    activityElement.style.color = '#EF4444';
    updateLiveStatus(false);
    
    loadAnimation('waiting', 'Waiting for activity...');
}

// Update live status
function updateLiveStatus(isLive) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    if (isLive) {
        statusDot.classList.remove('bg-red-500');
        statusDot.classList.add('bg-green-500', 'animate-pulse');
        statusText.classList.remove('text-red-500');
        statusText.classList.add('text-green-500');
        statusText.textContent = 'Live';
    } else {
        statusDot.classList.remove('bg-green-500', 'animate-pulse');
        statusDot.classList.add('bg-red-500');
        statusText.classList.remove('text-green-500');
        statusText.classList.add('text-red-500');
        statusText.textContent = 'Not Live';
    }
}

// Load and play animation
function loadAnimation(animationType, labelText) {
    const container = document.getElementById('animationContainer');
    const label = document.getElementById('activityLabel');
    
    if (!container || !label) {
        console.error('Animation container or label not found');
        return;
    }

    if (currentAnimation) {
        currentAnimation.destroy();
        currentAnimation = null;
    }

    try {
        const animationPath = animations[animationType];
        
        if (!animationPath) {
            console.error('Animation path not found for:', animationType);
            return;
        }

        currentAnimation = lottie.loadAnimation({
            container: container,
            renderer: 'svg',
            loop: true,
            autoplay: true,
            path: animationPath,
            rendererSettings: {
                preserveAspectRatio: 'xMidYMid meet'
            }
        });

        if (animationType === 'offline') {
            label.textContent = 'System Offline';
        } else if (labelText === 'Standing') {
            label.textContent = 'Standing Detected';
        } else if (labelText === 'Walking') {
            label.textContent = 'Walking Detected';
        } else {
            label.textContent = labelText;
        }

        currentAnimation.addEventListener('error', (error) => {
            console.error('Animation error:', error);
            label.textContent = 'Animation Error';
        });

    } catch (error) {
        console.error('Failed to load animation:', error);
        label.textContent = 'Animation Error';
    }
}

// Update visualization based on activity
function updateVisualization(activity, confidence) {
    if (!activity || activity === 'Waiting for CSI data...' || activity === 'Waiting for data...') {
        loadAnimation('waiting', 'Waiting for activity...');
    } else if (activity === 'System Offline') {
        loadAnimation('offline', 'System Offline');
    } else {
        const standingConfidence = confidence[1];
        const walkingConfidence = confidence[0];
        if (standingConfidence > walkingConfidence) {
            loadAnimation('standing', 'Standing Detected');
        } else {
            loadAnimation('walking', 'Walking Detected');
        }
    }
}

// Initialize WebSocket connection
function initWebSocket() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('Connected to WebSocket server');
        if (!hasReceivedData) {
            showWaitingState();
            loadAnimation('waiting', 'Waiting for activity...');
        }
    });

    socket.on('connection_response', (data) => {
        console.log('Connection response:', data);
        updateLiveStatus(data.is_live);
    });

    socket.on('activity_update', (data) => {
        hasReceivedData = true;
        updateDashboard(data);
        updateLiveStatus(true);
    });

    socket.on('live_status', (data) => {
        updateLiveStatus(data.is_live);
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from WebSocket server');
        updateLiveStatus(false);
        showWaitingState();
        loadAnimation('waiting', 'Waiting for activity...');
    });

    socket.on('error', (error) => {
        console.error('WebSocket error:', error);
        updateLiveStatus(false);
        showWaitingState();
        loadAnimation('waiting', 'Waiting for activity...');
    });
}

// Initialize confidence chart
function initConfidenceChart() {
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    confidenceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Walking', 'Standing'],
            datasets: [{
                label: 'Confidence',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: Object.values(activityColors),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 500
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#fff',
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#fff'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Confidence: ${(context.raw * 100).toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Add new activity to timeline
function addTimelineItem(activity) {
    const timeline = document.getElementById('timeline');
    const time = new Date().toLocaleTimeString();
    
    const item = document.createElement('div');
    item.className = 'bg-gray-700 p-3 rounded-lg flex items-center justify-between slide-in';
    item.innerHTML = `
        <div class="flex items-center">
            <div class="w-3 h-3 rounded-full mr-3" style="background-color: ${activityColors[activity] || '#ffffff'}"></div>
            <span class="font-medium">${activity.charAt(0).toUpperCase() + activity.slice(1)}</span>
        </div>
        <span class="text-gray-400 text-sm">${time}</span>
    `;
    
    timeline.insertBefore(item, timeline.firstChild);
    while (timeline.children.length > 10) {
        timeline.removeChild(timeline.lastChild);
    }
}

// Update confidence chart
function updateConfidenceChart(scores) {
    if (confidenceChart) {
        const walkingScore = Math.min(Math.round(scores[0] * 100), 100);
        const standingScore = Math.min(Math.round(scores[1] * 100), 100);
        confidenceChart.data.datasets[0].data = [
            walkingScore,  // Walking
            standingScore  // Standing
        ];
        confidenceChart.update('none');
        const standingMetric = document.getElementById('standingMetric');
        const walkingMetric = document.getElementById('walkingMetric');
        
        walkingMetric.textContent = `${walkingScore}%`;
        standingMetric.textContent = `${standingScore}%`;
    }
}

// Update activity duration
function updateActivityDuration() {
    if (currentActivity) {
        const duration = Math.floor((new Date() - activityStartTime) / 1000);
        const durationElement = document.getElementById('activityDuration');
        durationElement.textContent = `Duration: ${duration}s`;
    }
    setTimeout(updateActivityDuration, 1000);
}

// Update the updateDashboard function
function updateDashboard(data) {
    const activity = data.hypothesis;
    const isOffline = data.is_offline;
    
    if (activity !== currentActivity || isOffline) {
        activityStartTime = new Date();
        currentActivity = activity;
        
        // Update current activity display
        const activityElement = document.getElementById('currentActivity');
        const lastUpdatedElement = document.getElementById('lastUpdated');
        
        activityElement.textContent = isOffline ? 'System Offline' : activity;
        activityElement.style.color = activityColors[activity.toLowerCase()] || '#ffffff';
        activityElement.classList.add('pulse');
        
        // Update last updated time
        const currentTime = new Date().toLocaleTimeString();
        lastUpdatedElement.querySelector('span').textContent = currentTime;
        
        // Update metrics display
        const standingMetric = document.getElementById('standingMetric');
        const walkingMetric = document.getElementById('walkingMetric');
        
        if (isOffline) {
            standingMetric.textContent = '0%';
            walkingMetric.textContent = '0%';
        } else {
            const walkingConfidence = Math.min(Math.round(data.confidence_scores[0] * 100), 100);
            const standingConfidence = Math.min(Math.round(data.confidence_scores[1] * 100), 100);
            
            standingMetric.textContent = `${standingConfidence}%`;
            walkingMetric.textContent = `${walkingConfidence}%`;
        }
        
        setTimeout(() => activityElement.classList.remove('pulse'), 1000);

        // Update visualization based on system state
        if (isOffline) {
            loadAnimation('offline', 'System Offline');
        } else if (activity === 'Standing') {
            loadAnimation('standing', 'Standing');
        } else if (activity === 'Walking') {
            loadAnimation('walking', 'Walking');
        } else if (activity === 'Waiting for CSI data...') {
            loadAnimation('waiting', 'Waiting for activity...');
        }
        addTimelineItem(activity);
    }
    updateConfidenceChart(data.confidence_scores);
    updateActivityDuration();
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Check if animation files exist
    fetch('../static/animations/noactivity.json')
        .then(response => {
            if (!response.ok) throw new Error('Animation file not found');
            return response.json();
        })
        .then(data => {
            console.log('Animation file loaded successfully');
        })
        .catch(error => {
            console.error('Error loading animation file:', error);
        });

    initDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (socket) {
        socket.disconnect();
    }
    if (currentAnimation) {
        currentAnimation.destroy();
    }
});
