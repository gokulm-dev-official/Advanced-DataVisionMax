// Enhanced DataVision-AI Landing Page JavaScript

class DataVisionLanding {
    constructor() {
        this.currentFeatureIndex = 0;
        this.features = [
            {
                icon: '📊',
                title: 'Instant Visual Reports',
                description: 'Transform raw data into stunning, interactive charts and graphs instantly. Generate PowerBI-quality dashboards with advanced filtering, drill-down capabilities, and real-time updates.',
                color: 'from-blue-500 to-cyan-500',
                details: ['Interactive Charts', 'Real-time Updates', 'Advanced Filtering', 'Drill-down Analysis']
            },
            {
                icon: '🤖',
                title: 'AI-Powered Insights',
                description: 'Advanced AI algorithms analyze your data to discover hidden patterns, seasonal trends, and business opportunities. Get specific insights like "March is optimal for Electronics sales".',
                color: 'from-purple-500 to-pink-500',
                details: ['Pattern Recognition', 'Seasonal Analysis', 'Business Intelligence', 'Automated Insights']
            },
            {
                icon: '🔮',
                title: 'Predictive Forecasts',
                description: 'Machine learning models predict future trends, sales forecasts, and business outcomes with confidence intervals. Plan ahead with data-driven predictions.',
                color: 'from-green-500 to-emerald-500',
                details: ['Sales Forecasting', 'Trend Prediction', 'Confidence Intervals', 'Risk Assessment']
            },
            {
                icon: '🧹',
                title: 'Smart Data Cleaning',
                description: 'Automatically detect and fix missing values, remove duplicates, handle outliers, and normalize your datasets. Download cleaned data in CSV, Excel, or JSON formats.',
                color: 'from-orange-500 to-red-500',
                details: ['Auto Data Cleaning', 'Outlier Detection', 'Multiple Export Formats', 'Quality Scoring']
            },
            {
                icon: '📱',
                title: 'Interactive Dashboards',
                description: 'Create responsive, professional dashboards with drag-and-drop functionality, custom KPIs, and real-time collaboration features.',
                color: 'from-indigo-500 to-blue-500',
                details: ['Responsive Design', 'Custom KPIs', 'Real-time Collaboration', 'Mobile Optimized']
            },
            {
                icon: '📈',
                title: 'Business KPI Monitoring',
                description: 'Track revenue, sales, conversion rates, and custom business metrics with automated alerts, trend analysis, and performance benchmarking.',
                color: 'from-teal-500 to-green-500',
                details: ['Revenue Tracking', 'Performance Alerts', 'Trend Analysis', 'Benchmarking']
            },
            {
                icon: '📄',
                title: 'Professional Reports',
                description: 'Generate executive-ready PDF reports with charts, insights, recommendations, and your company branding. Perfect for board meetings and stakeholder presentations.',
                color: 'from-purple-500 to-indigo-500',
                details: ['Executive Ready', 'Custom Branding', 'Chart Integration', 'Professional Layout']
            },
            {
                icon: '🔗',
                title: 'Google Sheets Integration',
                description: 'Connect directly to Google Sheets for live data analysis. Changes in your spreadsheet automatically update your dashboard and insights in real-time.',
                color: 'from-pink-500 to-rose-500',
                details: ['Live Data Sync', 'Auto Updates', 'Real-time Analysis', 'Seamless Integration']
            }
        ];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.animateIntro();
        this.initParticles();
        this.createProgressIndicator();
    }

    setupEventListeners() {
        const startBtn = document.getElementById('start-btn');
        const demoBtn = document.getElementById('demo-btn');
        const skipShowcase = document.getElementById('skip-showcase');

        startBtn.addEventListener('click', () => this.startFeatureShowcase());
        demoBtn.addEventListener('click', () => this.startDemoMode());
        skipShowcase.addEventListener('click', () => this.navigateToDashboard());

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !document.getElementById('feature-showcase').classList.contains('hidden')) {
                this.navigateToDashboard();
            }
            if (e.key === 'Escape') {
                this.navigateToDashboard();
            }
        });
    }

    animateIntro() {
        // Staggered animations for better visual impact
        setTimeout(() => {
            document.getElementById('main-title').classList.add('fade-in');
            document.getElementById('main-title').style.opacity = '1';
        }, 500);

        setTimeout(() => {
            document.getElementById('feature-highlights').classList.add('fade-in');
            document.getElementById('feature-highlights').style.opacity = '1';
        }, 1500);

        setTimeout(() => {
            document.getElementById('action-buttons').classList.add('fade-in');
            document.getElementById('action-buttons').style.opacity = '1';
        }, 2500);
    }

    initParticles() {
        // Enhanced particles configuration
        if (typeof particlesJS !== 'undefined') {
            particlesJS('particles-js', {
                particles: {
                    number: {
                        value: 80,
                        density: {
                            enable: true,
                            value_area: 800
                        }
                    },
                    color: {
                        value: ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981']
                    },
                    shape: {
                        type: 'circle'
                    },
                    opacity: {
                        value: 0.6,
                        random: true,
                        anim: {
                            enable: true,
                            speed: 1,
                            opacity_min: 0.1
                        }
                    },
                    size: {
                        value: 3,
                        random: true,
                        anim: {
                            enable: true,
                            speed: 4,
                            size_min: 0.3
                        }
                    },
                    line_linked: {
                        enable: true,
                        distance: 150,
                        color: '#3b82f6',
                        opacity: 0.4,
                        width: 1
                    },
                    move: {
                        enable: true,
                        speed: 2,
                        direction: 'none',
                        random: false,
                        straight: false,
                        out_mode: 'out',
                        bounce: false
                    }
                },
                interactivity: {
                    detect_on: 'canvas',
                    events: {
                        onhover: {
                            enable: true,
                            mode: 'repulse'
                        },
                        onclick: {
                            enable: true,
                            mode: 'push'
                        },
                        resize: true
                    },
                    modes: {
                        repulse: {
                            distance: 100,
                            duration: 0.4
                        },
                        push: {
                            particles_nb: 4
                        }
                    }
                },
                retina_detect: true
            });
        }
    }

    createProgressIndicator() {
        const progressContainer = document.getElementById('progress-indicator');
        progressContainer.innerHTML = '';
        
        this.features.forEach((_, index) => {
            const dot = document.createElement('div');
            dot.className = 'progress-dot';
            if (index === 0) dot.classList.add('active');
            progressContainer.appendChild(dot);
        });
    }

    updateProgressIndicator(activeIndex) {
        const dots = document.querySelectorAll('.progress-dot');
        dots.forEach((dot, index) => {
            dot.classList.toggle('active', index === activeIndex);
        });
    }

    startFeatureShowcase() {
        const showcase = document.getElementById('feature-showcase');
        showcase.classList.remove('hidden');
        this.currentFeatureIndex = 0;
        this.showFeature(this.currentFeatureIndex);
    }

    startDemoMode() {
        // Skip to dashboard with demo data
        this.navigateToDashboard(true);
    }

    showFeature(index) {
        if (index >= this.features.length) {
            this.navigateToDashboard();
            return;
        }

        const feature = this.features[index];
        const content = document.getElementById('feature-content');
        
        // Fade out current content
        content.style.opacity = '0';
        content.style.transform = 'scale(0.95)';
        
        setTimeout(() => {
            content.innerHTML = `
                <div class="feature-slide max-w-4xl mx-auto">
                    <div class="feature-icon text-6xl md:text-8xl mb-8">${feature.icon}</div>
                    <h2 class="text-4xl md:text-6xl font-bold mb-8 bg-gradient-to-r ${feature.color} bg-clip-text text-transparent">
                        ${feature.title}
                    </h2>
                    <p class="text-lg md:text-xl text-gray-300 leading-relaxed mb-8 max-w-3xl mx-auto">
                        ${feature.description}
                    </p>
                    
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                        ${feature.details.map(detail => `
                            <div class="bg-gray-800 bg-opacity-50 p-3 rounded-lg border border-gray-600">
                                <span class="text-sm text-gray-300">${detail}</span>
                            </div>
                        `).join('')}
                    </div>
                    
                    <div class="w-full bg-gray-700 rounded-full h-2 mb-4">
                        <div class="bg-gradient-to-r ${feature.color} h-2 rounded-full transition-all duration-1000" 
                             style="width: ${((index + 1) / this.features.length) * 100}%"></div>
                    </div>
                    <p class="text-sm text-gray-400">${index + 1} of ${this.features.length} features</p>
                </div>
            `;

            // Fade in new content
            content.style.opacity = '1';
            content.style.transform = 'scale(1)';
            
            this.updateProgressIndicator(index);
        }, 300);

        // Auto-advance to next feature
        setTimeout(() => {
            this.currentFeatureIndex++;
            this.showFeature(this.currentFeatureIndex);
        }, 4000);
    }

    navigateToDashboard(isDemo = false) {
        const loadingScreen = document.getElementById('loading-screen');
        loadingScreen.classList.remove('hidden');

        // Simulate more realistic loading time
        const loadingMessages = [
            'Initializing AI engines...',
            'Loading visualization libraries...',
            'Preparing analytics workspace...',
            'Almost ready...'
        ];

        let messageIndex = 0;
        const loadingText = document.querySelector('#loading-screen p');
        
        const messageInterval = setInterval(() => {
            if (messageIndex < loadingMessages.length) {
                loadingText.textContent = loadingMessages[messageIndex];
                messageIndex++;
            }
        }, 500);

        setTimeout(() => {
            clearInterval(messageInterval);
            window.location.href = `dashboard.html${isDemo ? '?mode=demo' : ''}`;
        }, 3000);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DataVisionLanding();
});

// Enhanced visual effects (removed cursor following particles as requested)
document.addEventListener('mousemove', (e) => {
    // Create subtle glow effect on buttons
    const buttons = document.querySelectorAll('.glow-button');
    buttons.forEach(button => {
        const rect = button.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
            button.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(59, 130, 246, 0.3), transparent 50%)`;
        }
    });
});

// Add smooth scrolling and performance optimizations
window.addEventListener('beforeunload', () => {
    // Clean up particles if needed
    if (window.pJSDom && window.pJSDom[0]) {
        window.pJSDom[0].pJS.fn.vendors.destroyImg();
    }
});