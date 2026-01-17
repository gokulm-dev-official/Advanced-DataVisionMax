// Enhanced DataVision-AI Dashboard JavaScript - Complete Version

class DataVisionDashboard {
    constructor() {
        this.sessionId = null;
        this.currentDataset = null;
        this.charts = {};
        this.apiBase = 'http://localhost:5000';
        this.chartInstances = {};
        this.isDemoMode = false;
        this.retryCount = 0;
        this.maxRetries = 3;
        this.init();
    }

    init() {
        console.log('🚀 Initializing DataVision-AI Dashboard...');
        this.checkDemoMode();
        this.setupEventListeners();
        this.setupDropZone();
        this.initializeTooltips();
        this.setupKeyboardShortcuts();
        this.testBackendConnection();
    }

    async testBackendConnection() {
        try {
            console.log('🔗 Testing backend connection...');
            const response = await fetch(`${this.apiBase}/test_upload`);
            const result = await response.json();
            console.log('✅ Backend connection successful:', result);
        } catch (error) {
            console.error('❌ Backend connection failed:', error);
            this.showError('Cannot connect to backend server. Please ensure the backend is running on port 5000.');
        }
    }

    checkDemoMode() {
        const urlParams = new URLSearchParams(window.location.search);
        this.isDemoMode = urlParams.get('mode') === 'demo';
        
        if (this.isDemoMode) {
            console.log('🎭 Demo mode activated');
            this.loadDemoData();
        }
    }

    async loadDemoData() {
        this.showLoading('Loading demo data...');
        try {
            const response = await fetch(`${this.apiBase}/connect_sheets`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    sheets_url: 'https://docs.google.com/spreadsheets/d/demo-business-data' 
                })
            });

            const result = await response.json();
            if (response.ok) {
                this.sessionId = result.session_id;
                this.displayDatasetInfo(result);
                this.displayCleaningReport(result.cleaning_report);
                await this.loadDashboard();
                this.showSuccess('Demo data loaded successfully! Explore all features with sample business data.');
            } else {
                this.showError('Demo mode failed to load: ' + result.error);
            }
        } catch (error) {
            console.error('❌ Demo mode error:', error);
            this.showError('Demo mode initialization failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    setupEventListeners() {
        console.log('🎧 Setting up event listeners...');
        
        // File upload
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileUpload(e.target.files[0]);
                }
            });
        }

        // Google Sheets connection
        const connectBtn = document.getElementById('connect-sheets-btn');
        if (connectBtn) {
            connectBtn.addEventListener('click', () => {
                this.connectGoogleSheets();
            });
        }

        // Export functionalities
        const exportBtn = document.getElementById('export-pdf-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportPDF();
            });
        }

        // Refresh button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshDashboard();
            });
        }

        // Data download buttons
        this.setupDownloadButtons();

        // Chat functionality
        const chatSend = document.getElementById('chat-send');
        const chatInput = document.getElementById('chat-input');
        
        if (chatSend) {
            chatSend.addEventListener('click', () => {
                this.sendChatMessage();
            });
        }

        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendChatMessage();
                }
            });
        }

        // Settings dropdown
        this.setupSettingsDropdown();
    }

    setupDownloadButtons() {
        // Create download section if it doesn't exist
        let downloadSection = document.getElementById('download-section');
        if (!downloadSection) {
            downloadSection = document.createElement('div');
            downloadSection.id = 'download-section';
            downloadSection.className = 'hidden space-y-4';
            downloadSection.innerHTML = `
                <h3 class="text-lg font-semibold text-purple-400 flex items-center">
                    📥 Download Data
                    <span class="ml-2 text-xs bg-purple-900 text-purple-200 px-2 py-1 rounded-full">Clean</span>
                </h3>
                <div class="grid grid-cols-1 gap-2">
                    <button id="download-csv" class="px-3 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors text-sm flex items-center justify-center space-x-2">
                        <span>📄</span><span>CSV Format</span>
                    </button>
                    <button id="download-excel" class="px-3 py-2 bg-green-600 rounded-lg hover:bg-green-700 transition-colors text-sm flex items-center justify-center space-x-2">
                        <span>📊</span><span>Excel Format</span>
                    </button>
                    <button id="download-json" class="px-3 py-2 bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors text-sm flex items-center justify-center space-x-2">
                        <span>🔧</span><span>JSON Format</span>
                    </button>
                </div>
            `;

            const sidebar = document.querySelector('.w-80');
            if (sidebar) {
                sidebar.appendChild(downloadSection);
            }
        }

        // Add event listeners
        const downloadCsv = document.getElementById('download-csv');
        const downloadExcel = document.getElementById('download-excel');
        const downloadJson = document.getElementById('download-json');
        
        if (downloadCsv) downloadCsv.addEventListener('click', () => this.downloadCleanedData('csv'));
        if (downloadExcel) downloadExcel.addEventListener('click', () => this.downloadCleanedData('excel'));
        if (downloadJson) downloadJson.addEventListener('click', () => this.downloadCleanedData('json'));
    }

    setupSettingsDropdown() {
        const settingsBtn = document.getElementById('settings-btn');
        const settingsDropdown = document.getElementById('settings-dropdown');
        
        if (settingsBtn && settingsDropdown) {
            settingsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                settingsDropdown.classList.toggle('hidden');
            });

            document.addEventListener('click', () => {
                settingsDropdown.classList.add('hidden');
            });
        }
    }

    setupDropZone() {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');

        if (!dropZone || !fileInput) {
            console.warn('⚠️ Drop zone or file input not found');
            return;
        }

        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'e':
                        e.preventDefault();
                        this.exportPDF();
                        break;
                    case 'r':
                        e.preventDefault();
                        this.refreshDashboard();
                        break;
                    case 'd':
                        e.preventDefault();
                        this.downloadCleanedData('csv');
                        break;
                    case 'h':
                        e.preventDefault();
                        this.showKeyboardShortcuts();
                        break;
                    case '/':
                        e.preventDefault();
                        const chatInput = document.getElementById('chat-input');
                        if (chatInput) chatInput.focus();
                        break;
                }
            }
            
            if (e.key === 'Escape') {
                const modal = document.getElementById('chart-modal');
                const dropdown = document.getElementById('settings-dropdown');
                if (modal) modal.classList.add('hidden');
                if (dropdown) dropdown.classList.add('hidden');
            }
        });
    }

    initializeTooltips() {
        const tooltipElements = [
            { selector: '#export-pdf-btn', text: 'Export comprehensive PDF report (Ctrl+E)' },
            { selector: '#download-csv', text: 'Download cleaned dataset as CSV (Ctrl+D)' },
            { selector: '#chat-send', text: 'Ask AI about your data' },
            { selector: '#refresh-btn', text: 'Refresh Dashboard (Ctrl+R)' }
        ];

        tooltipElements.forEach(({ selector, text }) => {
            const element = document.querySelector(selector);
            if (element) {
                element.title = text;
            }
        });
    }

    async handleFileUpload(file) {
        if (!file) {
            console.warn('⚠️ No file provided');
            return;
        }

        console.log('📁 Starting file upload:', file.name, 'Size:', file.size, 'bytes', 'Type:', file.type);

        // Validate file type
        const allowedExtensions = ['.csv', '.xlsx', '.xls', '.json'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedExtensions.includes(fileExtension)) {
            this.showError(`Unsupported file type: ${fileExtension}. Please upload: ${allowedExtensions.join(', ')}`);
            return;
        }

        // Validate file size (max 100MB)
        const maxSize = 100 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File too large. Maximum size is 100MB.');
            return;
        }

        if (file.size === 0) {
            this.showError('File is empty. Please select a valid file.');
            return;
        }

        this.showLoading(`Uploading and processing ${file.name}...`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            console.log('📤 Sending file to server...');
            const response = await fetch(`${this.apiBase}/upload_dataset`, {
                method: 'POST',
                body: formData
            });

            console.log('📥 Response received:', response.status, response.statusText);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('📊 Processing result:', result);

            if (result.session_id) {
                this.sessionId = result.session_id;
                this.displayDatasetInfo(result);
                
                if (result.cleaning_report) {
                    this.displayCleaningReport(result.cleaning_report);
                }
                
                // Show connected status
                this.showConnectedStatus();
                
                await this.loadDashboard();
                this.showSuccess(`Dataset "${file.name}" uploaded and processed successfully!`);
                
                console.log('✅ File processing completed successfully');
            } else {
                throw new Error('No session ID returned from server');
            }

        } catch (error) {
            console.error('❌ Upload error:', error);
            this.showError(`Upload failed: ${error.message}`);
            
            // Retry logic for network errors
            if (this.retryCount < this.maxRetries && error.message.includes('fetch')) {
                this.retryCount++;
                console.log(`🔄 Retrying upload (${this.retryCount}/${this.maxRetries})...`);
                setTimeout(() => this.handleFileUpload(file), 2000);
                return;
            }
        } finally {
            this.hideLoading();
            this.retryCount = 0;
        }
    }

    showConnectedStatus() {
        const statusIndicator = document.getElementById('status-indicator');
        const downloadSection = document.getElementById('download-section');
        const quickStats = document.getElementById('quick-stats');
        
        if (statusIndicator) statusIndicator.classList.remove('hidden');
        if (downloadSection) downloadSection.classList.remove('hidden');
        if (quickStats) quickStats.classList.remove('hidden');
    }

    async connectGoogleSheets() {
        const sheetsUrl = document.getElementById('sheets-url');
        if (!sheetsUrl || !sheetsUrl.value.trim()) {
            this.showError('Please enter a Google Sheets URL');
            return;
        }

        const url = sheetsUrl.value.trim();
        console.log('🔗 Connecting to Google Sheets:', url);

        this.showLoading('Connecting to Google Sheets...');

        try {
            const response = await fetch(`${this.apiBase}/connect_sheets`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ sheets_url: url })
            });

            const result = await response.json();

            if (response.ok && result.session_id) {
                this.sessionId = result.session_id;
                this.displayDatasetInfo(result);
                
                if (result.cleaning_report) {
                    this.displayCleaningReport(result.cleaning_report);
                }
                
                this.showConnectedStatus();
                await this.loadDashboard();
                this.showSuccess('Google Sheets connected successfully!');
            } else {
                this.showError(result.error || 'Failed to connect to Google Sheets');
            }
        } catch (error) {
            console.error('❌ Sheets connection error:', error);
            this.showError('Network error: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayDatasetInfo(data) {
        console.log('📊 Displaying dataset info:', data);
        
        const infoDiv = document.getElementById('dataset-info');
        const statsDiv = document.getElementById('dataset-stats');
        
        if (!infoDiv || !statsDiv) {
            console.warn('⚠️ Dataset info elements not found');
            return;
        }
        
        const missingValues = data.stats?.missing_values || {};
        const totalMissing = Object.values(missingValues).reduce((a, b) => a + b, 0);
        const qualityScore = data.cleaning_report?.data_quality_score || 85;
        
        statsDiv.innerHTML = `
            <div class="space-y-3">
                <div class="flex justify-between items-center p-2 bg-gray-700 rounded">
                    <span class="text-gray-300">📊 Rows</span>
                    <span class="text-green-400 font-semibold">${(data.rows || 0).toLocaleString()}</span>
                </div>
                <div class="flex justify-between items-center p-2 bg-gray-700 rounded">
                    <span class="text-gray-300">📋 Columns</span>
                    <span class="text-blue-400 font-semibold">${(data.columns || []).length}</span>
                </div>
                <div class="flex justify-between items-center p-2 bg-gray-700 rounded">
                    <span class="text-gray-300">🗂️ Fields</span>
                    <span class="text-purple-400 font-semibold">${data.stats?.total_columns || 0}</span>
                </div>
                <div class="flex justify-between items-center p-2 bg-gray-700 rounded">
                    <span class="text-gray-300">⚠️ Missing</span>
                    <span class="text-yellow-400 font-semibold">${totalMissing}</span>
                </div>
                <div class="mt-4">
                    <div class="text-xs text-gray-400 mb-2">Data Quality Score</div>
                    <div class="w-full bg-gray-700 rounded-full h-2">
                        <div class="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-1000" 
                             style="width: ${qualityScore}%"></div>
                    </div>
                    <div class="text-xs text-gray-400 mt-1">${qualityScore}% Quality</div>
                </div>
            </div>
        `;
        
        infoDiv.classList.remove('hidden');
        
        // Update quick stats
        this.updateQuickStats(data);
    }

    updateQuickStats(data) {
        const statsGrid = document.getElementById('stats-grid');
        if (!statsGrid) return;

        const stats = data.stats || {};
        const numericCols = Object.keys(stats.numeric_summary || {});
        const categoricalCols = Object.keys(stats.categorical_summary || {});
        
        statsGrid.innerHTML = `
            <div class="bg-blue-900 bg-opacity-30 p-3 rounded-lg border border-blue-500 border-opacity-30">
                <div class="text-lg font-bold text-blue-400">${numericCols.length}</div>
                <div class="text-xs text-gray-400">Numeric</div>
            </div>
            <div class="bg-green-900 bg-opacity-30 p-3 rounded-lg border border-green-500 border-opacity-30">
                <div class="text-lg font-bold text-green-400">${categoricalCols.length}</div>
                <div class="text-xs text-gray-400">Categories</div>
            </div>
            <div class="bg-purple-900 bg-opacity-30 p-3 rounded-lg border border-purple-500 border-opacity-30">
                <div class="text-lg font-bold text-purple-400">${stats.memory_usage || 'N/A'}</div>
                <div class="text-xs text-gray-400">Memory</div>
            </div>
            <div class="bg-yellow-900 bg-opacity-30 p-3 rounded-lg border border-yellow-500 border-opacity-30">
                <div class="text-lg font-bold text-yellow-400">${(stats.data_quality_metrics?.overall_quality_score || 85)}%</div>
                <div class="text-xs text-gray-400">Quality</div>
            </div>
        `;
    }

    displayCleaningReport(cleaningReport) {
        if (!cleaningReport) {
            console.warn('⚠️ No cleaning report provided');
            return;
        }

        console.log('🧹 Displaying cleaning report:', cleaningReport);
        
        let cleaningSection = document.getElementById('cleaning-report');
        if (!cleaningSection) {
            cleaningSection = document.createElement('div');
            cleaningSection.id = 'cleaning-report';
            cleaningSection.className = 'space-y-4';
            
            const sidebar = document.querySelector('.w-80');
            if (sidebar) {
                sidebar.appendChild(cleaningSection);
            }
        }

        const steps = cleaningReport.cleaning_steps || [];
        const recommendations = cleaningReport.recommendations || [];

        cleaningSection.innerHTML = `
            <div class="flex items-center justify-between">
                <h3 class="text-lg font-semibold text-orange-400">🧹 Data Cleaning</h3>
                <div class="text-xs bg-orange-900 text-orange-200 px-2 py-1 rounded-full">Complete</div>
            </div>
            <div class="space-y-3 text-sm">
                <div class="bg-gray-700 p-3 rounded-lg">
                    <div class="text-green-400 font-semibold mb-2">Quality Score: ${cleaningReport.data_quality_score || 85}%</div>
                    <div class="text-gray-300 space-y-1">
                        <div>Original: ${cleaningReport.original_shape?.[0] || 0} × ${cleaningReport.original_shape?.[1] || 0}</div>
                        <div>Cleaned: ${cleaningReport.cleaned_shape?.[0] || 0} × ${cleaningReport.cleaned_shape?.[1] || 0}</div>
                        <div class="text-yellow-400">Rows removed: ${cleaningReport.rows_removed || 0}</div>
                    </div>
                </div>
                <div class="max-h-32 overflow-y-auto">
                    <div class="text-xs text-gray-400 mb-2">Cleaning Steps:</div>
                    ${steps.map(step => 
                        `<div class="text-xs text-gray-400 p-1">• ${step}</div>`
                    ).join('')}
                </div>
                ${recommendations.length > 0 ? `
                    <div class="max-h-20 overflow-y-auto">
                        <div class="text-xs text-gray-400 mb-2">Recommendations:</div>
                        ${recommendations.map(rec => 
                            `<div class="text-xs text-blue-400 p-1">• ${rec}</div>`
                        ).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }

    async loadDashboard() {
        if (!this.sessionId) {
            console.warn('⚠️ No session ID available for dashboard loading');
            return;
        }

        console.log('📊 Loading dashboard for session:', this.sessionId);

        const welcomeMessage = document.getElementById('welcome-message');
        const dashboardContent = document.getElementById('dashboard-content');
        const aiChat = document.getElementById('ai-chat');
        
        if (welcomeMessage) welcomeMessage.classList.add('hidden');
        if (dashboardContent) dashboardContent.classList.remove('hidden');
        if (aiChat) aiChat.classList.remove('hidden');

        // Load all dashboard components
        const loadingPromises = [
            this.loadCharts(),
            this.loadInsights(),
            this.loadPredictions(),
            this.loadAnomalies(),
            this.loadDataSummary()
        ];

        try {
            await Promise.all(loadingPromises);
            this.addInteractivityToCharts();
            this.updateLastUpdatedTime();
            console.log('✅ Dashboard loaded successfully');
        } catch (error) {
            console.error('❌ Dashboard loading error:', error);
            this.showError('Some dashboard components failed to load: ' + error.message);
        }
    }

    async loadCharts() {
        try {
            console.log('📊 Loading charts...');
            const response = await fetch(`${this.apiBase}/generate_charts/${this.sessionId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const charts = await response.json();
            console.log('📈 Charts data received:', Object.keys(charts));

            if (charts.kpis) {
                this.displayKPIs(charts.kpis);
            }
            
            this.displayCharts(charts);
            this.charts = charts;
            
        } catch (error) {
            console.error('❌ Error loading charts:', error);
            this.showChartError('Failed to load charts: ' + error.message);
        }
    }

    displayKPIs(kpis) {
        const kpiSection = document.getElementById('kpi-section');
        if (!kpiSection) {
            console.warn('⚠️ KPI section not found');
            return;
        }

        console.log('📊 Displaying KPIs:', Object.keys(kpis));
        kpiSection.innerHTML = '';

        Object.entries(kpis).forEach(([metric, values]) => {
            const kpiCard = document.createElement('div');
            kpiCard.className = 'kpi-card p-6 rounded-lg relative overflow-hidden';
            
            const growthRate = values.growth_rate || 0;
            const growthColor = growthRate >= 0 ? 'text-green-400' : 'text-red-400';
            const growthIcon = growthRate >= 0 ? '📈' : '📉';

            kpiCard.innerHTML = `
                <div class="relative z-10">
                    <div class="flex items-center justify-between mb-4">
                        <h4 class="text-sm font-medium text-gray-400 uppercase tracking-wide">${metric.replace(/_/g, ' ')}</h4>
                        <div class="text-2xl opacity-50">${this.getMetricIcon(metric)}</div>
                    </div>
                    
                    <div class="mb-4">
                        <p class="text-3xl font-bold text-white mb-1">${this.formatNumber(values.total || 0)}</p>
                        <p class="text-sm text-gray-400">
                            Avg: ${this.formatNumber(values.average || 0)} | 
                            Range: ${this.formatNumber(values.min || 0)} - ${this.formatNumber(values.max || 0)}
                        </p>
                    </div>
                    
                    <div class="flex items-center justify-between">
                        <div class="${growthColor} text-sm font-medium flex items-center">
                            <span class="mr-1">${growthIcon}</span>
                            ${Math.abs(growthRate).toFixed(1)}% Growth
                        </div>
                        <div class="text-xs text-gray-500">
                            ${values.count || 0} records
                        </div>
                    </div>
                </div>
            `;

            kpiSection.appendChild(kpiCard);
        });
    }

    getMetricIcon(metric) {
        const icons = {
            'sales_amount': '💰',
            'profit': '📈',
            'quantity_sold': '📦',
            'discount_applied': '🏷️',
            'revenue': '💵',
            'orders': '🛒',
            'customers': '👥',
            'transactions': '🧾'
        };
        
        const key = metric.toLowerCase().replace(/[^a-z]/g, '_');
        return icons[key] || '📊';
    }

    displayCharts(chartsData) {
        const chartsSection = document.getElementById('charts-section');
        if (!chartsSection) {
            console.warn('⚠️ Charts section not found');
            return;
        }

        console.log('📊 Displaying charts:', Object.keys(chartsData));
        chartsSection.innerHTML = '';

        // Revenue/Time Series Trend Chart
        if (chartsData.revenue_trend || chartsData.time_series) {
            const trendData = chartsData.revenue_trend || (chartsData.time_series?.[0]?.chart);
            if (trendData) {
                this.createPlotlyChart('revenue-trend', trendData, 'Trend Analysis', 'line');
            }
        }

        // Distribution Charts
        if (chartsData.distributions && Array.isArray(chartsData.distributions)) {
            chartsData.distributions.slice(0, 4).forEach((chart, index) => {
                if (chart && chart.chart) {
                    this.createPlotlyChart(`distribution-${index}`, chart.chart, 
                        `${chart.column} Distribution`, 'histogram');
                }
            });
        }

        // Categorical Charts
        if (chartsData.categorical && Array.isArray(chartsData.categorical)) {
            chartsData.categorical.slice(0, 4).forEach((chart, index) => {
                if (chart && chart.chart) {
                    this.createPlotlyChart(`categorical-${index}`, chart.chart, 
                        `${chart.column} ${chart.type}`, chart.type);
                }
            });
        }

        // Correlation Heatmap
        if (chartsData.correlation) {
            this.createPlotlyChart('correlation', chartsData.correlation, 'Correlation Heatmap', 'heatmap');
        }

        // Advanced business charts
        this.createAdvancedCharts(chartsData);
    }

    createAdvancedCharts(chartsData) {
        // Monthly Performance Chart
        if (chartsData.monthly_performance) {
            this.createPlotlyChart('monthly-performance', chartsData.monthly_performance, 
                'Monthly Performance Analysis', 'bar');
        }

        // Product Category Performance
        if (chartsData.category_performance) {
            this.createPlotlyChart('category-performance', chartsData.category_performance, 
                'Category Performance', 'scatter');
        }

        // Regional Analysis
        if (chartsData.regional_analysis) {
            this.createPlotlyChart('regional-analysis', chartsData.regional_analysis, 
                'Regional Analysis', 'bar');
        }
    }

    createPlotlyChart(id, chartData, title, chartType) {
        const chartsSection = document.getElementById('charts-section');
        if (!chartsSection) return;
        
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container relative';
        chartContainer.innerHTML = `
            <div class="flex justify-between items-center mb-4">
                <h4 class="text-lg font-semibold text-white">${title}</h4>
                <div class="flex space-x-2">
                    <button class="export-chart-btn text-xs px-3 py-1 bg-blue-600 rounded hover:bg-blue-700" 
                            data-chart-id="${id}" data-title="${title}">
                        📤 Export
                    </button>
                    <button class="fullscreen-chart-btn text-xs px-3 py-1 bg-gray-600 rounded hover:bg-gray-700" 
                            data-chart-id="${id}">
                        ⛶ Fullscreen
                    </button>
                </div>
            </div>
            <div id="${id}" style="height: 400px;"></div>
        `;
        
        chartsSection.appendChild(chartContainer);

        try {
            // Configure Plotly with enhanced dark theme
            const layout = {
                ...chartData.layout,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(31, 41, 55, 0.8)',
                font: { 
                    color: '#ffffff',
                    family: 'Inter, sans-serif'
                },
                margin: { t: 40, r: 40, b: 60, l: 60 },
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: -0.2,
                    x: 0.5,
                    xanchor: 'center'
                },
                hovermode: 'closest',
                hoverlabel: {
                    bgcolor: 'rgba(31, 41, 55, 0.95)',
                    bordercolor: '#3b82f6',
                    font: { color: '#ffffff' }
                }
            };

            // Enhanced data configuration
            const enhancedData = chartData.data?.map(trace => ({
                ...trace,
                hovertemplate: this.getHoverTemplate(chartType),
                line: chartType === 'line' ? { 
                    color: '#3b82f6', 
                    width: 3,
                    shape: 'spline'
                } : trace.line,
                marker: {
                    ...trace.marker,
                    size: chartType === 'scatter' ? 8 : trace.marker?.size,
                    opacity: 0.8
                }
            })) || [];

            const config = {
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                responsive: true,
                displaylogo: false
            };

            Plotly.newPlot(id, enhancedData, layout, config);

            // Store chart instance
            this.chartInstances[id] = { data: enhancedData, layout, config };

            // Add event listeners for chart controls
            const exportBtn = chartContainer.querySelector('.export-chart-btn');
            const fullscreenBtn = chartContainer.querySelector('.fullscreen-chart-btn');
            
            if (exportBtn) {
                exportBtn.addEventListener('click', (e) => {
                    this.exportChartImage(e.target.dataset.chartId, e.target.dataset.title);
                });
            }
            
            if (fullscreenBtn) {
                fullscreenBtn.addEventListener('click', (e) => {
                    this.openChartFullscreen(e.target.dataset.chartId);
                });
            }

        } catch (error) {
            console.error('❌ Error creating chart:', error);
            chartContainer.innerHTML = `
                <div class="flex items-center justify-center h-96 bg-gray-800 rounded-lg">
                    <div class="text-center text-gray-400">
                        <div class="text-4xl mb-2">⚠️</div>
                        <div>Chart Error</div>
                        <div class="text-sm mt-1">${error.message}</div>
                    </div>
                </div>
            `;
        }
    }

    showChartError(message) {
        const chartsSection = document.getElementById('charts-section');
        if (!chartsSection) return;

        chartsSection.innerHTML = `
            <div class="col-span-full flex items-center justify-center h-64 bg-gray-800 rounded-lg">
                <div class="text-center text-gray-400">
                    <div class="text-4xl mb-4">📊</div>
                    <div class="text-lg font-semibold mb-2">Charts Unavailable</div>
                    <div class="text-sm">${message}</div>
                </div>
            </div>
        `;
    }

    getHoverTemplate(chartType) {
        const templates = {
            'line': '<b>%{fullData.name}</b><br>%{x}<br>%{y:,.2f}<extra></extra>',
            'bar': '<b>%{fullData.name}</b><br>%{x}<br>%{y:,.2f}<extra></extra>',
            'pie': '<b>%{label}</b><br>%{value:,.2f} (%{percent})<extra></extra>',
            'histogram': '<b>Range</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>',
            'heatmap': '<b>X</b>: %{x}<br><b>Y</b>: %{y}<br><b>Value</b>: %{z:.3f}<extra></extra>',
            'scatter': '<b>%{text}</b><br>X: %{x:,.2f}<br>Y: %{y:,.2f}<extra></extra>'
        };
        return templates[chartType] || templates['line'];
    }

    addInteractivityToCharts() {
        Object.keys(this.chartInstances).forEach(chartId => {
            const chartElement = document.getElementById(chartId);
            if (chartElement) {
                chartElement.on('plotly_click', (data) => {
                    this.handleChartClick(chartId, data);
                });
            }
        });
    }

    handleChartClick(chartId, data) {
        if (data.points && data.points.length > 0) {
            const point = data.points[0];
            const insight = `Clicked on ${point.data.name || 'data point'}: ${point.x} with value ${point.y}`;
            this.addChatMessage(`Chart interaction: ${insight}`, 'system');
        }
    }

    async exportChartImage(chartId, title) {
        try {
            const chartElement = document.getElementById(chartId);
            if (chartElement && window.Plotly) {
                await Plotly.downloadImage(chartElement, {
                    format: 'png',
                    width: 1200,
                    height: 800,
                    filename: `${title.replace(/\s+/g, '_')}_chart`
                });
                this.showSuccess('Chart exported successfully!');
            }
        } catch (error) {
            console.error('❌ Chart export error:', error);
            this.showError('Failed to export chart: ' + error.message);
        }
    }

    openChartFullscreen(chartId) {
        const chartData = this.chartInstances[chartId];
        if (!chartData) return;

        const modal = document.getElementById('chart-modal');
        const modalContent = document.getElementById('modal-chart-content');
        const modalTitle = document.getElementById('modal-chart-title');
        
        if (!modal || !modalContent || !modalTitle) return;

        modalTitle.textContent = `${chartId.replace(/-/g, ' ').toUpperCase()} - Fullscreen View`;
        modalContent.innerHTML = `<div id="fullscreen-chart" style="width: 100%; height: 100%;"></div>`;
        
        modal.classList.remove('hidden');

        // Render chart in fullscreen
        setTimeout(() => {
            if (window.Plotly) {
                Plotly.newPlot('fullscreen-chart', chartData.data, {
                    ...chartData.layout,
                    height: undefined,
                    width: undefined
                }, chartData.config);
            }
        }, 100);

        // Close modal event
        const closeBtn = document.getElementById('close-modal');
        if (closeBtn) {
            closeBtn.onclick = () => modal.classList.add('hidden');
        }

        modal.onclick = (e) => {
            if (e.target === modal) {
                modal.classList.add('hidden');
            }
        };
    }

    async loadInsights() {
        try {
            console.log('🧠 Loading insights...');
            const response = await fetch(`${this.apiBase}/generate_insights/${this.sessionId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            console.log('💡 Insights received:', Object.keys(result.insights || {}));
            
            this.displayInsights(result.insights || {});
        } catch (error) {
            console.error('❌ Error loading insights:', error);
            this.showInsightsError('Failed to load insights: ' + error.message);
        }
    }

    displayInsights(insights) {
        const insightsDiv = document.getElementById('ai-insights');
        if (!insightsDiv) {
            console.warn('⚠️ AI insights container not found');
            return;
        }

        console.log('💡 Displaying insights:', Object.keys(insights));
        insightsDiv.innerHTML = '';

        // Create tabbed interface for insights
        const tabs = [
            { key: 'executive_summary', title: '📋 Executive Summary', icon: '📋' },
            { key: 'business_insights', title: '💡 Business Insights', icon: '💡' },
            { key: 'seasonal_trends', title: '📅 Seasonal Trends', icon: '📅' },
            { key: 'product_performance', title: '🏆 Product Performance', icon: '🏆' },
            { key: 'regional_analysis', title: '🌍 Regional Analysis', icon: '🌍' },
            { key: 'risk_factors', title: '⚠️ Risk Factors', icon: '⚠️' },
            { key: 'growth_opportunities', title: '🚀 Growth Opportunities', icon: '🚀' },
            { key: 'specific_predictions', title: '🔮 Specific Predictions', icon: '🔮' }
        ];

        // Filter tabs that have content
        const availableTabs = tabs.filter(tab => 
            insights[tab.key] && Array.isArray(insights[tab.key]) && insights[tab.key].length > 0
        );

        if (availableTabs.length === 0) {
            insightsDiv.innerHTML = `
                <div class="text-center py-8">
                    <div class="text-4xl mb-4">🧠</div>
                    <div class="text-gray-400">AI insights are being generated...</div>
                    <div class="text-sm text-gray-500 mt-2">This may take a moment for complex datasets</div>
                </div>
            `;
            return;
        }

        // Create tab navigation
        const tabNav = document.createElement('div');
        tabNav.className = 'flex flex-wrap gap-1 mb-4 border-b border-gray-600';
        
        availableTabs.forEach((tab, index) => {
            const tabButton = document.createElement('button');
            tabButton.className = `px-3 py-2 text-xs rounded-t-lg transition-colors ${
                index === 0 ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`;
            tabButton.innerHTML = `${tab.icon} ${tab.title.split(' ')[1] || tab.title}`;
            tabButton.addEventListener('click', () => this.switchInsightTab(tab.key, tabButton));
            tabNav.appendChild(tabButton);
        });

        insightsDiv.appendChild(tabNav);

        // Create tab content
        const tabContent = document.createElement('div');
        tabContent.id = 'insight-tab-content';
        insightsDiv.appendChild(tabContent);

        // Show first available tab
        if (availableTabs.length > 0) {
            this.switchInsightTab(availableTabs[0].key);
        }

        // Store insights for tab switching
        this.currentInsights = insights;
    }

    switchInsightTab(tabKey, clickedButton = null) {
        const tabContent = document.getElementById('insight-tab-content');
        if (!tabContent || !this.currentInsights) return;

        const insights = this.currentInsights[tabKey] || [];

        // Update active tab button
        if (clickedButton) {
            document.querySelectorAll('#ai-insights button').forEach(btn => {
                btn.className = btn.className.replace('bg-blue-600 text-white', 'bg-gray-700 text-gray-300 hover:bg-gray-600');
            });
            clickedButton.className = clickedButton.className.replace('bg-gray-700 text-gray-300 hover:bg-gray-600', 'bg-blue-600 text-white');
        }

        // Display insights for selected tab
        tabContent.innerHTML = `
            <div class="space-y-3 max-h-64 overflow-y-auto">
                ${insights.map((insight, index) => `
                    <div class="bg-gray-700 p-3 rounded-lg border-l-4 border-blue-500 animate-fadeIn" 
                         style="animation-delay: ${index * 0.1}s">
                        <p class="text-sm text-gray-300 leading-relaxed">${insight}</p>
                    </div>
                `).join('')}
            </div>
        `;
    }

    showInsightsError(message) {
        const insightsDiv = document.getElementById('ai-insights');
        if (!insightsDiv) return;

        insightsDiv.innerHTML = `
            <div class="text-center py-8">
                <div class="text-4xl mb-4">⚠️</div>
                <div class="text-gray-400">Insights Unavailable</div>
                <div class="text-sm text-gray-500 mt-2">${message}</div>
            </div>
        `;
    }

    async loadPredictions() {
        try {
            console.log('🔮 Loading predictions...');
            const response = await fetch(`${this.apiBase}/predict_trends/${this.sessionId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const predictions = await response.json();
            console.log('📈 Predictions received:', Object.keys(predictions));
            
            this.displayPredictions(predictions);
        } catch (error) {
            console.error('❌ Error loading predictions:', error);
            this.showPredictionsError('Failed to load predictions: ' + error.message);
        }
    }

    displayPredictions(predictions) {
        const predictionsDiv = document.getElementById('predictions');
        if (!predictionsDiv) {
            console.warn('⚠️ Predictions container not found');
            return;
        }

        predictionsDiv.innerHTML = '';

        if (!predictions || Object.keys(predictions).length === 0) {
            predictionsDiv.innerHTML = `
                <div class="col-span-full text-center py-8">
                    <div class="text-4xl mb-4">🔮</div>
                    <div class="text-gray-400">No predictions available</div>
                    <div class="text-sm text-gray-500 mt-2">Insufficient data for trend analysis</div>
                </div>
            `;
            return;
        }

        Object.entries(predictions).forEach(([column, prediction]) => {
            if (!prediction || typeof prediction !== 'object') return;

            const predictionCard = document.createElement('div');
            predictionCard.className = 'bg-gray-700 p-5 rounded-lg border border-gray-600 hover:border-blue-500 transition-colors';
            
            const trend = prediction.trend || 'unknown';
            const trendIcon = trend === 'increasing' ? '📈' : trend === 'decreasing' ? '📉' : '📊';
            const trendColor = trend === 'increasing' ? 'text-green-400' : trend === 'decreasing' ? 'text-red-400' : 'text-gray-400';
            const slope = prediction.slope || 0;
            const confidence = prediction.confidence_score || 0;
            const confidenceColor = confidence > 0.7 ? 'text-green-400' : confidence > 0.4 ? 'text-yellow-400' : 'text-red-400';

            predictionCard.innerHTML = `
                <div class="flex justify-between items-start mb-3">
                    <h5 class="font-semibold text-white text-lg">${column.replace(/_/g, ' ')}</h5>
                    <div class="text-2xl">${trendIcon}</div>
                </div>
                
                <div class="space-y-2 mb-4">
                    <div class="flex justify-between">
                        <span class="text-gray-400 text-sm">Trend:</span>
                        <span class="${trendColor} font-medium">${trend}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400 text-sm">Slope:</span>
                        <span class="text-gray-300 font-mono">${slope.toFixed(4)}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400 text-sm">Confidence:</span>
                        <span class="${confidenceColor} font-medium">${(confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
                
                <div class="mb-3">
                    <div class="text-xs text-gray-400 mb-2">Next 5 Predictions:</div>
                    <div class="grid grid-cols-5 gap-1">
                        ${(prediction.predictions || []).slice(0, 5).map((pred, index) => `
                            <div class="text-center">
                                <div class="text-xs bg-gray-600 px-2 py-1 rounded text-white font-mono">
                                    ${Number(pred).toFixed(0)}
                                </div>
                                <div class="text-xs text-gray-500 mt-1">+${index + 1}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="w-full bg-gray-600 rounded-full h-2">
                    <div class="bg-gradient-to-r ${trend === 'increasing' ? 'from-green-500 to-blue-500' : 'from-red-500 to-orange-500'} 
                              h-2 rounded-full transition-all duration-1000" 
                         style="width: ${Math.min(confidence * 100, 100)}%"></div>
                </div>
                <div class="text-xs text-gray-400 mt-1">Model: ${prediction.model_used || 'Linear Regression'}</div>
            `;

            predictionsDiv.appendChild(predictionCard);
        });
    }

    showPredictionsError(message) {
        const predictionsDiv = document.getElementById('predictions');
        if (!predictionsDiv) return;

        predictionsDiv.innerHTML = `
            <div class="col-span-full text-center py-8">
                <div class="text-4xl mb-4">⚠️</div>
                <div class="text-gray-400">Predictions Unavailable</div>
                <div class="text-sm text-gray-500 mt-2">${message}</div>
            </div>
        `;
    }

    async loadAnomalies() {
        try {
            console.log('🔍 Loading anomalies...');
            const response = await fetch(`${this.apiBase}/detect_anomalies/${this.sessionId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const anomalies = await response.json();
            console.log('⚠️ Anomalies received:', Object.keys(anomalies));
            
            this.displayAnomalies(anomalies);
        } catch (error) {
            console.error('❌ Error loading anomalies:', error);
            this.showAnomaliesError('Failed to load anomalies: ' + error.message);
        }
    }

    displayAnomalies(anomalies) {
        const anomaliesDiv = document.getElementById('anomalies');
        if (!anomaliesDiv) {
            console.warn('⚠️ Anomalies container not found');
            return;
        }

        anomaliesDiv.innerHTML = '';

        const hasAnomalies = Object.values(anomalies).some(anomaly => 
            anomaly && typeof anomaly === 'object' && (anomaly.count || 0) > 0
        );

        if (!hasAnomalies) {
            anomaliesDiv.innerHTML = `
                <div class="text-center py-8">
                    <div class="text-4xl mb-4">✅</div>
                    <p class="text-green-400 font-semibold">No significant anomalies detected</p>
                    <p class="text-gray-400 text-sm mt-2">Your data appears to be within normal ranges</p>
                </div>
            `;
            return;
        }

        Object.entries(anomalies).forEach(([column, anomaly]) => {
            if (!anomaly || typeof anomaly !== 'object' || (anomaly.count || 0) === 0) return;

            const count = anomaly.count || 0;
            const percentage = anomaly.percentage || 0;
            const values = anomaly.values || [];
            
            const severityLevel = this.getAnomalySeverity(count, percentage);
            const severityColor = this.getSeverityColor(severityLevel);
            
            const anomalyDiv = document.createElement('div');
            anomalyDiv.className = `mb-4 p-4 bg-red-900 bg-opacity-20 border border-red-700 rounded-lg hover:bg-opacity-30 transition-colors`;
            anomalyDiv.innerHTML = `
                <div class="flex justify-between items-start mb-3">
                    <h5 class="font-semibold text-red-400 text-lg">⚠️ ${column.replace(/_/g, ' ')}</h5>
                    <span class="px-2 py-1 text-xs rounded ${severityColor} font-semibold">
                        ${severityLevel.toUpperCase()}
                    </span>
                </div>
                
                <div class="grid grid-cols-2 gap-4 mb-3">
                    <div>
                        <div class="text-sm text-gray-300">Anomalies Found:</div>
                        <div class="text-xl font-bold text-red-400">${count}</div>
                    </div>
                    <div>
                        <div class="text-sm text-gray-300">Percentage:</div>
                        <div class="text-xl font-bold text-orange-400">${percentage.toFixed(1)}%</div>
                    </div>
                </div>
                
                <div class="text-sm">
                    <div class="text-gray-400 mb-2">Sample Anomalous Values:</div>
                    <div class="flex flex-wrap gap-2">
                        ${values.slice(0, 5).map(value => `
                            <span class="px-2 py-1 bg-red-800 bg-opacity-50 rounded text-red-300 font-mono text-xs">
                                ${typeof value === 'number' ? value.toFixed(2) : value}
                            </span>
                        `).join('')}
                        ${values.length > 5 ? `<span class="text-gray-500 text-xs">+${values.length - 5} more</span>` : ''}
                    </div>
                </div>
            `;
            anomaliesDiv.appendChild(anomalyDiv);
        });
    }

    getAnomalySeverity(count, percentage) {
        if (percentage > 10 || count > 50) return 'critical';
        if (percentage > 5 || count > 20) return 'high';
        if (percentage > 2 || count > 5) return 'medium';
        return 'low';
    }

    getSeverityColor(severity) {
        const colors = {
            'critical': 'bg-red-600 text-white',
            'high': 'bg-orange-600 text-white',
            'medium': 'bg-yellow-600 text-black',
            'low': 'bg-blue-600 text-white'
        };
        return colors[severity] || colors['low'];
    }

    showAnomaliesError(message) {
        const anomaliesDiv = document.getElementById('anomalies');
        if (!anomaliesDiv) return;

        anomaliesDiv.innerHTML = `
            <div class="text-center py-8">
                <div class="text-4xl mb-4">⚠️</div>
                <div class="text-gray-400">Anomaly Detection Unavailable</div>
                <div class="text-sm text-gray-500 mt-2">${message}</div>
            </div>
        `;
    }

    async loadDataSummary() {
        try {
            console.log('📋 Loading data summary...');
            const response = await fetch(`${this.apiBase}/get_data_summary/${this.sessionId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const summary = await response.json();
            console.log('📊 Data summary received');
            
            this.displayDataSummary(summary);
        } catch (error) {
            console.error('❌ Error loading data summary:', error);
            // Don't show error for data summary as it's optional
        }
    }

    displayDataSummary(summary) {
        // This can be used to display additional data summary information
        // Implementation depends on where you want to show this information
        console.log('📊 Data summary available:', summary);
    }

    async sendChatMessage() {
        const input = document.getElementById('chat-input');
        if (!input) return;
        
        const message = input.value.trim();
        
        if (!message) {
            this.showError('Please enter a message');
            return;
        }

        if (!this.sessionId) {
            this.showError('No dataset loaded. Please upload data first.');
            return;
        }

        const chatMessages = document.getElementById('chat-messages');
        
        // Add user message
        this.addChatMessage(message, 'user');
        input.value = '';

        // Add loading message
        const loadingMessage = this.addChatMessage('🤔 Analyzing your data...', 'ai');

        try {
            const response = await fetch(`${this.apiBase}/chat_query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    query: message
                })
            });

            const result = await response.json();

            // Remove loading message
            if (loadingMessage && loadingMessage.parentElement) {
                loadingMessage.remove();
            }

            if (response.ok) {
                this.addChatMessage(result.response, 'ai');
            } else {
                this.addChatMessage('Sorry, I encountered an error processing your query. Please try rephrasing your question.', 'ai');
            }
        } catch (error) {
            console.error('❌ Chat error:', error);
            if (loadingMessage && loadingMessage.parentElement) {
                loadingMessage.remove();
            }
            this.addChatMessage('Network error. Please check your connection and try again.', 'ai');
        }

        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    addChatMessage(message, sender) {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return null;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;
        
        const timestamp = new Date().toLocaleTimeString('en-US', { 
            hour12: true, 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        messageDiv.innerHTML = `
            <div class="flex items-center justify-between mb-1">
                <div class="text-xs text-gray-400 font-semibold">
                    ${sender === 'user' ? '👤 You' : sender === 'ai' ? '🤖 AI Assistant' : '🔧 System'}
                </div>
                <div class="text-xs text-gray-500">${timestamp}</div>
            </div>
            <div class="text-sm leading-relaxed">${message}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageDiv;
    }

    async downloadCleanedData(format) {
        if (!this.sessionId) {
            this.showError('No dataset loaded');
            return;
        }

        this.showLoading(`Preparing ${format.toUpperCase()} download...`);

        try {
            const response = await fetch(`${this.apiBase}/download_cleaned_data/${this.sessionId}?format=${format}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `cleaned_data_${new Date().toISOString().slice(0, 10)}.${format}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showSuccess(`Cleaned dataset downloaded as ${format.toUpperCase()} successfully!`);
            } else {
                const error = await response.json();
                this.showError(`Failed to download ${format.toUpperCase()} file: ${error.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('❌ Download error:', error);
            this.showError('Network error: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async exportPDF() {
        if (!this.sessionId) {
            this.showError('No dataset loaded');
            return;
        }

        this.showLoading('Generating comprehensive PDF report...');

        try {
            const response = await fetch(`${this.apiBase}/export_pdf/${this.sessionId}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `DataVision_AI_Professional_Report_${new Date().toISOString().slice(0, 10)}.pdf`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showSuccess('Professional PDF report downloaded successfully!');
            } else {
                const error = await response.json();
                this.showError(`Failed to generate PDF report: ${error.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('❌ PDF export error:', error);
            this.showError('Network error: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async refreshDashboard() {
        if (!this.sessionId) {
            this.showError('No dataset loaded to refresh');
            return;
        }

        this.showLoading('Refreshing dashboard...');
        
        try {
            await this.loadDashboard();
            this.updateLastUpdatedTime();
            this.showSuccess('Dashboard refreshed successfully!');
        } catch (error) {
            console.error('❌ Refresh error:', error);
            this.showError('Failed to refresh dashboard: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    updateLastUpdatedTime() {
        const lastUpdated = document.getElementById('last-updated');
        if (lastUpdated) {
            lastUpdated.textContent = new Date().toLocaleTimeString();
        }
    }

    showKeyboardShortcuts() {
        const shortcuts = [
            'Ctrl+E - Export PDF Report',
            'Ctrl+R - Refresh Dashboard',
            'Ctrl+D - Download CSV',
            'Ctrl+H - Show Shortcuts',
            'Ctrl+/ - Focus Chat',
            'Escape - Close Modals'
        ];

        const toast = this.createToast('⌨️ Keyboard Shortcuts', shortcuts.join('<br>'), 'info', 8000);
        const container = document.getElementById('toast-container');
        if (container) {
            container.appendChild(toast);
        }
    }

    formatNumber(num) {
        if (num == null || isNaN(num)) return '0';
        
        const absNum = Math.abs(num);
        if (absNum >= 1e9) return (num / 1e9).toFixed(1) + 'B';
        if (absNum >= 1e6) return (num / 1e6).toFixed(1) + 'M';
        if (absNum >= 1e3) return (num / 1e3).toFixed(1) + 'K';
        return Number(num).toFixed(2);
    }

    showLoading(text = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        const loadingText = document.getElementById('loading-text');
        if (overlay && loadingText) {
            loadingText.textContent = text;
            overlay.classList.remove('hidden');
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }

    showSuccess(message) {
        this.showAlert(message, 'success');
    }

    showError(message) {
        this.showAlert(message, 'error');
    }

    showAlert(message, type) {
        const toast = this.createToast(
            type === 'success' ? '✅ Success' : '❌ Error',
            message,
            type,
            type === 'error' ? 8000 : 5000
        );
        
        const container = document.getElementById('toast-container');
        if (container) {
            container.appendChild(toast);
        }
    }

    createToast(title, message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `transform transition-all duration-300 translate-x-full`;
        
        const colors = {
            success: 'bg-green-800 border-green-600 text-green-100',
            error: 'bg-red-800 border-red-600 text-red-100',
            warning: 'bg-yellow-800 border-yellow-600 text-yellow-100',
            info: 'bg-blue-800 border-blue-600 text-blue-100'
        };
        
        toast.innerHTML = `
            <div class="max-w-sm p-4 rounded-lg border ${colors[type]} shadow-lg">
                <div class="flex items-start">
                    <div class="flex-1">
                        <h4 class="font-semibold">${title}</h4>
                        <p class="text-sm mt-1 opacity-90">${message}</p>
                    </div>
                    <button class="ml-3 text-xl opacity-70 hover:opacity-100" onclick="this.parentElement.parentElement.parentElement.remove()">
                        ×
                    </button>
                </div>
            </div>
        `;
        
        // Animate in
        setTimeout(() => toast.classList.remove('translate-x-full'), 100);
        
        // Auto remove
        if (duration > 0) {
            setTimeout(() => {
                toast.classList.add('translate-x-full');
                setTimeout(() => {
                    if (toast.parentElement) toast.remove();
                }, 300);
            }, duration);
        }
        
        return toast;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DOM loaded, initializing DataVision-AI Dashboard...');
    new DataVisionDashboard();
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fadeIn {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    .chart-container:hover {
        transform: translateY(-2px);
        transition: transform 0.3s ease;
    }
    
    .kpi-card {
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
    }
    
    .chat-message {
        animation: fadeIn 0.3s ease-out;
    }
    
    .dragover {
        border-color: #3b82f6 !important;
        background-color: rgba(59, 130, 246, 0.1) !important;
    }
`;
document.head.appendChild(style);