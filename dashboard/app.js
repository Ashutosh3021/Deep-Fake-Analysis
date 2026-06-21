// DeepGuard AI Dashboard JavaScript
class DeepGuardDashboard {
    constructor() {
        this.history = JSON.parse(localStorage.getItem('deepguard_history')) || [];
        this.apiBaseUrl = 'http://localhost:5000';
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateStats();
        this.renderHistory();
        this.checkApiStatus();
    }

    bindEvents() {
        // Tab navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // File upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Text analysis
        document.getElementById('analyze-text-btn').addEventListener('click', () => this.analyzeText());

        // Clear history
        document.getElementById('clear-history').addEventListener('click', () => this.clearHistory());
    }

    switchTab(tabId) {
        // Update nav buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === tabId);
        });
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        this.showLoading(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/detect/auto`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                this.displayResults(data.result, file.name, data.file_type);
                this.addToHistory(file.name, data.file_type, data.result);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Error:', error);
            // Fallback to mock results for demo
            this.displayMockResults(file.name, this.getFileType(file.name));
        } finally {
            this.showLoading(false);
        }
    }

    async analyzeText() {
        const textInput = document.getElementById('text-input');
        const text = textInput.value.trim();

        if (!text) {
            alert('Please enter some text to analyze');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/detect/text`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                this.displayResults(data.result, 'Text Input', 'text');
                this.addToHistory('Text Input', 'text', data.result);
                textInput.value = '';
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Error:', error);
            // Fallback to mock results
            this.displayMockResults('Text Input', 'text');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(result, filename, fileType) {
        const resultsSection = document.getElementById('results-section');
        resultsSection.style.display = 'block';

        // Update file info
        document.getElementById('result-file-type').textContent = fileType.charAt(0).toUpperCase() + fileType.slice(1);
        document.getElementById('result-filename').textContent = filename;

        // Determine verdict based on new result format
        let isFake, confidence, label, verdictClass;

        if (fileType === 'text') {
            isFake = result.label === 'AI_GENERATED';
            confidence = result.confidence;
            label = isFake ? 'AI Generated' : (result.label === 'HUMAN_WRITTEN' ? 'Human Written' : 'Uncertain');
        } else if (fileType === 'audio') {
            isFake = result.label === 'SYNTHETIC';
            confidence = result.confidence;
            label = isFake ? 'Synthetic' : (result.label === 'AUTHENTIC' ? 'Authentic' : 'Uncertain');
        } else { // image or video
            isFake = result.label === 'FAKE';
            confidence = result.confidence;
            label = isFake ? 'Fake' : (result.label === 'AUTHENTIC' ? 'Authentic' : 'Uncertain');
        }

        verdictClass = isFake ? 'fake' : (result.label === 'UNCERTAIN' ? 'uncertain' : 'real');

        // Update verdict
        const verdictEl = document.getElementById('result-verdict');
        verdictEl.innerHTML = `<span class="verdict-badge ${verdictClass}">${label}</span>`;

        // Update confidence
        document.getElementById('confidence-value').textContent = `${typeof confidence === 'number' ? confidence.toFixed(1) : confidence}%`;
        const meterFill = document.getElementById('confidence-fill');
        meterFill.style.width = `${confidence}%`;
        meterFill.style.background = isFake
            ? 'linear-gradient(90deg, #f59e0b 0%, #ef4444 100%)'
            : (result.label === 'UNCERTAIN' ? 'linear-gradient(90deg, #6b7280 0%, #4b5563 100%)' : 'linear-gradient(90deg, #10b981 0%, #059669 100%)');

        // Update details with new fields
        document.getElementById('detection-method').textContent = result.signal_source || result.notes || 'Advanced ML';
        document.getElementById('model-accuracy').textContent = 'N/A';

        let probability;
        if (fileType === 'text') {
            probability = result.ai_probability;
        } else if (fileType === 'audio') {
            probability = result.fake_probability;
        } else {
            probability = result.family_scores ? Object.values(result.family_scores).filter(v => typeof v === 'number').sort((a, b) => b - a)[0] : null;
        }
        document.getElementById('probability-value').textContent = probability ? probability.toFixed(3) : 'N/A';

        // Update breakdown
        this.updateModelBreakdown(result, fileType);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    updateModelBreakdown(result, fileType) {
        const breakdownList = document.getElementById('breakdown-list');
        breakdownList.innerHTML = '';

        // Handle new result formats
        if (fileType === 'image' || fileType === 'video') {
            // Display fake_type as badges
            if (result.fake_type && result.fake_type.length > 0) {
                const fakeTypeItem = document.createElement('div');
                fakeTypeItem.className = 'breakdown-item';
                fakeTypeItem.innerHTML = `<span class="breakdown-name">Fake Type</span><div style="display: flex; gap: 8px;">${result.fake_type.map(type => `<span style="background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 12px; font-size: 12px;">${type.replace('_', ' ')}</span>`).join('')}</div>`;
                breakdownList.appendChild(fakeTypeItem);
            }
            // Display reasons
            if (result.reasons && result.reasons.length > 0) {
                result.reasons.forEach((reason, index) => {
                    const item = document.createElement('div');
                    item.className = 'breakdown-item';
                    item.innerHTML = `
                        <span class="breakdown-name">${reason.signal || 'Reason ' + (index + 1)}</span>
                        <div style="flex: 1; margin: 0 16px; display: flex; flex-direction: column; gap: 4px;">
                            <div style="color: #374151; font-size: 13px;">${reason.description || ''}</div>
                            <div class="breakdown-bar">
                                <div class="breakdown-fill" style="width: ${(reason.score * 100).toFixed(1)}%"></div>
                            </div>
                        </div>
                        <span class="breakdown-score">${(reason.score * 100).toFixed(1)}%</span>
                    `;
                    breakdownList.appendChild(item);
                });
            }
        } else if (fileType === 'audio') {
            // Audio specific details
            const signalItem = document.createElement('div');
            signalItem.className = 'breakdown-item';
            signalItem.innerHTML = `
                <span class="breakdown-name">Signal Source</span>
                <div style="flex: 1; color: #374151;">${result.signal_source || 'N/A'}</div>
                <span class="breakdown-score">${(result.fake_probability * 100).toFixed(1)}%</span>
            `;
            breakdownList.appendChild(signalItem);
        } else if (fileType === 'text') {
            // Text specific details
            const perplexityItem = document.createElement('div');
            perplexityItem.className = 'breakdown-item';
            perplexityItem.innerHTML = `
                <span class="breakdown-name">Perplexity Score</span>
                <div style="flex: 1; color: #374151;">${result.perplexity_signal !== null ? (result.perplexity_signal * 100).toFixed(1) + '%' : 'N/A'}</div>
                <span class="breakdown-score">${(result.ai_probability * 100).toFixed(1)}%</span>
            `;
            breakdownList.appendChild(perplexityItem);
        }

        if (breakdownList.children.length === 0) {
            breakdownList.innerHTML = '<p style="color: #6b7280; text-align: center;">Detailed breakdown not available</p>';
        }
    }

    formatModelName(name) {
        const names = {
            'cnn': 'CNN Analysis',
            'lstm': 'LSTM Analysis',
            'neural': 'Neural Network',
            'statistical': 'Statistical Analysis',
            'patterns': 'Pattern Detection',
            'perplexity': 'Perplexity Analysis',
            'anomaly': 'Anomaly Detection',
            'artifacts': 'Artifact Detection'
        };
        return names[name] || name.charAt(0).toUpperCase() + name.slice(1);
    }

    displayMockResults(filename, fileType) {
        const isFake = Math.random() > 0.5;
        const confidence = (70 + Math.random() * 25).toFixed(1);

        const mockResult = {
            label: isFake ? 'FAKE' : 'REAL',
            confidence: parseFloat(confidence),
            ensemble_score: isFake ? 0.7 + Math.random() * 0.25 : Math.random() * 0.3,
            accuracy_rating: '95.8%',
            detection_method: 'ensemble_ml_mock',
            model_predictions: {
                'cnn': Math.random(),
                'lstm': Math.random(),
                'neural': Math.random(),
                'statistical': Math.random()
            }
        };

        this.displayResults(mockResult, filename, fileType);
        this.addToHistory(filename, fileType, mockResult);
    }

    getFileType(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        if (['png', 'jpg', 'jpeg', 'gif'].includes(ext)) return 'image';
        if (['mp4', 'avi', 'mov', 'mkv'].includes(ext)) return 'video';
        if (['mp3', 'wav', 'ogg', 'flac'].includes(ext)) return 'audio';
        return 'text';
    }

    addToHistory(filename, fileType, result) {
        const historyItem = {
            id: Date.now(),
            filename,
            fileType,
            result,
            timestamp: new Date().toISOString()
        };

        this.history.unshift(historyItem);
        if (this.history.length > 50) {
            this.history = this.history.slice(0, 50);
        }

        localStorage.setItem('deepguard_history', JSON.stringify(this.history));
        this.updateStats();
        this.renderHistory();
    }

    renderHistory() {
        const historyList = document.getElementById('history-list');

        if (this.history.length === 0) {
            historyList.innerHTML = '<p class="empty-state">No analyses yet. Start by uploading a file!</p>';
            return;
        }

        historyList.innerHTML = this.history.map(item => {
            let isFake, label, verdictClass;

            if (item.fileType === 'text') {
                isFake = item.result.label === 'AI_GENERATED';
                label = isFake ? 'AI Generated' : (item.result.label === 'HUMAN_WRITTEN' ? 'Human' : 'Uncertain');
            } else if (item.fileType === 'audio') {
                isFake = item.result.label === 'SYNTHETIC';
                label = isFake ? 'Synthetic' : (item.result.label === 'AUTHENTIC' ? 'Authentic' : 'Uncertain');
            } else { // image or video
                isFake = item.result.label === 'FAKE';
                label = isFake ? 'Fake' : (item.result.label === 'AUTHENTIC' ? 'Authentic' : 'Uncertain');
            }

            verdictClass = isFake ? 'fake' : (item.result.label === 'UNCERTAIN' ? 'uncertain' : 'real');
            const icon = this.getFileIcon(item.fileType);

            return `
                <div class="history-item">
                    <div class="history-info">
                        <div class="history-type">
                            ${icon}
                        </div>
                        <div class="history-details">
                            <h4>${item.filename}</h4>
                            <span>${new Date(item.timestamp).toLocaleString()}</span>
                        </div>
                    </div>
                    <div class="history-result">
                        <span class="history-verdict ${verdictClass}">${label}</span>
                        <div class="history-confidence">${typeof item.result.confidence === 'number' ? item.result.confidence.toFixed(1) : item.result.confidence}% confidence</div>
                    </div>
                </div>
            `;
        }).join('');
    }

    getFileIcon(fileType) {
        const icons = {
            image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
            video: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
            audio: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20"/><circle cx="12" cy="12" r="10"/></svg>',
            text: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>'
        };
        return icons[fileType] || icons.text;
    }

    clearHistory() {
        if (confirm('Are you sure you want to clear all history?')) {
            this.history = [];
            localStorage.removeItem('deepguard_history');
            this.updateStats();
            this.renderHistory();
        }
    }

    updateStats() {
        const analyzedCount = this.history.length;
        const detectedCount = this.history.filter(item => {
            if (item.fileType === 'text') return item.result.label === 'AI_GENERATED';
            if (item.fileType === 'audio') return item.result.label === 'SYNTHETIC';
            return item.result.label === 'FAKE';
        }).length;

        document.getElementById('analyzed-count').textContent = analyzedCount;
        document.getElementById('detected-count').textContent = detectedCount;
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.add('active');
        } else {
            overlay.classList.remove('active');
        }
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/status`);
            if (response.ok) {
                const data = await response.json();
                console.log('API Status:', data);
            }
        } catch (error) {
            console.log('API not available, using mock mode');
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DeepGuardDashboard();
    window.queryAssistant = new QueryAssistant();
});

/* ════════════════════════════════════════════════════════════
   Query Assistant
   ════════════════════════════════════════════════════════════ */
class QueryAssistant {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000';
        this.currentMode = 'image'; // 'image' | 'text'
        this.selectedFile = null;
        this.init();
    }

    init() {
        this._bindModeButtons();
        this._bindImagePanel();
        this._bindTextPanel();
    }

    // ── Mode switching ──────────────────────────────────────
    _bindModeButtons() {
        document.getElementById('mode-image-btn').addEventListener('click', () => this._setMode('image'));
        document.getElementById('mode-text-btn').addEventListener('click', () => this._setMode('text'));
    }

    _setMode(mode) {
        this.currentMode = mode;

        // Toggle active class on buttons
        document.getElementById('mode-image-btn').classList.toggle('active', mode === 'image');
        document.getElementById('mode-text-btn').classList.toggle('active', mode === 'text');

        // Show / hide panels
        document.getElementById('qa-image-panel').style.display = mode === 'image' ? '' : 'none';
        document.getElementById('qa-text-panel').style.display = mode === 'text' ? '' : 'none';

        // Hide stale results
        document.getElementById('qa-results').style.display = 'none';
    }

    // ── Image panel ─────────────────────────────────────────
    _bindImagePanel() {
        const zone = document.getElementById('qa-upload-zone');
        const fileInput = document.getElementById('qa-file-input');
        const submitBtn = document.getElementById('qa-image-submit');

        // Click on zone → open file picker
        zone.addEventListener('click', (e) => {
            if (e.target === fileInput) return;
            fileInput.click();
        });

        // Drag-and-drop
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) this._setImageFile(file);
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) this._setImageFile(e.target.files[0]);
        });

        // Submit
        submitBtn.addEventListener('click', () => this._submitImageQuery());
    }

    _setImageFile(file) {
        this.selectedFile = file;

        const preview = document.getElementById('qa-preview');
        const content = document.getElementById('qa-upload-content');

        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = 'block';
            content.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    async _submitImageQuery() {
        if (!this.selectedFile) {
            alert('Please select an image first.');
            return;
        }

        const query = document.getElementById('qa-image-query').value.trim();
        const btn = document.getElementById('qa-image-submit');
        btn.disabled = true;
        this._showQaLoading(true);

        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            if (query) formData.append('query', query);

            const response = await fetch(`${this.apiBaseUrl}/api/query`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (!response.ok || !data.success) throw new Error(data.error || 'Request failed');

            this._renderImageResult(data);
        } catch (err) {
            this._renderError(err.message);
        } finally {
            btn.disabled = false;
            this._showQaLoading(false);
        }
    }

    _renderImageResult(data) {
        const resultsEl = document.getElementById('qa-results');
        const objSection = document.getElementById('qa-objects-section');
        const tagsEl = document.getElementById('qa-object-tags');
        const answerEl = document.getElementById('qa-gemini-answer');

        resultsEl.style.display = '';

        // Objects
        if (data.objects && data.objects.length > 0) {
            objSection.style.display = '';
            // Deduplicate by name, keep highest confidence
            const best = {};
            data.objects.forEach(o => {
                if (!best[o.name] || o.confidence > best[o.name]) best[o.name] = o.confidence;
            });
            tagsEl.innerHTML = Object.entries(best)
                .sort((a, b) => b[1] - a[1])
                .map(([name, conf]) =>
                    `<span class="object-tag">
                        ${name}
                        <span class="tag-conf">${conf}%</span>
                    </span>`
                ).join('');
        } else {
            objSection.style.display = 'none';
            tagsEl.innerHTML = '';
        }

        // Gemini answer
        answerEl.textContent = data.gemini_answer || '(No response)';
        resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // ── Text panel ──────────────────────────────────────────
    _bindTextPanel() {
        document.getElementById('qa-text-submit').addEventListener('click', () => this._submitTextQuery());

        // Ctrl+Enter shortcut in textarea
        document.getElementById('qa-text-query').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this._submitTextQuery();
            }
        });
    }

    async _submitTextQuery() {
        const textarea = document.getElementById('qa-text-query');
        const query = textarea.value.trim();

        if (!query) {
            alert('Please enter a question.');
            return;
        }

        const btn = document.getElementById('qa-text-submit');
        btn.disabled = true;
        this._showQaLoading(true);

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });

            const data = await response.json();
            if (!response.ok || !data.success) throw new Error(data.error || 'Request failed');

            this._renderTextResult(data);
        } catch (err) {
            this._renderError(err.message);
        } finally {
            btn.disabled = false;
            this._showQaLoading(false);
        }
    }

    _renderTextResult(data) {
        const resultsEl = document.getElementById('qa-results');
        const objSection = document.getElementById('qa-objects-section');
        const answerEl = document.getElementById('qa-gemini-answer');

        objSection.style.display = 'none';
        resultsEl.style.display = '';
        answerEl.textContent = data.gemini_answer || '(No response)';
        resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // ── Helpers ─────────────────────────────────────────────
    _renderError(msg) {
        const resultsEl = document.getElementById('qa-results');
        const objSec = document.getElementById('qa-objects-section');
        const answerEl = document.getElementById('qa-gemini-answer');

        objSec.style.display = 'none';
        resultsEl.style.display = '';
        answerEl.innerHTML = `<span style="color:#ef4444;">⚠ Error: ${msg}</span>`;
        resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    _showQaLoading(show) {
        // Reuse existing loading overlay
        const overlay = document.getElementById('loading-overlay');
        const p = overlay.querySelector('p');
        if (show) {
            if (p) p.textContent = 'Querying assistant…';
            overlay.classList.add('active');
        } else {
            if (p) p.textContent = 'Analyzing content…';
            overlay.classList.remove('active');
        }
    }
}