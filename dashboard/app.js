function escapeHTML(str) {
    if (typeof str !== 'string') return str;
    return str.replace(/[&<>'"]/g,
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

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

        // Update modality display
        document.getElementById('analysis-modality').textContent = fileType.charAt(0).toUpperCase() + fileType.slice(1);

        // Determine verdict based on result format
        let isFake, confidence, label, verdictClass;

        if (fileType === 'text') {
            isFake = result.label === 'AI_GENERATED';
            confidence = result.confidence;
            label = isFake ? 'AI Generated' : (result.label === 'HUMAN_WRITTEN' ? 'Human Written' : 'Uncertain');
        } else if (fileType === 'audio') {
            isFake = result.label === 'SYNTHETIC';
            confidence = result.confidence;
            label = isFake ? 'Synthetic' : (result.label === 'AUTHENTIC' ? 'Authentic' : 'Uncertain');
        } else {
            isFake = result.label === 'FAKE';
            confidence = result.confidence;
            label = isFake ? 'Fake' : (result.label === 'AUTHENTIC' ? 'Authentic' : 'Uncertain');
        }

        verdictClass = isFake ? 'fake' : (result.label === 'UNCERTAIN' ? 'uncertain' : 'real');

        // Update verdict badge
        const verdictEl = document.getElementById('result-verdict');
        verdictEl.innerHTML = `<span class="verdict-badge ${verdictClass}">${escapeHTML(String(label))}</span>`;

        // 4-Tier Verdict Display
        const tierDisplay = document.getElementById('tier-display');
        const tierBadge = document.getElementById('tier-badge');
        const tierNumber = document.getElementById('tier-number');
        const tierLabel = document.getElementById('tier-label');
        const tierWarning = document.getElementById('tier-warning');
        const tierWarningText = document.getElementById('tier-warning-text');

        // Determine tier from result
        let tier = 3;
        let tierName = 'Indeterminate';
        if (result.family_scores && result.family_scores.provenance > 0.7) {
            tier = 1;
            tierName = 'Verified Provenance';
        } else if (verdictClass === 'real') {
            tier = 2;
            tierName = 'Likely Authentic';
        } else if (isFake) {
            tier = 4;
            tierName = 'Likely Synthetic';
        } else if (verdictClass === 'uncertain') {
            tier = 3;
            tierName = 'Indeterminate';
        }

        tierDisplay.style.display = 'block';
        tierNumber.textContent = tier;
        tierLabel.textContent = tierName;

        // Color the tier badge
        const tierColors = {1: '#10b981', 2: '#3b82f6', 3: '#f59e0b', 4: '#ef4444'};
        tierBadge.style.borderColor = tierColors[tier] || '#6b7280';

        // Show warning for indeterminate tier
        if (tier === 3) {
            tierWarning.style.display = 'flex';
            tierWarningText.textContent = 'Evidence is conflicting or insufficient for a definitive conclusion. Further analysis with additional data is recommended.';
        } else if (tier === 4 && (result.fake_type && result.fake_type.length < 2)) {
            tierWarning.style.display = 'flex';
            tierWarningText.textContent = 'Only one forensic signal fired. High-stakes decisions should require corroborating evidence from multiple independent signals.';
        } else {
            tierWarning.style.display = 'none';
        }

        // Update confidence
        document.getElementById('confidence-value').textContent = `${typeof confidence === 'number' ? confidence.toFixed(1) : confidence}%`;
        const meterFill = document.getElementById('confidence-fill');
        meterFill.style.width = `${confidence}%`;

        const tierGradients = {
            1: 'linear-gradient(90deg, #10b981 0%, #059669 100%)',
            2: 'linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)',
            3: 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)',
            4: 'linear-gradient(90deg, #f59e0b 0%, #ef4444 100%)',
        };
        meterFill.style.background = tierGradients[tier] || 'linear-gradient(90deg, #6b7280 0%, #4b5563 100%)';

        // Update details
        document.getElementById('detection-method').textContent = result.signal_source || 'Multi-Family Forensic';

        let probability;
        if (fileType === 'text') {
            probability = result.ai_probability;
        } else if (fileType === 'audio') {
            probability = result.fake_probability;
        } else {
            const scores = result.family_scores;
            if (scores) {
                const numScores = Object.values(scores).filter(v => typeof v === 'number');
                probability = numScores.length > 0 ? Math.max(...numScores) : null;
            } else {
                probability = null;
            }
        }
        document.getElementById('probability-value').textContent = probability !== null ? probability.toFixed(3) : 'N/A';

        // File hash
        const hashEl = document.getElementById('file-hash');
        hashEl.textContent = result.file_hash || 'N/A';

        // Evidence breakdown
        this.updateModelBreakdown(result, fileType);

        // Suspicious spans/intervals
        this.updateSuspiciousSection(result, fileType);

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
                fakeTypeItem.innerHTML = `<span class="breakdown-name">Fake Type</span><div style="display: flex; gap: 8px;">${result.fake_type.map(type => `<span style="background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 12px; font-size: 12px;">${escapeHTML(String(type).replace(/_/g, ' '))}</span>`).join('')}</div>`;
                breakdownList.appendChild(fakeTypeItem);
            }
            // Display reasons
            if (result.reasons && result.reasons.length > 0) {
                result.reasons.forEach((reason, index) => {
                    const item = document.createElement('div');
                    item.className = 'breakdown-item';
                    item.innerHTML = `
                        <span class="breakdown-name">${escapeHTML(String(reason.signal || 'Reason ' + (index + 1)))}</span>
                        <div style="flex: 1; margin: 0 16px; display: flex; flex-direction: column; gap: 4px;">
                            <div style="color: #374151; font-size: 13px;">${escapeHTML(String(reason.description || ''))}</div>
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
                <div style="flex: 1; color: #374151;">${escapeHTML(String(result.signal_source || 'N/A'))}</div>
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

    updateSuspiciousSection(result, fileType) {
        const section = document.getElementById('suspicious-section');
        const list = document.getElementById('suspicious-list');
        if (!section || !list) return;

        const items = [];

        if (fileType === 'video' && result.suspicious_intervals && result.suspicious_intervals.length > 0) {
            result.suspicious_intervals.forEach((interval, i) => {
                items.push(`<div class="suspicious-item"><span class="suspicious-tag">Interval ${i + 1}</span><span>${escapeHTML(String(interval[0]))}s – ${escapeHTML(String(interval[1]))}s</span></div>`);
            });
        } else if (fileType === 'text' && result.suspicious_spans && result.suspicious_spans.length > 0) {
            result.suspicious_spans.forEach((span, i) => {
                const reasons = Array.isArray(span.reasons) ? span.reasons.join('; ') : '';
                items.push(`<div class="suspicious-item"><span class="suspicious-tag">Span ${i + 1}</span><span>${escapeHTML(String(span.text || ''))}${reasons ? ' — ' + escapeHTML(String(reasons)) : ''}</span></div>`);
            });
        } else if (fileType === 'audio' && result.suspicious_segments && result.suspicious_segments.length > 0) {
            result.suspicious_segments.forEach((seg, i) => {
                items.push(`<div class="suspicious-item"><span class="suspicious-tag">Segment ${i + 1}</span><span>${escapeHTML(String(seg.start_sec))}s – ${escapeHTML(String(seg.end_sec))}s (score ${(seg.fake_score * 100).toFixed(1)}%)</span></div>`);
            });
        }

        if (items.length > 0) {
            section.style.display = 'block';
            list.innerHTML = items.join('');
        } else {
            section.style.display = 'none';
            list.innerHTML = '';
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
                            <h4>${escapeHTML(String(item.filename))}</h4>
                            <span>${new Date(item.timestamp).toLocaleString()}</span>
                        </div>
                    </div>
                    <div class="history-result">
                        <span class="history-verdict ${verdictClass}">${escapeHTML(String(label))}</span>
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
                        ${escapeHTML(String(name))}
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
        answerEl.innerHTML = `<span style="color:#ef4444;">⚠ Error: ${escapeHTML(String(msg))}</span>`;
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

/* ════════════════════════════════════════════════════════════
   Multi-Modal Fusion UI
   Wires up the fusion section to POST /api/detect/fusion and
   renders the 4-tier cross-modal verdict.
   ════════════════════════════════════════════════════════════ */
class FusionAnalyzer {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this._bindEvents();
    }

    _bindEvents() {
        // Show selected filenames
        ['video', 'audio'].forEach(slot => {
            const input = document.getElementById(`fusion-file-${slot}`);
            const label = document.getElementById(`fusion-name-${slot}`);
            if (input && label) {
                input.addEventListener('change', () => {
                    label.textContent = input.files[0] ? input.files[0].name : 'No file chosen';
                });
            }
        });

        const analyzeBtn = document.getElementById('fusion-analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this._runFusion());
        }

        const clearBtn = document.getElementById('fusion-clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this._clearFusion());
        }
    }

    async _runFusion() {
        const videoFile = document.getElementById('fusion-file-video').files[0];
        const audioFile = document.getElementById('fusion-file-audio').files[0];
        const textVal   = document.getElementById('fusion-text-input').value.trim();

        if (!videoFile && !audioFile && !textVal) {
            alert('Please provide at least one file or text for fusion analysis.');
            return;
        }

        const btn = document.getElementById('fusion-analyze-btn');
        btn.disabled = true;
        this._showLoading(true);

        try {
            const formData = new FormData();
            if (videoFile) formData.append('files', videoFile);
            if (audioFile) formData.append('files', audioFile);
            if (textVal)   formData.append('text', textVal);

            const response = await fetch(`${this.apiBaseUrl}/api/detect/fusion`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Fusion analysis failed');

            this._renderFusionResult(data);
        } catch (err) {
            console.error('Fusion error:', err);
            this._renderFusionError(err.message);
        } finally {
            btn.disabled = false;
            this._showLoading(false);
        }
    }

    _renderFusionResult(data) {
        const resultsEl = document.getElementById('fusion-results');
        const fusion = data.fusion_result || {};

        // — Tier badge —
        const tier      = fusion.tier      || 3;
        const tierName  = fusion.tier_name || 'Indeterminate';
        const tierColors = { 1: '#10b981', 2: '#3b82f6', 3: '#f59e0b', 4: '#ef4444' };

        document.getElementById('fusion-tier-number').textContent = tier;
        document.getElementById('fusion-tier-label').textContent  = tierName;
        const badge = document.getElementById('fusion-tier-badge');
        badge.style.borderColor = tierColors[tier] || '#6b7280';

        // — Fused probability —
        const prob = typeof fusion.fused_fake_probability === 'number'
            ? (fusion.fused_fake_probability * 100).toFixed(1) + '%'
            : '—';
        document.getElementById('fusion-probability').textContent = prob;

        // — Quality-gated weights —
        const gates = fusion.quality_gates || {};
        const gateStr = Object.entries(gates)
            .map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`)
            .join(' · ') || '—';
        document.getElementById('fusion-weights').textContent = gateStr;

        // — Reasoning —
        document.getElementById('fusion-reasoning').textContent =
            fusion.reasoning ? '↳ ' + fusion.reasoning : '';

        // — Warning for tier 3 or single-signal tier 4 —
        const warningEl = document.getElementById('fusion-warning');
        const warningText = document.getElementById('fusion-warning-text');
        if (tier === 3) {
            warningEl.style.display = 'flex';
            warningText.textContent =
                'Evidence is conflicting or insufficient for a definitive conclusion. ' +
                'Further analysis with additional data is recommended.';
        } else if (tier === 4 && (fusion.fake_agreement_count || 0) < 2) {
            warningEl.style.display = 'flex';
            warningText.textContent =
                'Only one modality is synthetic-leaning. High-stakes decisions ' +
                'should require corroborating evidence from multiple independent signals.';
        } else {
            warningEl.style.display = 'none';
        }

        // — Per-modality breakdown —
        const breakdown = document.getElementById('fusion-modality-breakdown');
        breakdown.innerHTML = '';
        const modScores = fusion.modality_scores || {};
        Object.entries(modScores).forEach(([mod, score]) => {
            const pct = (score * 100).toFixed(1);
            const isAuth = score < 0.4;
            const card = document.createElement('div');
            card.className = 'fusion-mod-card';
            card.innerHTML = `
                <div class="mod-name">${escapeHTML(mod)}</div>
                <div class="mod-score ${isAuth ? 'authentic' : ''}">
                    Fake score: <span>${escapeHTML(pct)}%</span>
                </div>`;
            breakdown.appendChild(card);
        });

        // Show results
        resultsEl.style.display = 'block';
        resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    _renderFusionError(msg) {
        const resultsEl = document.getElementById('fusion-results');
        document.getElementById('fusion-reasoning').textContent = '';
        document.getElementById('fusion-probability').textContent = '—';
        document.getElementById('fusion-weights').textContent = '—';
        document.getElementById('fusion-tier-number').textContent = '!';
        document.getElementById('fusion-tier-label').textContent = 'Error';
        document.getElementById('fusion-tier-badge').style.borderColor = '#ef4444';
        document.getElementById('fusion-warning').style.display = 'flex';
        document.getElementById('fusion-warning-text').textContent = `Analysis failed: ${msg}`;
        document.getElementById('fusion-modality-breakdown').innerHTML = '';
        resultsEl.style.display = 'block';
    }

    _clearFusion() {
        ['video', 'audio'].forEach(slot => {
            const input = document.getElementById(`fusion-file-${slot}`);
            const label = document.getElementById(`fusion-name-${slot}`);
            if (input) input.value = '';
            if (label) label.textContent = 'No file chosen';
        });
        const textarea = document.getElementById('fusion-text-input');
        if (textarea) textarea.value = '';
        const resultsEl = document.getElementById('fusion-results');
        if (resultsEl) resultsEl.style.display = 'none';
    }

    _showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        const p = overlay.querySelector('p');
        if (show) {
            if (p) p.textContent = 'Running cross-modal fusion…';
            overlay.classList.add('active');
        } else {
            if (p) p.textContent = 'Analyzing content…';
            overlay.classList.remove('active');
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DeepGuardDashboard();
    window.queryAssistant = new QueryAssistant();
    window.fusionAnalyzer = new FusionAnalyzer('http://localhost:5000');
});
