document.addEventListener('DOMContentLoaded', () => {
    let telemetryChartInstance = null;

    const navLinks = document.querySelectorAll('.sidebar__link');
    const contentViews = document.querySelectorAll('.content__view');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const targetTab = link.getAttribute('data-target');

            navLinks.forEach(nl => nl.classList.remove('sidebar__link--active'));
            contentViews.forEach(cv => cv.classList.remove('content__view--active'));

            link.classList.add('sidebar__link--active');
            document.getElementById(targetTab).classList.add('content__view--active');
        });
    });

    const btnRunInference = document.getElementById('btnRunInference');
    const singleInputText = document.getElementById('singleInputText');
    const singleResultContainer = document.getElementById('singleResultContainer');
    const resCleanedText = document.getElementById('resCleanedText');
    const resBadge = document.getElementById('resBadge');
    const resConfidence = document.getElementById('resConfidence');

    btnRunInference.addEventListener('click', async () => {
        const rawText = singleInputText.value.trim();
        if (!rawText) return;

        btnRunInference.disabled = true;
        btnRunInference.innerText = 'Analyzing...';

        try {
            const response = await fetch('/predict/single', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: rawText })
            });

            if (!response.ok) throw new Error('API processing failure');

            const data = await response.json();

            resCleanedText.innerText = data.cleaned_text || '(Empty tokens after noise removal)';
            resConfidence.innerText = `${(data.confidence * 100).toFixed(2)}%`;

            if (data.prediction === 1) {
                resBadge.innerText = 'Positive (1)';
                resBadge.className = 'crisis-badge crisis-badge--positive';
            } else {
                resBadge.innerText = 'Negative (0)';
                resBadge.className = 'crisis-badge crisis-badge--critical';
            }

            let aspectWrapper = document.getElementById('singleAspectWrapper');
            if (!aspectWrapper) {
                const resultGrid = document.querySelector('.result-grid');
                const aspectItem = document.createElement('div');
                aspectItem.className = 'result-item';
                aspectItem.innerHTML = `<span class="result-item__label">Detected Aspect</span><div id="singleAspectWrapper" style="margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap;"></div>`;
                resultGrid.appendChild(aspectItem);
                aspectWrapper = document.getElementById('singleAspectWrapper');
            }
            
            aspectWrapper.innerHTML = '';
            data.aspect.forEach(asp => {
                const badge = document.createElement('span');
                const cssClassModifier = asp.toLowerCase().replace(' ', '-');
                badge.className = `crisis-badge crisis-badge--${cssClassModifier}`;
                badge.innerText = asp;
                aspectWrapper.appendChild(badge);
            });

            singleResultContainer.style.display = 'block';
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btnRunInference.disabled = false;
            btnRunInference.innerText = 'Analyze Sentiment';
        }
    });

    const batchFileInput = document.getElementById('batchFileInput');
    const dropzone = document.getElementById('dropzone');
    const batchProgress = document.getElementById('batchProgress');

    batchFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleBatchUpload(e.target.files[0]);
        }
    });

    async function handleBatchUpload(file) {
        if (!file) return;

        dropzone.style.display = 'none';
        batchProgress.style.display = 'flex';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict/batch', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Batch calculation failed');
            }

            const data = await response.json();

            document.getElementById('metricTotal').innerText = data.telemetry.total_records.toLocaleString();
            document.getElementById('metricPositive').innerText = `${data.telemetry.positive_percentage}%`;
            document.getElementById('countPositive').innerText = `${data.telemetry.positive_count.toLocaleString()} instances`;
            document.getElementById('metricNegative').innerText = `${data.telemetry.negative_percentage}%`;
            document.getElementById('countNegative').innerText = `${data.telemetry.negative_count.toLocaleString()} instances`;
            document.getElementById('metricCrisis').innerText = data.crisis_queue.length.toLocaleString();

            renderTelemetryChart(data.telemetry.positive_count, data.telemetry.negative_count);
            renderCrisisTable(data.crisis_queue);

            document.querySelector('[data-target="tab-dashboard"]').click();
        } catch (error) {
            alert(`Error processing file: ${error.message}`);
            dropzone.style.display = 'block';
        } finally {
            batchProgress.style.display = 'none';
            batchFileInput.value = '';
            dropzone.style.display = 'block';
        }
    }

    function renderTelemetryChart(positive, negative) {
        const ctx = document.getElementById('telemetryChart').getContext('2d');
        
        if (telemetryChartInstance) {
            telemetryChartInstance.destroy();
        }

        telemetryChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive Voice', 'Negative Voice'],
                datasets: [{
                    data: [positive, negative],
                    backgroundColor: ['#10b981', '#f59e0b'],
                    borderWidth: 1,
                    borderColor: '#1f2937'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#f3f4f6', font: { size: 12 } }
                    }
                }
            }
        });
    }

    function renderCrisisTable(queue) {
        const tbody = document.getElementById('crisisTableBody');
        tbody.innerHTML = '';

        if (queue.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: #64748b; padding: 40px 0;">No severe crisis detected in this batch.</td></tr>`;
            return;
        }

        queue.forEach((item, index) => {
            const row = document.createElement('tr');
            
            const idCell = document.createElement('td');
            idCell.style.fontWeight = '600';
            idCell.style.color = '#64748b';
            idCell.innerText = String(index + 1).padStart(2, '0');
            row.appendChild(idCell);

            const textCell = document.createElement('td');
            textCell.innerText = item.review_text;
            row.appendChild(textCell);

            const aspectCell = document.createElement('td');
            const aspectContainer = document.createElement('div');
            aspectContainer.style.display = 'flex';
            aspectContainer.style.gap = '6px';
            aspectContainer.style.flexWrap = 'wrap';
            
            item.aspect.forEach(asp => {
                const aspBadge = document.createElement('span');
                const cssClassModifier = asp.toLowerCase().replace(' ', '-');
                aspBadge.className = `crisis-badge crisis-badge--${cssClassModifier}`;
                aspBadge.innerText = asp;
                aspectContainer.appendChild(aspBadge);
            });
            aspectCell.appendChild(aspectContainer);
            row.appendChild(aspectCell);

            const confCell = document.createElement('td');
            const confBadge = document.createElement('span');
            confBadge.className = 'crisis-badge crisis-badge--critical';
            confBadge.innerText = `${(item.confidence * 100).toFixed(2)}%`;
            confCell.appendChild(confBadge);
            row.appendChild(confCell);

            tbody.appendChild(row);
        });
    }
});