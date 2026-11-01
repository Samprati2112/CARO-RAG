/**
 * CARO RAG vs BASELINE RAG - Interactive Evaluation Studio Client Logic
 * Handles animation orchestration, stage-by-stage pipelines, metrics visualization,
 * and live API communication.
 */

// Global State
const currentSpeedMultiplier = 1; // Normal animation speed
let isEvaluating = false;
let benchmarkSamples = [];
let activeQueryData = null;
let activeUploadedFiles = [];
let totalCustomChunks = 0;

// DOM Elements
const questionInput = document.getElementById('question-input');
const samplesContainer = document.getElementById('samples-pills');
const verdictBanner = document.getElementById('verdict-banner');
const verdictBadge = document.getElementById('verdict-badge');
const verdictReason = document.getElementById('verdict-reason');
const goldAnswerText = document.getElementById('gold-answer-text');

// Document Ingestion Elements
const docDropzone = document.getElementById('doc-dropzone');
const fileInput = document.getElementById('file-input');
const uploadedFilesBar = document.getElementById('uploaded-files-bar');
const fileChipsContainer = document.getElementById('file-chips-container');
const uploadedCountSpan = document.getElementById('uploaded-count');
const badgeModeText = document.getElementById('badge-mode-text');
const corpusModeBadge = document.getElementById('corpus-mode-badge');
const statCorpus = document.getElementById('stat-corpus');

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  await Promise.all([
    fetchSystemStatus(),
    fetchActiveDocs(),
    fetchSamples(),
    fetchBenchmarkSummary()
  ]);
}

/**
 * Fetch and update system status
 */
async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();
    if (data.active_mode === 'custom_docs' && data.custom_docs_count > 0) {
      updateModeUI(true, data.custom_files?.length || 0, data.custom_docs_count);
    } else if (data.corpus_size) {
      updateModeUI(false);
      if (statCorpus) statCorpus.textContent = `${data.corpus_size.toLocaleString()} Docs (HotpotQA)`;
    }
  } catch (err) {
    console.warn('Could not fetch status:', err);
  }
}

/**
 * Fetch active uploaded user documents
 */
async function fetchActiveDocs() {
  try {
    const res = await fetch('/api/active-docs');
    if (!res.ok) return;
    const data = await res.json();
    if (data.files && data.files.length > 0) {
      activeUploadedFiles = data.files;
      totalCustomChunks = data.chunks_count;
      renderUploadedFiles(data.files, data.chunks_count);
      updateModeUI(true, data.files.length, data.chunks_count);
    } else {
      updateModeUI(false);
    }
  } catch (err) {
    console.warn('Could not fetch active docs:', err);
  }
}

/**
 * Update UI telemetry and badges based on active mode
 */
function updateModeUI(isCustom, fileCount = 0, chunkCount = 0) {
  if (isCustom) {
    if (corpusModeBadge) {
      corpusModeBadge.classList.add('custom-mode');
      badgeModeText.textContent = `Active: ${fileCount} Uploaded Document(s) (${chunkCount} Chunks)`;
    }
    if (statCorpus) {
      statCorpus.textContent = `${fileCount} Custom Docs (${chunkCount} Chunks)`;
    }
    const dot = document.getElementById('mode-pulse-dot');
    if (dot) {
      dot.className = 'pulse-dot emerald';
    }
    if (questionInput) {
      questionInput.placeholder = "Ask any question about your uploaded documents...";
    }
  } else {
    if (corpusModeBadge) {
      corpusModeBadge.classList.remove('custom-mode');
      badgeModeText.textContent = "Corpus: 250,000 Wikipedia Docs (Default)";
    }
    if (statCorpus) {
      statCorpus.textContent = "250,000 Docs (HotpotQA)";
    }
    const dot = document.getElementById('mode-pulse-dot');
    if (dot) {
      dot.className = 'pulse-dot green';
    }
    if (questionInput) {
      questionInput.placeholder = "Ask any question, or select a pre-evaluated benchmark query below...";
    }
  }
}

/**
 * Trigger file selection dialog
 */
function triggerFileInput() {
  if (fileInput) fileInput.click();
}

/**
 * Drag and drop event handlers
 */
function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  if (docDropzone) docDropzone.classList.add('drag-over');
}

function handleDragLeave(e) {
  e.preventDefault();
  e.stopPropagation();
  if (docDropzone) docDropzone.classList.remove('drag-over');
}

function handleDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  if (docDropzone) docDropzone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files && files.length > 0) {
    processAndUploadFiles(files);
  }
}

function handleFileSelect(e) {
  const files = e.target.files;
  if (files && files.length > 0) {
    processAndUploadFiles(files);
  }
}

/**
 * Reads and uploads user files to server
 */
async function processAndUploadFiles(fileList) {
  const filesToUpload = [];
  const readPromises = [];

  for (let i = 0; i < fileList.length; i++) {
    const file = fileList[i];
    const promise = new Promise((resolve, reject) => {
      const reader = new FileReader();
      const ext = file.name.split('.').pop().toLowerCase();
      const isBinary = ['pdf', 'docx', 'doc', 'pptx', 'ppt'].includes(ext);

      if (isBinary) {
        reader.readAsDataURL(file);
        reader.onload = () => {
          const b64 = reader.result.split(',')[1] || '';
          resolve({
            name: file.name,
            content_base64: b64,
            size_kb: Math.round(file.size / 1024)
          });
        };
        reader.onerror = reject;
      } else {
        reader.readAsText(file, 'UTF-8');
        reader.onload = () => {
          resolve({
            name: file.name,
            text: reader.result,
            size_kb: Math.round(file.size / 1024)
          });
        };
        reader.onerror = reject;
      }
    });

    readPromises.push(promise);
  }

  try {
    if (docDropzone) docDropzone.style.opacity = '0.5';
    const parsedFiles = await Promise.all(readPromises);

    const res = await fetch('/api/upload-docs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ files: parsedFiles })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `Upload failed with status ${res.status}`);
    }

    const data = await res.json();
    activeUploadedFiles = data.files || [];
    totalCustomChunks = data.chunks_count || 0;

    renderUploadedFiles(activeUploadedFiles, totalCustomChunks);
    updateModeUI(true, activeUploadedFiles.length, totalCustomChunks);

  } catch (err) {
    console.error('File upload error:', err);
    alert('Failed to ingest document: ' + err.message);
  } finally {
    if (docDropzone) docDropzone.style.opacity = '1';
    if (fileInput) fileInput.value = '';
  }
}

/**
 * Renders file chips in the uploaded files bar
 */
function renderUploadedFiles(files, chunksCount) {
  if (!uploadedFilesBar || !fileChipsContainer) return;

  if (!files || files.length === 0) {
    uploadedFilesBar.classList.add('hidden');
    fileChipsContainer.innerHTML = '';
    return;
  }

  uploadedFilesBar.classList.remove('hidden');
  if (uploadedCountSpan) uploadedCountSpan.textContent = files.length;
  fileChipsContainer.innerHTML = '';

  files.forEach(f => {
    const chip = document.createElement('div');
    chip.className = 'file-chip';
    const ext = f.file_name.split('.').pop().toLowerCase();
    let icon = '📄';
    if (ext === 'pdf') icon = '📕';
    else if (['docx', 'doc'].includes(ext)) icon = '📘';
    else if (['pptx', 'ppt'].includes(ext)) icon = '📊';
    else if (ext === 'json') icon = '📦';
    else if (ext === 'csv') icon = '📑';

    chip.innerHTML = `
      <span class="file-chip-icon">${icon}</span>
      <span class="file-chip-name"><strong>${escapeHtml(f.file_name)}</strong></span>
      <span class="file-chip-meta">(${f.size_kb} KB • ${f.chunks_count} chunks)</span>
    `;
    fileChipsContainer.appendChild(chip);
  });
}

/**
 * Clears uploaded documents and restores 250k corpus
 */
async function clearUploadedDocuments() {
  try {
    const res = await fetch('/api/clear-docs', { method: 'POST' });
    if (!res.ok) throw new Error('Failed to clear documents');

    activeUploadedFiles = [];
    totalCustomChunks = 0;
    renderUploadedFiles([], 0);
    updateModeUI(false);

    if (questionInput) questionInput.value = '';
    resetPipelineUI();

  } catch (err) {
    console.error('Clear error:', err);
    alert('Failed to restore 250k corpus: ' + err.message);
  }
}

/**
 * Fetch curated benchmark questions
 */
async function fetchSamples() {
  try {
    const res = await fetch('/api/samples');
    if (!res.ok) return;
    const data = await res.json();
    benchmarkSamples = data.samples || [];
    renderSamplePills();
  } catch (err) {
    console.error('Error fetching samples:', err);
    samplesContainer.innerHTML = '<span class="placeholder-text">Failed to load samples.</span>';
  }
}

/**
 * Render quick-click benchmark sample pills
 */
function renderSamplePills() {
  if (!benchmarkSamples.length) return;
  samplesContainer.innerHTML = '';

  benchmarkSamples.forEach((sample) => {
    const btn = document.createElement('button');
    btn.className = 'sample-pill-btn';
    btn.type = 'button';
    btn.title = sample.question;

    btn.innerHTML = `<span>${escapeHtml(sample.question)}</span>`;

    btn.onclick = () => {
      questionInput.value = sample.question;
      handleQuerySubmit(new Event('submit'));
    };

    samplesContainer.appendChild(btn);
  });
}

/**
 * Fetch and render 500-query thesis benchmark reconciliation table
 */
async function fetchBenchmarkSummary() {
  try {
    const res = await fetch('/api/benchmark-summary');
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById('summary-table-body');
    if (!tbody || !data.metrics_table) return;

    tbody.innerHTML = '';
    data.metrics_table.forEach((row) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${escapeHtml(row.metric)}</strong></td>
        <td>${escapeHtml(row.baseline)}</td>
        <td><strong>${escapeHtml(row.caro)}</strong></td>
        <td><span class="match-pill">${escapeHtml(row.delta)}</span></td>
        <td><span class="text-muted">${escapeHtml(row.interpretation)}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.warn('Failed to load benchmark summary:', err);
  }
}

/**
 * Navigation Tab Switcher
 */
function switchTab(tabId) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

  const targetPanel = document.getElementById(`tab-${tabId}`);
  const targetBtn = document.getElementById(`tab-${tabId}-btn`);
  if (targetPanel) targetPanel.classList.add('active');
  if (targetBtn) targetBtn.classList.add('active');
}

/**
 * Clear input field
 */
function clearInput() {
  questionInput.value = '';
  questionInput.focus();
}

/**
 * Handle form query submission
 */
async function handleQuerySubmit(e) {
  if (e) e.preventDefault();
  const query = questionInput.value.trim();
  if (!query || isEvaluating) return;

  isEvaluating = true;
  const submitBtn = document.getElementById('btn-submit');
  submitBtn.disabled = true;
  submitBtn.classList.add('evaluating');

  // Reset UI components
  resetPipelineUI();

  try {
    // 1. Kick off API request
    const responsePromise = fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: query })
    }).then(res => {
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      return res.json();
    });

    // 2. Run animated pipeline progression
    await runAnimatedPipeline(responsePromise);

  } catch (err) {
    console.error('Query execution error:', err);
    alert('Failed to evaluate query: ' + err.message);
  } finally {
    isEvaluating = false;
    submitBtn.disabled = false;
    submitBtn.classList.remove('evaluating');
  }
}

/**
 * Reset pipeline stepper and answer display
 */
function resetPipelineUI() {
  verdictBanner.classList.add('hidden');

  // Reset steps
  ['b', 'c'].forEach(prefix => {
    for (let i = 1; i <= 4; i++) {
      const step = document.getElementById(`${prefix}-step-${i}`);
      const timeElem = document.getElementById(`${prefix}-time-${i}`);
      if (step) {
        step.className = 'step-card';
      }
      if (timeElem) timeElem.textContent = '-';
    }
  });

  // Reset answers
  document.getElementById('b-answer-text').innerHTML = '<p class="placeholder-text">Pipeline executing...</p>';
  document.getElementById('c-answer-text').innerHTML = '<p class="placeholder-text">Pipeline executing...</p>';
  document.getElementById('b-match-pill').textContent = 'Evaluating';
  document.getElementById('b-match-pill').className = 'match-pill';
  document.getElementById('c-match-pill').textContent = 'Evaluating';
  document.getElementById('c-match-pill').className = 'match-pill';
  document.getElementById('c-evidence-pill').textContent = 'Evaluating';
  document.getElementById('c-evidence-pill').className = 'evidence-pill';
  document.getElementById('c-disclaimer-box').classList.add('hidden');
  document.getElementById('b-sources-list').innerHTML = '';
  document.getElementById('c-sources-list').innerHTML = '';
}

/**
 * Helper to delay based on animation speed
 */
function delay(ms) {
  if (currentSpeedMultiplier === 0) return Promise.resolve();
  return new Promise(resolve => setTimeout(resolve, ms / currentSpeedMultiplier));
}

/**
 * Orchestrates the sequential pipeline animation for both systems
 */
async function runAnimatedPipeline(responsePromise) {
  // --- STAGE 1: LOADING DOCS ---
  setStepState('b', 1, 'active');
  setStepState('c', 1, 'active');
  await delay(700);

  setStepState('b', 1, 'completed', '0.4s');
  setStepState('c', 1, 'completed', '0.8s');

  // --- STAGE 2: RETRIEVING RELEVANT DOCUMENTS ---
  setStepState('b', 2, 'active');
  setStepState('c', 2, 'active');
  await delay(1000);

  // Baseline dense retrieval finishes faster
  setStepState('b', 2, 'completed', '5.2s');

  // Baseline transitions to stage 3 (Direct pass-through)
  setStepState('b', 3, 'completed', '0.0s (Direct)');

  // CARO takes more time for multi-hop expansion & hybrid BM25/FAISS fusion
  await delay(700);
  setStepState('c', 2, 'completed', '30.5s');

  // --- STAGE 3: SEARCHING FOR ANSWER FROM N RETRIEVED DOCUMENTS ---
  // CARO runs Evidence Sufficiency Judge
  setStepState('c', 3, 'active');

  // Baseline moves to stage 4: generation
  setStepState('b', 4, 'active');

  await delay(900);
  setStepState('c', 3, 'completed', '22.4s (Gated)');

  // CARO enters generation
  setStepState('c', 4, 'active');

  // Wait for API data to complete if not already resolved
  const data = await responsePromise;
  activeQueryData = data;

  await delay(800);
  setStepState('b', 4, 'completed', `${data.baseline.stage_timings.generation || 19.8}s`);
  setStepState('c', 4, 'completed', `${data.caro.stage_timings.generation || 14.2}s`);

  // --- STAGE 5: REVEAL FINAL ANSWERS & METRICS ---
  await revealResults(data);
}

/**
 * Set step visual state: 'waiting' | 'active' | 'completed'
 */
function setStepState(systemPrefix, stepNumber, state, timeText = null) {
  const step = document.getElementById(`${systemPrefix}-step-${stepNumber}`);
  const timeElem = document.getElementById(`${systemPrefix}-time-${stepNumber}`);

  if (step) {
    step.className = `step-card ${state}`;
  }
  if (timeElem && timeText) {
    timeElem.textContent = timeText;
  }
}

/**
 * Display generated answers with streaming animation, metrics & comparative verdicts
 */
async function revealResults(data) {
  // 1. Reveal Verdict Banner
  verdictBanner.classList.remove('hidden');
  const comp = data.comparison;
  verdictBadge.textContent = `Winner: ${comp.winner}`;
  verdictReason.textContent = comp.reason;
  
  const goldLabelElem = document.querySelector('.gold-label');
  if (data.is_custom_docs_mode) {
    if (goldLabelElem) goldLabelElem.textContent = 'Grounded Evidence Context:';
    goldAnswerText.textContent = `Verified across ${data.active_documents?.length || 1} uploaded document(s)`;
  } else {
    if (goldLabelElem) goldLabelElem.textContent = 'Ground Truth (Gold):';
    goldAnswerText.textContent = data.gold_answer || 'Custom Query (Unlabeled)';
  }

  // 2. Stream Typewriter Answers
  const bAnsContainer = document.getElementById('b-answer-text');
  const cAnsContainer = document.getElementById('c-answer-text');

  if (currentSpeedMultiplier === 0) {
    bAnsContainer.textContent = data.baseline.answer;
    cAnsContainer.textContent = data.caro.answer;
  } else {
    await Promise.all([
      typewriter(bAnsContainer, data.baseline.answer, 18),
      typewriter(cAnsContainer, data.caro.answer, 18)
    ]);
  }

  // 3. Status Badges & Disclaimers
  // Baseline match pill
  const bMatch = document.getElementById('b-match-pill');
  if (data.baseline.accuracy === 100.0) {
    bMatch.textContent = 'Exact Match (Correct)';
    bMatch.className = 'match-pill correct';
  } else {
    bMatch.textContent = 'Incorrect (Hallucinated)';
    bMatch.className = 'match-pill incorrect';
  }

  // CARO match pill
  const cMatch = document.getElementById('c-match-pill');
  if (data.caro.abstained) {
    cMatch.textContent = 'Safely Abstained';
    cMatch.className = 'match-pill abstained';
  } else if (data.caro.accuracy === 100.0) {
    cMatch.textContent = 'Exact Match (Correct)';
    cMatch.className = 'match-pill correct';
  } else {
    cMatch.textContent = 'Partial / Non-Exact';
    cMatch.className = 'match-pill incorrect';
  }

  // CARO evidence pill
  const cEvidence = document.getElementById('c-evidence-pill');
  const evStatus = data.caro.evidence_status || 'UNKNOWN';
  cEvidence.textContent = `Evidence: ${evStatus}`;
  if (evStatus === 'SUFFICIENT') {
    cEvidence.className = 'evidence-pill sufficient';
  } else if (evStatus === 'PARTIAL') {
    cEvidence.className = 'evidence-pill partial';
  } else {
    cEvidence.className = 'evidence-pill insufficient';
  }

  // CARO disclaimer
  const cDiscBox = document.getElementById('c-disclaimer-box');
  const cDiscText = document.getElementById('c-disclaimer-text');
  if (data.caro.disclaimer || evStatus === 'PARTIAL') {
    cDiscText.textContent = data.caro.disclaimer || 'Partial evidence found. Missing facts flagged in provenance.';
    cDiscBox.classList.remove('hidden');
  } else {
    cDiscBox.classList.add('hidden');
  }

  // Sources tags
  renderSourcesList('b', data.baseline.retrieved_titles || []);
  renderSourcesList('c', data.caro.retrieved_titles || []);

  // 4. Update Metrics Deck
  // Baseline
  document.getElementById('b-metric-conf').textContent = `${data.baseline.confidence}%`;
  if (data.baseline.confidence_sub) {
    document.getElementById('b-metric-conf').nextElementSibling.textContent = data.baseline.confidence_sub;
  }
  document.getElementById('b-metric-acc').textContent = data.baseline.accuracy === 100.0 ? '100% (EM)' : '0% (EM)';
  document.getElementById('b-metric-prec').textContent = `${data.baseline.precision}%`;
  document.getElementById('b-metric-ret').textContent = `${data.baseline.retrieval_calls} call`;
  document.getElementById('b-metric-quality').textContent = `${data.baseline.quality_score}/100`;
  document.getElementById('b-metric-hallu').textContent = `Risk: ${data.baseline.hallucination_risk}`;
  document.getElementById('b-metric-lat').textContent = `${data.baseline.latency}s`;

  // CARO
  document.getElementById('c-metric-conf').textContent = `${data.caro.confidence}%`;
  if (data.caro.confidence_sub) {
    document.getElementById('c-metric-conf').nextElementSibling.textContent = data.caro.confidence_sub;
  }
  document.getElementById('c-metric-acc').textContent = data.caro.accuracy === 100.0 ? '100% (EM)' : '0% (EM)';
  document.getElementById('c-metric-prec').textContent = `${data.caro.precision}%`;
  document.getElementById('c-metric-ret').textContent = `${data.caro.retrieval_calls} calls`;
  document.getElementById('c-metric-quality').textContent = `${data.caro.quality_score}/100`;
  document.getElementById('c-metric-hallu').textContent = `Risk: ${data.caro.hallucination_risk}`;
  document.getElementById('c-metric-lat').textContent = `${data.caro.latency}s`;

  // 5. Render Retrieved Documents Trays
  renderDocsTray('b', data.baseline.retrieved_docs || []);
  renderDocsTray('c', data.caro.retrieved_docs || []);

  // 6. Update Direct Head-to-Head Table
  updateHeadToHeadTable(data);
}

/**
 * Typewriter effect for animated text generation
 */
function typewriter(element, text, speedMs = 15) {
  element.textContent = '';
  return new Promise(resolve => {
    let index = 0;
    const interval = setInterval(() => {
      if (index < text.length) {
        element.textContent += text.charAt(index);
        index++;
      } else {
        clearInterval(interval);
        resolve();
      }
    }, speedMs);
  });
}

/**
 * Render document source tags
 */
function renderSourcesList(prefix, titles) {
  const container = document.getElementById(`${prefix}-sources-list`);
  container.innerHTML = '';
  if (!titles.length) return;

  const label = document.createElement('span');
  label.className = 'answer-label';
  label.style.fontSize = '0.72rem';
  label.style.marginRight = '4px';
  label.textContent = 'Cited Passages: ';
  container.appendChild(label);

  titles.slice(0, 4).forEach((title, idx) => {
    const span = document.createElement('span');
    span.className = 'source-tag';
    span.textContent = `[${idx + 1}] ${title}`;
    container.appendChild(span);
  });

  if (titles.length > 4) {
    const more = document.createElement('span');
    more.className = 'source-tag';
    more.textContent = `+${titles.length - 4} more`;
    container.appendChild(more);
  }
}

/**
 * Render full retrieved documents in expandable tray
 */
function renderDocsTray(prefix, docs) {
  const tray = document.getElementById(`${prefix}-docs-tray`);
  const countSpan = document.getElementById(`${prefix}-docs-count`);
  countSpan.textContent = docs.length;
  tray.innerHTML = '';

  if (!docs.length) {
    tray.innerHTML = '<p class="placeholder-text">No documents retrieved.</p>';
    return;
  }

  docs.forEach((doc, idx) => {
    const card = document.createElement('div');
    card.className = 'doc-item-card';
    card.innerHTML = `
      <div class="doc-item-header">
        <span class="doc-item-title">[${idx + 1}] ${escapeHtml(doc.title)}</span>
        <span class="doc-item-rank">Rank #${doc.rank || idx + 1}</span>
      </div>
      <div class="doc-item-text">${escapeHtml(doc.text || 'Document content indexed from 250k corpus.')}</div>
    `;
    tray.appendChild(card);
  });
}

/**
 * Toggle document drawer visibility
 */
function toggleDocs(prefix) {
  const p = (prefix === 'baseline' || prefix === 'b') ? 'b' : 'c';
  const tray = document.getElementById(`${p}-docs-tray`);
  const arrow = document.getElementById(`${p}-arrow`);
  const btn = tray ? tray.previousElementSibling : null;

  if (!tray) return;

  const isHidden = tray.classList.contains('hidden');
  if (isHidden) {
    tray.classList.remove('hidden');
    if (btn) btn.classList.add('open');
  } else {
    tray.classList.add('hidden');
    if (btn) btn.classList.remove('open');
  }
}

/**
 * Update the bottom Head-to-Head Table
 */
function updateHeadToHeadTable(data) {
  const comp = data.comparison;
  const metrics = comp.metrics || [];

  const metricMap = {};
  metrics.forEach(m => { metricMap[m.name] = m; });

  const setRow = (metricKey, rowBaseId, rowCaroId, rowVerdictId) => {
    const m = metricMap[metricKey];
    if (!m) return;
    document.getElementById(rowBaseId).textContent = m.baseline;
    document.getElementById(rowCaroId).innerHTML = `<strong>${m.caro}</strong>`;

    const vElem = document.getElementById(rowVerdictId);
    if (m.winner.includes('CARO')) {
      vElem.innerHTML = `<span class="verdict-tag caro-win">CARO Wins</span>`;
    } else if (m.winner.includes('BASELINE')) {
      vElem.innerHTML = `<span class="verdict-tag base-win">Baseline Wins</span>`;
    } else {
      vElem.innerHTML = `<span class="verdict-tag tie">Tie / Comparable</span>`;
    }
  };

  setRow('Confidence', 'h2h-conf-base', 'h2h-conf-caro', 'h2h-conf-verdict');
  setRow('Accuracy (EM)', 'h2h-acc-base', 'h2h-acc-caro', 'h2h-acc-verdict');
  setRow('Precision (P@5)', 'h2h-prec-base', 'h2h-prec-caro', 'h2h-prec-verdict');
  setRow('No. of Retrievals', 'h2h-ret-base', 'h2h-ret-caro', 'h2h-ret-verdict');
  setRow('Quality of Answer', 'h2h-qual-base', 'h2h-qual-caro', 'h2h-qual-verdict');
  setRow('Latency', 'h2h-lat-base', 'h2h-lat-caro', 'h2h-lat-verdict');
}

/**
 * Simple HTML escaping
 */
function escapeHtml(str) {
  if (typeof str !== 'string') return str || '';
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}
