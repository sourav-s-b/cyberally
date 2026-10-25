/**
 * CyberAlly - Core Frontend Controller
 * Matches Homepage, Simulation Configuration & Updated Dashboard Screens
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- View Elements ---
  const homeView = document.getElementById('home-view');
  const configView = document.getElementById('config-view');
  const dashboardView = document.getElementById('dashboard-view');
  
  // --- Navigation Elements ---
  const startSimButtons = document.querySelectorAll('.start-sim-btn');
  const dashNavItems = document.querySelectorAll('.dash-nav-item');
  const homeLogoBtn = document.getElementById('home-logo-btn');
  const sidebarBrandBtn = document.getElementById('sidebar-brand-btn');
  const configBackBtn = document.getElementById('config-back-btn');
  const configBrandBtn = document.getElementById('config-brand-btn');
  const btnLaunchSim = document.getElementById('btn-launch-sim');
  
  // --- Config Mode Selection ---
  let selectedAttackMode = 'scripted'; // 'scripted' or 'llm'
  const cardScripted = document.getElementById('config-card-scripted');
  const cardLlm = document.getElementById('config-card-llm');
  const indicatorScripted = document.getElementById('indicator-scripted');
  const indicatorLlm = document.getElementById('indicator-llm');
  
  // --- Dashboard Tab Panels ---
  const dashTabContents = document.querySelectorAll('.dash-tab-content');
  const dashDynamicTitle = document.getElementById('dash-dynamic-title');
  const dashDynamicDesc = document.getElementById('dash-dynamic-desc');
  
  // --- Controls per tab in dashboard topbar ---
  const liveMonitoringControls = document.getElementById('live-monitoring-controls');
  const analyticsControls = document.getElementById('analytics-controls');
  const reportsControls = document.getElementById('reports-controls');
  
  // --- Simulation State ---
  let isSimRunning = true;
  const simStatusBadge = document.getElementById('sim-status-badge');
  const btnToggleSim = document.getElementById('btn-toggle-sim');
  const simTimer = document.getElementById('sim-timer');
  const liveActiveAttacks = document.getElementById('live-active-attacks');
  const liveActiveDefenses = document.getElementById('live-active-defenses');
  const liveTotalEvents = document.getElementById('live-total-events');
  const liveEventsTbody = document.getElementById('live-events-tbody');
  
  // --- Simulation Running Timer (00:12:34) ---
  let simSeconds = 754; // starts at 12m 34s matching screenshot
  setInterval(() => {
    if (isSimRunning && simTimer) {
      simSeconds++;
      const hrs = String(Math.floor(simSeconds / 3600)).padStart(2, '0');
      const mins = String(Math.floor((simSeconds % 3600) / 60)).padStart(2, '0');
      const secs = String(simSeconds % 60).padStart(2, '0');
      simTimer.textContent = `${hrs}:${mins}:${secs}`;
    }
  }, 1000);
  
  // --- Theme Toggle Elements ---
  const homeThemeToggle = document.getElementById('home-theme-toggle');
  const dashThemeToggle = document.getElementById('dash-theme-toggle');
  const moonIcon = document.querySelector('.moon-icon');
  const sunIcon = document.querySelector('.sun-icon');
  
  // --- Modals ---
  const reportViewModal = document.getElementById('report-view-modal');
  const generateReportModal = document.getElementById('generate-report-modal');
  const demoModal = document.getElementById('demo-modal');
  const docModal = document.getElementById('doc-modal');
  
  // =========================================================================
  // VIEW SWITCHING (Home <-> Config <-> Dashboard)
  // =========================================================================
  function showView(viewId) {
    homeView.classList.remove('active');
    if (configView) configView.classList.remove('active');
    dashboardView.classList.remove('active');

    if (viewId === 'dashboard') {
      dashboardView.classList.add('active');
    } else if (viewId === 'config') {
      if (configView) configView.classList.add('active');
    } else {
      homeView.classList.add('active');
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Hash-based routing to allow direct URL access (#config, #live-monitoring, #analytics, etc.)
  function handleHashRoute() {
    const rawHash = window.location.hash.replace(/^#\/?/, '').toLowerCase();
    if (rawHash === 'config' || rawHash === 'configuration' || rawHash === 'simulation-config') {
      closeAllModals();
      showView('config');
    } else if (rawHash === 'live-monitoring' || rawHash === 'live') {
      closeAllModals();
      showView('dashboard');
      activateDashTab('live-monitoring');
    } else if (rawHash === 'analytics') {
      closeAllModals();
      showView('dashboard');
      activateDashTab('analytics');
    } else if (rawHash === 'reports') {
      closeAllModals();
      showView('dashboard');
      activateDashTab('reports');
    } else if (rawHash === 'export') {
      closeAllModals();
      showView('dashboard');
      activateDashTab('export');
    } else if (rawHash === 'home' || rawHash === '') {
      showView('home');
    }
  }
  window.addEventListener('hashchange', handleHashRoute);

  // "Start Simulation" on Homepage -> goes to Simulation Configuration
  startSimButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      closeAllModals();
      window.location.hash = 'config';
      showView('config');
    });
  });

  // Back button on Config View -> returns to Home
  if (configBackBtn) {
    configBackBtn.addEventListener('click', () => {
      window.location.hash = 'home';
      showView('home');
    });
  }
  if (configBrandBtn) {
    configBrandBtn.addEventListener('click', () => {
      window.location.hash = 'home';
      showView('home');
    });
  }

  // Attack Mode Selection Cards
  function updateAttackModeUI(mode) {
    selectedAttackMode = mode;
    if (mode === 'scripted') {
      if (cardScripted) cardScripted.classList.add('selected');
      if (cardLlm) cardLlm.classList.remove('selected');
      if (indicatorScripted) {
        indicatorScripted.className = 'config-selection-indicator checked';
        indicatorScripted.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
      }
      if (indicatorLlm) {
        indicatorLlm.className = 'config-selection-indicator unchecked';
        indicatorLlm.innerHTML = '';
      }
    } else {
      if (cardLlm) cardLlm.classList.add('selected');
      if (cardScripted) cardScripted.classList.remove('selected');
      if (indicatorLlm) {
        indicatorLlm.className = 'config-selection-indicator checked';
        indicatorLlm.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
      }
      if (indicatorScripted) {
        indicatorScripted.className = 'config-selection-indicator unchecked';
        indicatorScripted.innerHTML = '';
      }
    }
  }

  if (cardScripted) {
    cardScripted.addEventListener('click', () => updateAttackModeUI('scripted'));
  }
  if (cardLlm) {
    cardLlm.addEventListener('click', () => updateAttackModeUI('llm'));
  }

  // "Next →" on Config View -> launches Live Monitoring
  if (btnLaunchSim) {
    btnLaunchSim.addEventListener('click', () => {
      window.location.hash = 'live-monitoring';
      showView('dashboard');
      activateDashTab('live-monitoring');
      const modeName = selectedAttackMode === 'llm' ? 'LLM Red Team Agent' : 'Scripted Attacks';
      showToast(`Simulation launched with ${modeName}`, 'success');
    });
  }

  // Brand logos return to Home
  if (homeLogoBtn) {
    homeLogoBtn.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.hash = 'home';
      showView('home');
    });
  }
  if (sidebarBrandBtn) {
    sidebarBrandBtn.addEventListener('click', () => {
      window.location.hash = 'home';
      showView('home');
    });
  }

  // =========================================================================
  // DASHBOARD TAB NAVIGATION
  // =========================================================================
  const tabMetadata = {
    'home': {
      title: 'Home',
      desc: ''
    },
    'live-monitoring': {
      title: 'Live Monitoring',
      desc: 'Real-time view of attack, defense and system activity'
    },
    'analytics': {
      title: 'Analytics',
      desc: 'Performance metrics, attack/defense insights and penalty analysis'
    },
    'reports': {
      title: 'Reports',
      desc: 'Detailed analysis and summaries of simulation runs'
    },
    'export': {
      title: 'Export',
      desc: 'Export simulation data, reports and logs'
    }
  };

  function activateDashTab(tabId) {
    if (tabId === 'home') {
      showView('home');
      return;
    }

    // Update active nav button
    dashNavItems.forEach((item) => {
      item.classList.toggle('active', item.dataset.dashTab === tabId);
    });

    // Update active panel content
    dashTabContents.forEach((panel) => {
      panel.classList.toggle('active', panel.id === `tab-${tabId}`);
    });

    // Update Header titles
    if (tabMetadata[tabId]) {
      dashDynamicTitle.textContent = tabMetadata[tabId].title;
      dashDynamicDesc.textContent = tabMetadata[tabId].desc;
    }

    // Toggle Tab-Specific Topbar Controls
    liveMonitoringControls.style.display = tabId === 'live-monitoring' ? 'flex' : 'none';
    analyticsControls.style.display = tabId === 'analytics' ? 'flex' : 'none';
    reportsControls.style.display = tabId === 'reports' ? 'flex' : 'none';
  }

  dashNavItems.forEach((btn) => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.dashTab;
      if (tab === 'home') {
        window.location.hash = 'home';
      } else {
        window.location.hash = tab;
      }
      activateDashTab(tab);
    });
  });

  // =========================================================================
  // SIMULATION STATE TOGGLE (Live Monitoring)
  // =========================================================================
  if (btnToggleSim) {
    btnToggleSim.addEventListener('click', () => {
      isSimRunning = !isSimRunning;
      if (isSimRunning) {
        simStatusBadge.className = 'sim-status-badge running';
        simStatusBadge.innerHTML = `<span class="pulse-dot"></span><span class="status-text">Simulation Running</span> <span class="sim-timer-text" id="sim-timer">00:12:34</span>`;
        btnToggleSim.textContent = 'Stop Simulation';
        btnToggleSim.className = 'btn-stop-sim';
        showToast('Autonomous Cyber Simulation Resumed', 'success');
      } else {
        simStatusBadge.className = 'sim-status-badge stopped';
        simStatusBadge.innerHTML = `<span class="pulse-dot"></span><span class="status-text">Simulation Paused</span>`;
        btnToggleSim.textContent = 'Resume Simulation';
        btnToggleSim.className = 'btn-stop-sim resume-state';
        showToast('Simulation execution paused', 'warning');
      }
    });
  }

  // Live Event Log Filter Pills
  const eventFilterPills = document.querySelectorAll('.filter-pill');
  eventFilterPills.forEach((pill) => {
    pill.addEventListener('click', () => {
      eventFilterPills.forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');
      const filter = pill.dataset.filter;

      const rows = liveEventsTbody ? liveEventsTbody.querySelectorAll('tr') : [];
      rows.forEach((row) => {
        if (filter === 'all' || row.dataset.type === filter) {
          row.style.display = '';
        } else {
          row.style.display = 'none';
        }
      });
    });
  });

  // Simulated telemetry ticker for realistic live activity
  let simEventCount = 128;
  const attackTypes = ['SQL injection attempt', 'Lateral movement probing', 'Remote Code Execution (RCE)', 'Credential brute-force'];
  const defenseTypes = ['Blocked suspicious traffic', 'Reinforced authentication token', 'Isolated compromised container', 'Applied zero-day egress filter'];

  setInterval(() => {
    if (!isSimRunning || !dashboardView.classList.contains('active')) return;
    
    if (Math.random() > 0.65) {
      simEventCount++;
      if (liveTotalEvents) liveTotalEvents.textContent = simEventCount;

      const isAttack = Math.random() > 0.45;
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];

      const tr = document.createElement('tr');
      if (isAttack) {
        const desc = attackTypes[Math.floor(Math.random() * attackTypes.length)];
        tr.dataset.type = 'attack';
        tr.innerHTML = `
          <td class="font-mono">${timeStr}</td>
          <td><span class="type-badge badge-attack">ATTACK</span></td>
          <td>RedAgent-1</td>
          <td>DVWA</td>
          <td>${desc}</td>
          <td><span class="status-badge-pill badge-in-progress">In Progress</span></td>
        `;
      } else {
        const desc = defenseTypes[Math.floor(Math.random() * defenseTypes.length)];
        tr.dataset.type = 'defense';
        tr.innerHTML = `
          <td class="font-mono">${timeStr}</td>
          <td><span class="type-badge badge-defense">DEFENSE</span></td>
          <td>BlueAgent-2</td>
          <td>Firewall</td>
          <td>${desc}</td>
          <td><span class="status-badge-pill badge-success">Success</span></td>
        `;
      }

      if (liveEventsTbody) {
        liveEventsTbody.insertBefore(tr, liveEventsTbody.firstChild);
        if (liveEventsTbody.children.length > 8) {
          liveEventsTbody.removeChild(liveEventsTbody.lastChild);
        }
      }
    }
  }, 7000);

  // =========================================================================
  // INTERACTIVE NETWORK TOPOLOGY TOOLTIP
  // =========================================================================
  const networkNodes = document.querySelectorAll('.network-node');
  const nodeTooltip = document.getElementById('node-tooltip');
  const ttName = document.getElementById('tt-name');
  const ttRole = document.getElementById('tt-role');
  const ttIp = document.getElementById('tt-ip');
  const ttStatus = document.getElementById('tt-status');

  networkNodes.forEach((node) => {
    node.addEventListener('mouseenter', () => {
      const name = node.dataset.name || 'Node';
      const role = node.dataset.role || 'Host';
      const ip = node.dataset.ip || '10.0.0.1';
      const status = node.dataset.status || 'Active';

      ttName.textContent = name;
      ttRole.textContent = role;
      ttIp.textContent = ip;
      ttStatus.textContent = status;

      nodeTooltip.classList.add('show');
    });

    node.addEventListener('mousemove', (e) => {
      const container = document.getElementById('network-topology-container');
      if (!container) return;
      const containerRect = container.getBoundingClientRect();
      const x = e.clientX - containerRect.left + 15;
      const y = e.clientY - containerRect.top + 15;
      nodeTooltip.style.left = `${Math.min(x, containerRect.width - 180)}px`;
      nodeTooltip.style.top = `${Math.min(y, containerRect.height - 90)}px`;
    });

    node.addEventListener('mouseleave', () => {
      nodeTooltip.classList.remove('show');
    });
  });

  // =========================================================================
  // ANALYTICS TIME FILTER DROPDOWN
  // =========================================================================
  const analyticsFilterBtn = document.getElementById('analytics-filter-btn');
  const analyticsFilterMenu = document.getElementById('analytics-filter-menu');

  if (analyticsFilterBtn && analyticsFilterMenu) {
    analyticsFilterBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      analyticsFilterMenu.classList.toggle('show');
    });

    document.addEventListener('click', () => {
      analyticsFilterMenu.classList.remove('show');
    });

    analyticsFilterMenu.querySelectorAll('.dropdown-item').forEach((item) => {
      item.addEventListener('click', () => {
        analyticsFilterMenu.querySelectorAll('.dropdown-item').forEach((i) => i.classList.remove('active'));
        item.classList.add('active');
        analyticsFilterBtn.querySelector('span').textContent = item.textContent;
        analyticsFilterMenu.classList.remove('show');
        showToast(`Analytics filtered by: ${item.textContent}`, 'info');
      });
    });
  }

  // =========================================================================
  // REPORTS VIEW & DOWNLOAD LOGIC
  // =========================================================================
  const reportData = {
    '1': {
      title: 'Simulation Report - Run 5',
      date: 'Oct 18, 2026',
      duration: '30 min',
      negativeScore: -125,
      score: '91.4%',
      attempts: 42,
      mitigations: '38 (90.5%)',
      compromised: 1,
      mtt: '1.2s / 2.4s',
      summary: 'Red Team agents launched chained multi-stage penetration attacks against the DMZ and internal microservices. The decentralized MARL Blue Team autonomously deployed host isolation, firewall policy updates, and token reinforcement within an average containment time of 1.8 seconds. Cumulative negative penalty score: -125.'
    },
    '2': {
      title: 'Simulation Report - Run 4',
      date: 'Oct 17, 2026',
      duration: '45 min',
      negativeScore: -80,
      score: '88.2%',
      attempts: 64,
      mitigations: '57 (89.1%)',
      compromised: 1,
      mtt: '1.5s / 2.8s',
      summary: 'Evaluation of LLM-generated attack payloads utilizing dynamic prompt mutation. 57 out of 64 exploitation trajectories were intercepted prior to privilege escalation. Cumulative negative penalty score: -80.'
    },
    '3': {
      title: 'Simulation Report - Run 3',
      date: 'Oct 16, 2026',
      duration: '30 min',
      negativeScore: -210,
      score: '74.0%',
      attempts: 55,
      mitigations: '39 (70.9%)',
      compromised: 3,
      mtt: '2.1s / 3.9s',
      summary: 'Stress test against aggressive multi-vector lateral movement. Higher negative score reflects 3 host compromises during the initial policy exploration phase. Cumulative negative penalty score: -210.'
    },
    '4': {
      title: 'Simulation Report - Run 2',
      date: 'Oct 15, 2026',
      duration: '25 min',
      negativeScore: -45,
      score: '96.5%',
      attempts: 32,
      mitigations: '31 (96.8%)',
      compromised: 0,
      mtt: '1.1s / 1.8s',
      summary: 'High defense resilience run. Zero hosts compromised, with early detection on SQLi and credential brute-force attempts. Cumulative negative penalty score: -45.'
    },
    '5': {
      title: 'Simulation Report - Run 1',
      date: 'Oct 14, 2026',
      duration: '40 min',
      negativeScore: -95,
      score: '89.7%',
      attempts: 52,
      mitigations: '47 (90.4%)',
      compromised: 1,
      mtt: '1.4s / 2.6s',
      summary: 'Initial baseline comparative run establishing policy reward bounds. Cumulative negative penalty score: -95.'
    }
  };

  // Report Modal Tabs (Executive Summary, Attack Timeline, Penalty Breakdown, Findings, Recommendations)
  const reportModalTabs = document.querySelectorAll('.report-modal-tab');
  const reportTabPanels = document.querySelectorAll('.report-tab-panel');

  reportModalTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      reportModalTabs.forEach((t) => t.classList.remove('active'));
      reportTabPanels.forEach((p) => p.classList.remove('active'));
      tab.classList.add('active');
      const targetPanel = document.getElementById(`report-tab-${tab.dataset.reportTab}`);
      if (targetPanel) targetPanel.classList.add('active');
    });
  });

  // View Report Click
  document.querySelectorAll('.btn-view-report').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.id;
      const data = reportData[id] || reportData['1'];
      
      document.getElementById('modal-report-title').textContent = data.title;
      document.getElementById('modal-report-meta').textContent = `Executed on ${data.date} | Duration: ${data.duration} | Status: Completed`;
      
      // Reset to first tab
      reportModalTabs.forEach((t, i) => t.classList.toggle('active', i === 0));
      reportTabPanels.forEach((p, i) => p.classList.toggle('active', i === 0));

      reportViewModal.classList.add('active');
    });
  });

  // Download Report Click
  document.querySelectorAll('.btn-download-report').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.id;
      const data = reportData[id] || reportData['1'];
      downloadFile(`${data.title.replace(/\s+/g, '_')}.json`, JSON.stringify(data, null, 2), 'application/json');
      showToast(`Downloaded: ${data.title}`, 'success');
    });
  });

  // Generate Report Modal
  const btnOpenGenerate = document.getElementById('btn-open-generate-modal');
  const btnCancelGenerate = document.getElementById('btn-cancel-generate');
  const btnConfirmGenerate = document.getElementById('btn-confirm-generate');
  const reportsTbody = document.getElementById('reports-tbody');

  if (btnOpenGenerate) {
    btnOpenGenerate.addEventListener('click', () => {
      generateReportModal.classList.add('active');
    });
  }
  if (btnCancelGenerate) {
    btnCancelGenerate.addEventListener('click', () => {
      generateReportModal.classList.remove('active');
    });
  }

  if (btnConfirmGenerate) {
    btnConfirmGenerate.addEventListener('click', () => {
      const nameInput = document.getElementById('new-report-name');
      const title = nameInput ? nameInput.value : 'Autonomous Defense Analysis';
      
      const newRow = document.createElement('tr');
      const count = reportsTbody.children.length + 1;
      newRow.innerHTML = `
        <td class="text-muted font-mono">${count}</td>
        <td class="font-semibold text-white">${title}</td>
        <td class="text-muted">Just now</td>
        <td class="text-muted">Live</td>
        <td><span class="penalty-score-badge">-75</span></td>
        <td><span class="status-badge-pill badge-success">Completed</span></td>
        <td style="text-align: right;">
          <button class="btn-table-action btn-view-report" data-id="1">View</button>
          <button class="btn-table-action btn-download-report" data-id="1">Download</button>
        </td>
      `;

      reportsTbody.insertBefore(newRow, reportsTbody.firstChild);
      generateReportModal.classList.remove('active');
      showToast(`Report '${title}' successfully generated!`, 'success');

      newRow.querySelector('.btn-view-report').addEventListener('click', () => {
        document.getElementById('modal-report-title').textContent = title;
        reportViewModal.classList.add('active');
      });
      newRow.querySelector('.btn-download-report').addEventListener('click', () => {
        downloadFile(`${title.replace(/\s+/g, '_')}.json`, JSON.stringify({ title, date: 'Oct 24, 2026', negativeScore: -75, status: 'Completed' }, null, 2), 'application/json');
        showToast(`Downloaded: ${title}`, 'success');
      });
    });
  }

  // =========================================================================
  // EXPORT DATA FUNCTIONALITY
  // =========================================================================
  const btnSubmitExport = document.getElementById('btn-submit-export');
  if (btnSubmitExport) {
    btnSubmitExport.addEventListener('click', () => {
      const exportOptions = [];
      if (document.getElementById('exp-event-logs').checked) exportOptions.push('Event Logs');
      if (document.getElementById('exp-network-traffic').checked) exportOptions.push('Network Traffic');
      if (document.getElementById('exp-system-metrics').checked) exportOptions.push('System Metrics');
      if (document.getElementById('exp-analytics-data').checked) exportOptions.push('Analytics Data');
      if (document.getElementById('exp-simulation-report').checked) exportOptions.push('Simulation Report');

      if (exportOptions.length === 0) {
        showToast('Please select at least one data category to export', 'warning');
        return;
      }

      const formatRadio = document.querySelector('input[name="export-format"]:checked');
      const format = formatRadio ? formatRadio.value : 'PDF';

      btnSubmitExport.disabled = true;
      btnSubmitExport.innerHTML = `<span>Exporting Package...</span>`;

      setTimeout(() => {
        btnSubmitExport.disabled = false;
        btnSubmitExport.innerHTML = `
          <span>Export Data</span>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <line x1="5" y1="12" x2="19" y2="12"></line>
            <polyline points="12 5 19 12 12 19"></polyline>
          </svg>
        `;

        const exportPayload = {
          exportDate: new Date().toISOString(),
          platform: 'CyberAlly',
          attackMode: selectedAttackMode,
          format: format,
          datasetsIncluded: exportOptions,
          telemetryStats: {
            totalAttacks: 342,
            detectionRate: '89%',
            mitigationRate: '76%',
            negativePenaltyScore: -125,
            avgResponseSeconds: 1.8,
            activeHosts: 10,
            compromisedHosts: 1
          },
          penaltyBreakdown: {
            criticalHostCompromise: -60,
            criticalBreaches: -40,
            dataExfiltration: -25
          },
          exportSummary: 'Autonomous CyberAlly environment telemetry package verified.'
        };

        const filename = `CyberAlly_Export_${Date.now()}.${format.toLowerCase()}`;
        
        if (format.toUpperCase() === 'PDF') {
          const pdfBlob = generateCyberAllyPDF(
            'CyberAlly Simulation Report',
            selectedAttackMode === 'llm' ? 'LLM Red Team Agent' : 'Scripted Attacks',
            -125,
            exportOptions
          );
          downloadFile(filename, pdfBlob, 'application/pdf');
        } else if (format.toUpperCase() === 'CSV') {
          const csvLines = [
            'Category,Metric,Value',
            `Metadata,Platform,CyberAlly`,
            `Metadata,Date,"${new Date().toISOString()}"`,
            `Metadata,Attack Mode,"${selectedAttackMode === 'llm' ? 'LLM Red Team Agent' : 'Scripted Attacks'}"`,
            `Performance,Total Attacks,342`,
            `Performance,Detection Rate,89%`,
            `Performance,Mitigation Rate,76%`,
            `Penalties,Negative Penalty Score,-125`,
            `Penalties,Critical Host Compromise,-60`,
            `Penalties,Critical Breaches,-40`,
            `Penalties,Data Exfiltration,-25`,
            `Network,Active Hosts,10`,
            `Network,Compromised Hosts,1`
          ];
          downloadFile(filename, csvLines.join('\n'), 'text/csv');
        } else if (format.toUpperCase() === 'PCAP') {
          // Standard PCAP Global Header (24 bytes) + packet record
          const pcapHeader = new Uint8Array([
            0xd4, 0xc3, 0xb2, 0xa1, // Magic number
            0x02, 0x00, 0x04, 0x00, // Version 2.4
            0x00, 0x00, 0x00, 0x00, // GMT to local correction
            0x00, 0x00, 0x00, 0x00, // Accuracy of timestamps
            0xff, 0xff, 0x00, 0x00, // Max length of captured packets
            0x01, 0x00, 0x00, 0x00  // Data link type (Ethernet)
          ]);
          downloadFile(filename, new Blob([pcapHeader], { type: 'application/vnd.tcpdump.pcap' }), 'application/vnd.tcpdump.pcap');
        } else {
          downloadFile(filename, JSON.stringify(exportPayload, null, 2), 'application/json');
        }
        
        showToast(`Successfully exported ${exportOptions.length} datasets in ${format} format!`, 'success');
      }, 700);
    });
  }

  // Authentic Valid PDF-1.4 Document Generator (Opens natively in Edge, Chrome, Safari, Acrobat)
  function generateCyberAllyPDF(title, attackMode, penaltyScore, datasets) {
    const lines = [
      "BT",
      "/F2 20 Tf",
      "50 780 Td",
      "(CYBERALLY - SIMULATION & TELEMETRY REPORT) Tj",
      "ET",
      "q",
      "0.22 0.54 0.98 rg",
      "50 765 495 2 re f",
      "Q",
      "BT",
      "/F2 11 Tf",
      "50 740 Td",
      `(${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })} | Platform: CyberAlly Autonomous Cyber-Range) Tj`,
      "/F1 10 Tf",
      "0 -18 Td",
      `(Attack Mode: ${attackMode} | Active Hosts: 10 | Target: Production Sandbox) Tj`,
      "ET",
      "q",
      "0.92 0.95 0.99 rg",
      "50 640 495 65 re f",
      "0.7 0.8 0.95 RG",
      "1 w",
      "50 640 495 65 re S",
      "Q",
      "BT",
      "/F2 11 Tf",
      "65 685 Td",
      "(EXECUTIVE METRICS) Tj",
      "/F1 10 Tf",
      "0 -18 Td",
      "(Total Attacks: 342    |    Detected: 89%    |    Contained: 76%) Tj",
      "0 -16 Td",
      `(Negative Penalty Score: ${penaltyScore}    |    Status: Completed) Tj`,
      "ET",
      "q",
      "0.99 0.92 0.92 rg",
      "50 535 495 85 re f",
      "0.9 0.6 0.6 RG",
      "1 w",
      "50 535 495 85 re S",
      "Q",
      "BT",
      "/F2 11 Tf",
      "65 600 Td",
      "(NEGATIVE SCORING & PENALTY ANALYSIS) Tj",
      "/F1 10 Tf",
      "0 -18 Td",
      "(- Critical Host Compromise: -60 pts) Tj",
      "0 -15 Td",
      "(- Critical Breaches:        -40 pts) Tj",
      "0 -15 Td",
      "(- Data Exfiltration:       -25 pts) Tj",
      "ET",
      "BT",
      "/F2 11 Tf",
      "50 500 Td",
      "(EXPORTED TELEMETRY DATASETS) Tj",
      "/F1 10 Tf"
    ];

    let yOffset = -18;
    datasets.forEach((d) => {
      lines.push(`0 ${yOffset} Td`);
      lines.push(`([x] ${String(d).replace(/[()\\]/g, '')}) Tj`);
      yOffset = -15;
    });

    lines.push("ET");
    lines.push("q");
    lines.push("0.8 0.8 0.8 rg");
    lines.push("50 60 495 1 re f");
    lines.push("Q");
    lines.push("BT");
    lines.push("/F1 8 Tf");
    lines.push("50 45 Td");
    lines.push("(Confidential - CyberAlly Autonomous Security Platform - Simulation Artifact) Tj");
    lines.push("ET");

    const streamContent = lines.join("\n");
    const streamLen = streamContent.length;

    const objects = [];
    objects.push("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    objects.push("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    objects.push(
      "3 0 obj\n" +
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842]\n" +
      "   /Contents 4 0 R\n" +
      "   /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >>\n" +
      ">>\nendobj\n"
    );
    objects.push(
      `4 0 obj\n<< /Length ${streamLen} >>\nstream\n${streamContent}\nendstream\nendobj\n`
    );
    objects.push("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");
    objects.push("6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n");

    const header = "%PDF-1.4\n%\xE2\xE3\xCF\xD3\n";
    let out = header;
    const offsets = [0];
    let curr = header.length;

    for (let i = 0; i < objects.length; i++) {
      offsets.push(curr);
      curr += objects[i].length;
      out += objects[i];
    }

    const xrefOffset = curr;
    out += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
    for (let j = 1; j < offsets.length; j++) {
      const offStr = String(offsets[j]).padStart(10, '0');
      out += `${offStr} 00000 n \n`;
    }

    out += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;

    const buf = new Uint8Array(out.length);
    for (let k = 0; k < out.length; k++) {
      buf[k] = out.charCodeAt(k) & 0xff;
    }
    return new Blob([buf], { type: 'application/pdf' });
  }

  // Generic Download Helper
  function downloadFile(filename, content, mimeType) {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    }, 100);
  }

  // Expose globally for testing & external calls
  window.generateCyberAllyPDF = generateCyberAllyPDF;

  // =========================================================================
  // DEMO & DOCUMENTATION MODALS
  // =========================================================================
  const heroWatchDemoBtn = document.getElementById('hero-watch-demo-btn');
  const btnDocumentation = document.getElementById('btn-documentation');

  if (heroWatchDemoBtn) {
    heroWatchDemoBtn.addEventListener('click', () => {
      demoModal.classList.add('active');
    });
  }

  if (btnDocumentation) {
    btnDocumentation.addEventListener('click', () => {
      docModal.classList.add('active');
    });
  }

  // Modal Closers
  function closeAllModals() {
    document.querySelectorAll('.modal-overlay').forEach((modal) => {
      modal.classList.remove('active');
    });
  }

  document.querySelectorAll('.btn-close-modal, #btn-modal-close, #btn-close-demo, #btn-close-doc').forEach((btn) => {
    btn.addEventListener('click', closeAllModals);
  });

  document.querySelectorAll('.modal-overlay').forEach((modal) => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeAllModals();
      }
    });
  });

  // Modal Download JSON button
  const btnModalDownloadJson = document.getElementById('btn-modal-download-json');
  if (btnModalDownloadJson) {
    btnModalDownloadJson.addEventListener('click', () => {
      const data = reportData['1'];
      downloadFile('Simulation_Report_Run_5.json', JSON.stringify(data, null, 2), 'application/json');
      showToast('Simulation Report downloaded successfully', 'success');
    });
  }

  // =========================================================================
  // THEME TOGGLING (Dark / Light)
  // =========================================================================
  function toggleTheme() {
    const isDark = document.body.classList.toggle('dark-theme');
    document.body.classList.toggle('light-theme', !isDark);

    if (moonIcon && sunIcon) {
      moonIcon.style.display = isDark ? 'block' : 'none';
      sunIcon.style.display = isDark ? 'none' : 'block';
    }
    showToast(`Switched to ${isDark ? 'Dark Neon' : 'Light Contrast'} mode`, 'info');
  }

  if (homeThemeToggle) homeThemeToggle.addEventListener('click', toggleTheme);
  if (dashThemeToggle) dashThemeToggle.addEventListener('click', toggleTheme);

  // =========================================================================
  // TOAST NOTIFICATION SYSTEM
  // =========================================================================
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let iconSvg = '';
    if (type === 'success') {
      iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
    } else if (type === 'warning') {
      iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f97316" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
    } else {
      iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;
    }

    toast.innerHTML = `${iconSvg}<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(15px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => {
        if (toast.parentNode) {
          toast.parentNode.removeChild(toast);
        }
      }, 300);
    }, 3200);
  }

  // Check initial hash route on page load (e.g., #config, #live-monitoring)
  handleHashRoute();
});
