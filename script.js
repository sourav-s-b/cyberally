/**
 * CyberRangeAI - Core Frontend Controller
 * Matches Homepage & Dashboard Screens (Panels 2, 3, 4, 5)
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- View Elements ---
  const homeView = document.getElementById('home-view');
  const dashboardView = document.getElementById('dashboard-view');
  
  // --- Navigation Elements ---
  const startSimButtons = document.querySelectorAll('.start-sim-btn');
  const dashNavItems = document.querySelectorAll('.dash-nav-item');
  const homeLogoBtn = document.getElementById('home-logo-btn');
  const sidebarBrandBtn = document.getElementById('sidebar-brand-btn');
  
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
  const liveActiveAttacks = document.getElementById('live-active-attacks');
  const liveActiveDefenses = document.getElementById('live-active-defenses');
  const liveTotalEvents = document.getElementById('live-total-events');
  const liveEventsTbody = document.getElementById('live-events-tbody');
  
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
  // VIEW SWITCHING (Home <-> Dashboard)
  // =========================================================================
  function showView(viewId) {
    if (viewId === 'dashboard') {
      homeView.classList.remove('active');
      dashboardView.classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      dashboardView.classList.remove('active');
      homeView.classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  // "Start Simulation" buttons transition to Dashboard -> Live Monitoring
  startSimButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      closeAllModals();
      showView('dashboard');
      activateDashTab('live-monitoring');
      showToast('Entered Autonomous Cyber-Range Simulation', 'success');
    });
  });

  // Clicking Logo returns to Home
  if (homeLogoBtn) {
    homeLogoBtn.addEventListener('click', (e) => {
      e.preventDefault();
      showView('home');
    });
  }
  if (sidebarBrandBtn) {
    sidebarBrandBtn.addEventListener('click', () => {
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
      desc: 'Performance metrics, learning progress and attack/defense insights'
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
        simStatusBadge.innerHTML = `<span class="pulse-dot"></span><span class="status-text">Simulation Running</span>`;
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

  // Periodic simulated events ticker to make the dashboard feel alive
  let simEventCount = 128;
  const attackTypes = ['SQL injection attempt', 'Lateral movement probing', 'Remote Code Execution (RCE)', 'Credential brute-force'];
  const defenseTypes = ['Blocked suspicious traffic', 'Reinforced authentication token', 'Isolated compromised container', 'Applied zero-day egress filter'];

  setInterval(() => {
    if (!isSimRunning || !dashboardView.classList.contains('active')) return;
    
    // 25% chance every tick to trigger a simulated background telemetry event
    if (Math.random() > 0.65) {
      simEventCount++;
      if (liveTotalEvents) liveTotalEvents.textContent = simEventCount;

      const isAttack = Math.random() > 0.45;
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];

      const tr = document.createElement('tr');
      if (isAttack) {
        const desc = attackTypes[Math.floor(Math.random() * attackTypes.length)];
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
        // keep table at reasonable length
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
    node.addEventListener('mouseenter', (e) => {
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
      const containerRect = document.getElementById('network-topology-container').getBoundingClientRect();
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
      score: '91.4%',
      attempts: 48,
      mitigations: '44 (91.7%)',
      mtt: '1.2s / 2.4s',
      summary: 'Red Team agents launched chained multi-stage penetration attacks against the DMZ and internal microservices. The decentralized MARL Blue Team autonomously deployed host isolation, firewall policy updates, and token reinforcement within an average containment time of 1.8 seconds.'
    },
    '2': {
      title: 'Attack Analysis Report',
      date: 'Oct 17, 2026',
      duration: '45 min',
      score: '88.2%',
      attempts: 64,
      mitigations: '57 (89.1%)',
      mtt: '1.5s / 2.8s',
      summary: 'Evaluation of LLM-generated attack payloads utilizing dynamic prompt mutation. 57 out of 64 exploitation trajectories were intercepted prior to privilege escalation.'
    },
    '3': {
      title: 'Defense Performance',
      date: 'Oct 16, 2026',
      duration: '30 min',
      score: '94.0%',
      attempts: 40,
      mitigations: '38 (95.0%)',
      mtt: '1.0s / 1.9s',
      summary: 'Evaluation of cooperative multi-agent PPO policies on coalition subnet defense. Fast convergence and minimal cross-zone policy conflict observed.'
    },
    '4': {
      title: 'System Vulnerabilities',
      date: 'Oct 15, 2026',
      duration: '25 min',
      score: '86.5%',
      attempts: 32,
      mitigations: '28 (87.5%)',
      mtt: '1.8s / 3.1s',
      summary: 'Synthetic vulnerability scanning report detailing CVE distribution in JuiceShop, DVWA, and internal RPC microservices.'
    },
    '5': {
      title: 'Simulation Report - Run 4',
      date: 'Oct 14, 2026',
      duration: '40 min',
      score: '89.7%',
      attempts: 52,
      mitigations: '47 (90.4%)',
      mtt: '1.4s / 2.6s',
      summary: 'Baseline comparative run between static heuristic defense and decentralized multi-agent reinforcement learning defense.'
    }
  };

  // Sub-tabs in Reports (Simulation Reports vs Performance Reports)
  const reportSubtabs = document.querySelectorAll('.reports-subtab');
  reportSubtabs.forEach((st) => {
    st.addEventListener('click', () => {
      reportSubtabs.forEach((s) => s.classList.remove('active'));
      st.classList.add('active');
      showToast(`Showing: ${st.textContent}`, 'info');
    });
  });

  // View Report Click
  document.querySelectorAll('.btn-view-report').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.id;
      const data = reportData[id] || reportData['1'];
      
      document.getElementById('modal-report-title').textContent = data.title;
      document.getElementById('modal-report-meta').textContent = `Executed on ${data.date} | Duration: ${data.duration} | Status: Completed`;
      
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
        <td><span class="status-badge-pill badge-success">Completed</span></td>
        <td style="text-align: right;">
          <button class="btn-table-action btn-view-report" data-id="1">View</button>
          <button class="btn-table-action btn-download-report" data-id="1">Download</button>
        </td>
      `;

      reportsTbody.insertBefore(newRow, reportsTbody.firstChild);
      generateReportModal.classList.remove('active');
      showToast(`Report '${title}' successfully generated!`, 'success');

      // Re-attach view event for new element
      newRow.querySelector('.btn-view-report').addEventListener('click', () => {
        document.getElementById('modal-report-title').textContent = title;
        reportViewModal.classList.add('active');
      });
      newRow.querySelector('.btn-download-report').addEventListener('click', () => {
        downloadFile(`${title.replace(/\s+/g, '_')}.json`, JSON.stringify({ title, date: 'Oct 18, 2026', status: 'Completed' }, null, 2), 'application/json');
        showToast(`Downloaded: ${title}`, 'success');
      });
    });
  }

  // =========================================================================
  // EXPORT DATA FUNCTIONALITY (Panel 5 from image.png)
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
          format: format,
          datasetsIncluded: exportOptions,
          telemetryStats: {
            totalAttacks: 342,
            detectionRate: '89%',
            mitigationRate: '76%',
            avgResponseSeconds: 1.8,
            activeHosts: 10,
            compromisedHosts: 1
          },
          exportSummary: 'Autonomous CyberRangeAI environment telemetry package verified.'
        };

        const filename = `CyberRangeAI_Export_${Date.now()}.${format.toLowerCase()}`;
        downloadFile(filename, JSON.stringify(exportPayload, null, 2), 'text/plain');
        showToast(`Successfully exported ${exportOptions.length} datasets in ${format} format!`, 'success');
      }, 700);
    });
  }

  // Generic Download Helper
  function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
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
});
