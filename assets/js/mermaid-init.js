/**
 * Mermaid Initialization Script for Jekyll Blog
 * Handles both raw <div class="mermaid"> and ```mermaid code blocks
 * Provides robust error handling and fallback mechanisms
 */

(function() {
    'use strict';

    // Configuration
    const MERMAID_CDN = 'https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js';
    const DEBUG = false; // Set to true for debugging

    // Logging utility
    function log(...args) {
        if (DEBUG && window.console) {
            console.debug('[Mermaid Init]', ...args);
        }
    }

    // Error handling
    function handleError(message, error) {
        if (window.console) {
            console.warn('[Mermaid Init]', message, error);
        }
    }

    // Load Mermaid library
    function loadMermaid(callback) {
        if (window.mermaid) {
            log('Mermaid already loaded');
            return callback();
        }

        log('Loading Mermaid from CDN');
        const script = document.createElement('script');
        script.src = MERMAID_CDN;
        script.onload = () => {
            log('Mermaid loaded successfully');
            callback();
        };
        script.onerror = () => {
            handleError('Failed to load Mermaid script from CDN');
        };
        document.head.appendChild(script);
    }

    // Convert code blocks to mermaid divs
    function convertCodeBlocks() {
        const codeBlocks = document.querySelectorAll('pre code.language-mermaid, pre code.mermaid');
        let converted = 0;

        codeBlocks.forEach((codeEl, index) => {
            try {
                const code = codeEl.textContent.trim();
                if (!code) return;

                const wrapper = document.createElement('div');
                wrapper.className = 'mermaid';
                wrapper.textContent = code;
                wrapper.setAttribute('data-converted', 'true');

                codeEl.parentNode.replaceWith(wrapper);
                converted++;
                log(`Converted code block ${index + 1} to mermaid div`);
            } catch (error) {
                handleError(`Failed to convert code block ${index + 1}`, error);
            }
        });

        log(`Converted ${converted} code blocks to mermaid divs`);
        return converted;
    }

    // Render all mermaid diagrams
    function renderDiagrams() {
        if (!window.mermaid) {
            handleError('Mermaid not available for rendering');
            return;
        }

        const diagrams = document.querySelectorAll('div.mermaid');
        let rendered = 0;

        diagrams.forEach((element, index) => {
            try {
                const code = element.textContent.trim();
                if (!code) return;

                // Skip if already rendered
                if (element.getAttribute('data-processed') === 'true') {
                    return;
                }

                const id = `mermaid-diagram-${Date.now()}-${index}`;

                window.mermaid.mermaidAPI.render(id, code, (svg, bindFunctions) => {
                    element.innerHTML = svg;
                    element.setAttribute('data-processed', 'true');

                    // Execute any binding functions if available
                    if (typeof bindFunctions === 'function') {
                        bindFunctions(element);
                    }

                    rendered++;
                    log(`Rendered diagram ${index + 1}/${diagrams.length}`);
                }, element);

            } catch (error) {
                handleError(`Failed to render diagram ${index + 1}`, error);
                element.innerHTML = `<div class="mermaid-error">Failed to render diagram</div>`;
            }
        });

        log(`Attempted to render ${diagrams.length} diagrams, ${rendered} successful`);
    }

    // Initialize Mermaid with configuration
    function initializeMermaid() {
        try {
            window.mermaid.initialize({
                startOnLoad: false,
                theme: 'default',
                securityLevel: 'loose',
                fontFamily: 'ui-sans-serif, system-ui, -apple-system, sans-serif',
                fontSize: 16,
                logLevel: DEBUG ? 'debug' : 'error',
                themeVariables: {
                    // Network-focused color scheme
                    primaryColor: '#1e3a8a',        // Deep blue for primary nodes
                    primaryTextColor: '#ffffff',     // White text on primary
                    primaryBorderColor: '#1e40af',  // Border for primary elements
                    lineColor: '#6b7280',           // Gray for connections
                    secondaryColor: '#059669',      // Green for secondary nodes
                    tertiaryColor: '#dc2626',       // Red for critical/error states
                    background: '#f8fafc',          // Light background
                    mainBkg: '#ffffff',             // White background for nodes
                    secondBkg: '#f1f5f9',          // Light gray secondary background
                    tertiaryColor: '#fbbf24',       // Amber for warnings

                    // Network-specific colors
                    pie1: '#1e3a8a',               // Router/Gateway
                    pie2: '#059669',               // Switch/Network
                    pie3: '#dc2626',               // Firewall/Security
                    pie4: '#7c3aed',               // Server/Service
                    pie5: '#ea580c',               // Storage/Database
                    pie6: '#0891b2',               // Client/Endpoint
                    pie7: '#65a30d',               // Monitoring/Management
                    pie8: '#be185d',               // Critical/Alert

                    // Enhanced contrast for readability
                    edgeLabelBackground: '#ffffff',
                    clusterBkg: '#f8fafc',
                    clusterBorder: '#cbd5e1',
                    defaultLinkColor: '#6b7280',
                    titleColor: '#1f2937',

                    // Git/version control colors for flow diagrams
                    git0: '#1e3a8a',
                    git1: '#059669',
                    git2: '#dc2626',
                    git3: '#7c3aed',
                    git4: '#ea580c',
                    git5: '#0891b2',
                    git6: '#65a30d',
                    git7: '#be185d'
                },
                flowchart: {
                    htmlLabels: true,
                    curve: 'basis',
                    padding: 20,
                    nodeSpacing: 100,
                    rankSpacing: 100,
                    diagramPadding: 20,
                    useMaxWidth: true
                },
                sequence: {
                    showSequenceNumbers: true,
                    wrap: true,
                    width: 200,
                    height: 65,
                    boxMargin: 10,
                    boxTextMargin: 5,
                    noteMargin: 10,
                    messageMargin: 35,
                    mirrorActors: true,
                    diagramPadding: 20
                },
                gantt: {
                    titleTopMargin: 25,
                    barHeight: 20,
                    fontFamily: 'ui-sans-serif, system-ui, sans-serif',
                    fontSize: 14,
                    gridLineStartPadding: 35,
                    bottomPadding: 25,
                    leftPadding: 120,
                    rightPadding: 40
                },
                class: {
                    titleTopMargin: 25,
                    diagramPadding: 20,
                    htmlLabels: true
                },
                state: {
                    titleTopMargin: 25,
                    diagramPadding: 20,
                    forkWidth: 70,
                    forkHeight: 7
                },
                er: {
                    titleTopMargin: 25,
                    diagramPadding: 20,
                    layoutDirection: 'TB',
                    minEntityWidth: 100,
                    minEntityHeight: 75,
                    entityPadding: 15,
                    stroke: '#6b7280',
                    fill: '#f8fafc'
                },
                pie: {
                    titleTopMargin: 25,
                    diagramPadding: 20,
                    legendPosition: 'right'
                },
                requirement: {
                    titleTopMargin: 25,
                    diagramPadding: 20,
                    color: '#1f2937',
                    fillColor: '#f8fafc',
                    fontSize: 14
                }
            });

            log('Mermaid initialized with network-optimized configuration');
        } catch (error) {
            handleError('Failed to initialize Mermaid', error);
        }
    }

    // Main processing function
    function processMermaidDiagrams() {
        log('Starting Mermaid diagram processing');

        // Convert code blocks first
        const converted = convertCodeBlocks();

        // Initialize Mermaid
        initializeMermaid();

        // Render all diagrams
        renderDiagrams();

        log('Mermaid processing completed');
    }

    // Mutation observer for dynamic content
    function observeDynamicContent() {
        if (!window.MutationObserver) return;

        const observer = new MutationObserver((mutations) => {
            let hasNewContent = false;

            mutations.forEach((mutation) => {
                if (mutation.addedNodes.length > 0) {
                    for (let node of mutation.addedNodes) {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            if (node.querySelector &&
                                (node.querySelector('div.mermaid') ||
                                 node.querySelector('code.language-mermaid') ||
                                 node.querySelector('code.mermaid'))) {
                                hasNewContent = true;
                                break;
                            }
                        }
                    }
                }
            });

            if (hasNewContent) {
                log('New mermaid content detected, reprocessing');
                setTimeout(processMermaidDiagrams, 100);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        log('Mutation observer initialized for dynamic content');
    }

    // Main initialization
    function initialize() {
        loadMermaid(() => {
            processMermaidDiagrams();
            observeDynamicContent();
        });
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }

    // Expose global function for manual triggering
    window.refreshMermaidDiagrams = processMermaidDiagrams;

    // Network diagram helper functions
    function initializeNetworkHelpers() {
        log('Initializing network diagram helpers...');
        
        // Add custom event listener for post-render processing
        document.addEventListener('mermaidRenderComplete', function() {
            addNetworkComponentClasses();
            addNetworkLegends();
            enhanceNetworkInteractivity();
        });
    }

    // Add CSS classes to network components based on labels
    function addNetworkComponentClasses() {
        const nodes = document.querySelectorAll('.mermaid .node');
        
        nodes.forEach(node => {
            const textElement = node.querySelector('text');
            if (!textElement) return;
            
            const text = textElement.textContent.toLowerCase();
            
            // Apply network component classes based on text content
            if (text.includes('router') || text.includes('라우터')) {
                node.classList.add('network-router');
            } else if (text.includes('switch') || text.includes('스위치')) {
                node.classList.add('network-switch');
            } else if (text.includes('firewall') || text.includes('방화벽')) {
                node.classList.add('network-firewall');
            } else if (text.includes('server') || text.includes('서버')) {
                node.classList.add('network-server');
            } else if (text.includes('database') || text.includes('db') || text.includes('데이터베이스')) {
                node.classList.add('network-database');
            } else if (text.includes('client') || text.includes('클라이언트')) {
                node.classList.add('network-client');
            } else if (text.includes('cloud') || text.includes('클라우드')) {
                node.classList.add('network-cloud');
            } else if (text.includes('gateway') || text.includes('게이트웨이')) {
                node.classList.add('network-gateway');
            }
        });
    }

    // Add legends to network diagrams
    function addNetworkLegends() {
        const networkDiagrams = document.querySelectorAll('.mermaid');
        
        networkDiagrams.forEach(diagram => {
            // Check if this is a network diagram (contains network components)
            const hasNetworkComponents = diagram.querySelector('.network-router, .network-switch, .network-firewall, .network-server');
            
            if (hasNetworkComponents && !diagram.querySelector('.network-legend')) {
                const legend = createNetworkLegend();
                diagram.style.position = 'relative';
                diagram.appendChild(legend);
            }
        });
    }

    // Create network legend
    function createNetworkLegend() {
        const legend = document.createElement('div');
        legend.className = 'network-legend';
        
        legend.innerHTML = `
            <h4>Network Components</h4>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #1e3a8a;"></div>
                <span>Router</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #059669;"></div>
                <span>Switch</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #dc2626;"></div>
                <span>Firewall</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #7c3aed;"></div>
                <span>Server</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #f59e0b;"></div>
                <span>Database</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #0ea5e9;"></div>
                <span>Cloud</span>
            </div>
        `;
        
        return legend;
    }

    // Enhance network diagram interactivity
    function enhanceNetworkInteractivity() {
        const nodes = document.querySelectorAll('.mermaid .node');
        
        nodes.forEach(node => {
            // Add tooltips for network components
            node.addEventListener('mouseenter', function() {
                showNetworkTooltip(this);
            });
            
            node.addEventListener('mouseleave', function() {
                hideNetworkTooltip();
            });
        });
    }

    // Show network component tooltip
    function showNetworkTooltip(node) {
        const textElement = node.querySelector('text');
        if (!textElement) return;
        
        const tooltip = document.createElement('div');
        tooltip.className = 'network-tooltip';
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            white-space: nowrap;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;
        
        const text = textElement.textContent;
        const componentType = getNetworkComponentType(text);
        tooltip.textContent = `${componentType}: ${text}`;
        
        document.body.appendChild(tooltip);
        
        // Position tooltip
        const rect = node.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width / 2 - tooltip.offsetWidth / 2}px`;
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 8}px`;
    }

    // Hide network tooltip
    function hideNetworkTooltip() {
        const tooltip = document.querySelector('.network-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }

    // Get network component type
    function getNetworkComponentType(text) {
        const lowerText = text.toLowerCase();
        
        if (lowerText.includes('router') || lowerText.includes('라우터')) return 'Router';
        if (lowerText.includes('switch') || lowerText.includes('스위치')) return 'Switch';
        if (lowerText.includes('firewall') || lowerText.includes('방화벽')) return 'Firewall';
        if (lowerText.includes('server') || lowerText.includes('서버')) return 'Server';
        if (lowerText.includes('database') || lowerText.includes('db') || lowerText.includes('데이터베이스')) return 'Database';
        if (lowerText.includes('client') || lowerText.includes('클라이언트')) return 'Client';
        if (lowerText.includes('cloud') || lowerText.includes('클라우드')) return 'Cloud Service';
        if (lowerText.includes('gateway') || lowerText.includes('게이트웨이')) return 'Gateway';
        
        return 'Network Component';
    }

    // Initialize network helpers after DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeNetworkHelpers);
    } else {
        initializeNetworkHelpers();
    }

})();