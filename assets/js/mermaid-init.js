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
                    primaryColor: '#ff6b6b',
                    primaryTextColor: '#2c3e50',
                    primaryBorderColor: '#e74c3c',
                    lineColor: '#34495e',
                    secondaryColor: '#3498db',
                    tertiaryColor: '#f39c12'
                },
                flowchart: {
                    htmlLabels: true,
                    curve: 'basis'
                },
                sequence: {
                    showSequenceNumbers: true,
                    wrap: true
                },
                gantt: {
                    titleTopMargin: 25,
                    barHeight: 20,
                    fontFamily: 'ui-sans-serif, system-ui, sans-serif'
                }
            });
            
            log('Mermaid initialized with custom configuration');
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
    
})();