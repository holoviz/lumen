import Split from 'https://esm.sh/split.js@1.6.5'

export function render({ model, view }) {
    const splitDiv = document.createElement('div');
    splitDiv.className = 'split';
    splitDiv.classList.add('loading');

    const split0 = document.createElement('div');
    const split1 = document.createElement('div');
    splitDiv.append(split0, split1);

    // position and width now handled by CSS

    // Create content wrapper for right panel
    const contentWrapper = document.createElement('div');
    contentWrapper.classList.add('content-wrapper');

    // Create toggle icons for both sides of the divider
    const leftArrowButton = document.createElement('div');  // < button
    const rightArrowButton = document.createElement('div'); // > button

    // Left arrow button (<) - positioned on left side of divider
    leftArrowButton.className = 'toggle-button-left';
    leftArrowButton.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`; // < arrow

    // Right arrow button (>) - positioned on right side of divider
    rightArrowButton.className = 'toggle-button-right';
    rightArrowButton.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`; // > arrow
    
    // Add both buttons to the right panel (split1) so they're positioned relative to the divider
    split1.appendChild(leftArrowButton);
    split1.appendChild(rightArrowButton);

    // Determine initial state - collapsed means right panel is hidden
    const initSizes = model.collapsed ? [100, 0] : model.expanded_sizes;

    // Track click counts for toggle behavior
    let leftClickCount = 0;  // For < button
    let rightClickCount = 0; // For > button

    // Function to reset click counts when sizes change via dragging
    function resetClickCounts() {
        leftClickCount = 0;
        rightClickCount = 0;
    }

    // Use minSize of 0 to allow full collapse via buttons
    const splitInstance = Split([split0, split1], {
        sizes: initSizes,
        minSize: [0, 0], // Allow full collapse for both panels
        gutterSize: 8, // Match the 8px width in CSS
        onDragEnd: (sizes) => {
            view.invalidate_layout();

            // Determine the new collapsed state based on panel sizes
            const rightPanelCollapsed = sizes[1] <= 5;
            const leftPanelCollapsed = sizes[0] <= 5;
            
            // The model's collapsed state represents whether the right panel is collapsed
            const newCollapsedState = rightPanelCollapsed;

            if (model.collapsed !== newCollapsedState) {
                // Send message to Python about collapsed state change
                model.send_msg({ collapsed: newCollapsedState });
                model.collapsed = newCollapsedState;
            }

            // Update UI based on current sizes
            updateUIForCollapsedState(newCollapsedState, sizes);
            
            // Reset click counts when user drags the splitter
            resetClickCounts();
        },
    });

    // Function to update UI elements based on collapsed state
    function updateUIForCollapsedState(isCollapsed, sizes = null) {
        // Determine current panel state
        const leftPanelHidden = sizes ? sizes[0] <= 5 : false;
        const rightPanelHidden = sizes ? sizes[1] <= 5 : false;
        
        // Update content visibility
        if (rightPanelHidden) {
            contentWrapper.className = 'collapsed-content';
        } else {
            contentWrapper.className = 'content-wrapper';
        }
        
        if (leftPanelHidden) {
            leftContentWrapper.className = 'collapsed-content';
        } else {
            leftContentWrapper.className = 'left-content-wrapper';
        }
    }

    // Left arrow button (<) event listener - two-step toggle
    leftArrowButton.addEventListener('click', () => {
        leftClickCount++;
        rightClickCount = 0; // Reset other button's count
        
        let newSizes;
        
        if (leftClickCount === 1) {
            // First tap: make sizes 50, 50
            newSizes = [50, 50];
        } else {
            // Second tap (or more): make sizes 0, 100
            newSizes = [0, 100];
            leftClickCount = 0; // Reset after second tap
        }
        
        splitInstance.setSizes(newSizes);
        
        // Update collapsed state based on new sizes
        const newCollapsedState = newSizes[1] <= 5;
        model.send_msg({ collapsed: newCollapsedState });
        model.collapsed = newCollapsedState;
        
        updateUIForCollapsedState(newCollapsedState, newSizes);
        view.invalidate_layout();
    });

    // Right arrow button (>) event listener - two-step toggle
    rightArrowButton.addEventListener('click', () => {
        rightClickCount++;
        leftClickCount = 0; // Reset other button's count
        
        let newSizes;
        
        if (rightClickCount === 1) {
            // First tap: make sizes 50, 50
            newSizes = [50, 50];
        } else {
            // Second tap (or more): make sizes 100, 0
            newSizes = [100, 0];
            rightClickCount = 0; // Reset after second tap
        }
        
        splitInstance.setSizes(newSizes);
        
        // Update collapsed state based on new sizes
        const newCollapsedState = newSizes[1] <= 5;
        model.send_msg({ collapsed: newCollapsedState });
        model.collapsed = newCollapsedState;
        
        updateUIForCollapsedState(newCollapsedState, newSizes);
        view.invalidate_layout();
    });

    // Listen for collapsed state changes from Python
    model.on("msg:custom", (event) => {
        if (event.type === "update_collapsed") {
            const newCollapsedState = event.collapsed;

            model.collapsed = newCollapsedState;
            console.log('Received collapsed state update:', event.collapsed);

            // Update split sizes based on new collapsed state
            if (newCollapsedState) {
                // Collapse right panel (show only left)
                splitInstance.setSizes([100, 0]);
            } else {
                // Expand to show both panels
                splitInstance.setSizes(model.expanded_sizes);
            }

            // Update UI elements
            updateUIForCollapsedState(newCollapsedState);
            view.invalidate_layout();
        }
    });

    model.on("after_layout", () => {
        setTimeout(() => {
            splitDiv.classList.remove('loading');

            // Only add animation on initial load
            if (!window._toggleAnimationShown) {
                // Add animation on first load only
                leftArrowButton.classList.add('animated');
                rightArrowButton.classList.add('animated');

                // Remove animation after it completes and set flag
                setTimeout(() => {
                    leftArrowButton.classList.remove('animated');
                    rightArrowButton.classList.remove('animated');
                    window._toggleAnimationShown = true;
                }, 1500);
            }

            window.dispatchEvent(new Event('resize'));
        }, 100);
    });

    // Create a centered content wrapper for the left panel
    const leftContentWrapper = document.createElement('div');
    leftContentWrapper.classList.add('left-content-wrapper');

    // Set initial display based on collapsed state
    if (model.collapsed) {
        contentWrapper.className = 'collapsed-content';
    }

    // Apply left-panel-content class to the left panel
    leftContentWrapper.classList.add('left-panel-content');

    // Append children to the appropriate containers
    leftContentWrapper.append(model.get_child("left"));
    split0.append(leftContentWrapper);
    contentWrapper.append(model.get_child("right"));
    split1.append(contentWrapper);

    return splitDiv;
}
