import Split from 'https://esm.sh/split.js@1.6.5'

export function render({ model, view }) {
    const splitDiv = document.createElement('div');
    splitDiv.className = 'split';
    splitDiv.style.visibility = 'hidden';

    const split0 = document.createElement('div');
    const split1 = document.createElement('div');
    splitDiv.append(split0, split1);

    split1.style.position = 'relative';
    split1.style.width = '100%';

    // Create content wrapper for right panel
    const contentWrapper = document.createElement('div');
    contentWrapper.style.width = '100%';
    contentWrapper.style.height = '100%';
    contentWrapper.style.display = 'block';

    // Apply background color based on invert parameter
    if (model.invert) {
    split0.style.backgroundColor = 'whitesmoke';
    } else {
    contentWrapper.style.backgroundColor = 'whitesmoke';
    }

    // Create toggle icon for panel toggling
    const toggleIcon = document.createElement('div');

    // Set class and initial arrow direction based on invert parameter
    if (model.invert) {
    toggleIcon.className = 'toggle-icon-inverted';
    toggleIcon.innerHTML = model.collapsed
        ? `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`  // Right arrow when collapsed
        : `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`;  // Left arrow when expanded
    // Set initial position based on collapsed state
    toggleIcon.style.right = model.collapsed ? '-30px' : '5px';
    } else {
    toggleIcon.className = 'toggle-icon';
    toggleIcon.innerHTML = model.collapsed
        ? `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`  // Left arrow when collapsed
        : `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`;  // Right arrow when expanded
    // Set initial position based on collapsed state
    toggleIcon.style.left = model.collapsed ? '-30px' : '5px';
    }

    // Determine which panel gets the toggle based on invert parameter
    const togglePanel = model.invert ? split0 : split1;
    const toggleTarget = model.invert ? split1 : split0;

    // Add the toggle icon to the appropriate panel but positioned relative to the gutter
    togglePanel.style.position = 'relative'; // Ensure the container has relative positioning
    togglePanel.appendChild(toggleIcon);

    // Determine initial state based on the invert parameter
    let initSizes;
    if (model.invert) {
    // In inverted mode, right panel is shown first
    initSizes = model.collapsed ? [0, 100] : model.expanded_sizes;
    } else {
    // In normal mode, left panel is shown first
    initSizes = model.collapsed ? [100, 0] : model.expanded_sizes;
    }

    const splitInstance = Split([split0, split1], {
    sizes: initSizes,
    minSize: model.min_sizes,
    gutterSize: 6,
    onDragEnd: (sizes) => {
        view.invalidate_layout();

        // Update collapsed state based on panel size and invert parameter
        const newCollapsedState = model.invert ? sizes[0] <= 5 : sizes[1] <= 5;

        if (model.collapsed !== newCollapsedState) {
        // Send message to Python about collapsed state change
        model.send_msg({ collapsed: newCollapsedState });

        model.collapsed = newCollapsedState;

        // Update UI based on new collapsed state
        updateUIForCollapsedState(newCollapsedState);
        }
    },
    });

    // Function to update UI elements based on collapsed state
    function updateUIForCollapsedState(isCollapsed) {
    if (isCollapsed) {
        // Collapsed state UI updates
        if (model.invert) {
        toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`; // Right arrow
        toggleIcon.style.right = '-30px'; // Adjust position when collapsed
        // Hide left content (which is the target content in inverted mode)
        leftContentWrapper.className = 'collapsed-content';
        leftContentWrapper.style.display = 'none';
        } else {
        toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`; // Left arrow
        toggleIcon.style.left = '-30px'; // Adjust position when collapsed
        contentWrapper.className = 'collapsed-content';
        contentWrapper.style.display = 'none';
        }
    } else {
        // Expanded state UI updates
        if (model.invert) {
        leftContentWrapper.className = '';
        leftContentWrapper.style.display = 'block';
        toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`; // Left arrow
        toggleIcon.style.right = '5px'; // Adjust position when expanded
        } else {
        contentWrapper.className = '';
        contentWrapper.style.display = 'block';
        toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`; // Right arrow
        toggleIcon.style.left = '5px'; // Adjust position when expanded
        }
    }
    }

    // Toggle button event listener
    toggleIcon.addEventListener('click', () => {
    const newCollapsedState = !model.collapsed;

    if (newCollapsedState) {
        // Collapse with appropriate sizes based on invert parameter
        if (model.invert) {
        splitInstance.setSizes([0, 100]);
        } else {
        splitInstance.setSizes([100, 0]);
        }
    } else {
        // Expand
        splitInstance.setSizes(model.expanded_sizes);
    }

    // Send message to Python about collapsed state change
    model.send_msg({ collapsed: newCollapsedState });

    // Update model and UI
    model.collapsed = newCollapsedState;
    updateUIForCollapsedState(newCollapsedState);

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
        // Collapse
        if (model.invert) {
            splitInstance.setSizes([0, 100]);
        } else {
            splitInstance.setSizes([100, 0]);
        }
        } else {
        // Expand
        splitInstance.setSizes(model.expanded_sizes);
        }

        // Update UI elements
        updateUIForCollapsedState(newCollapsedState);
        view.invalidate_layout();
    }
    });

    model.on("after_layout", () => {
    setTimeout(() => {
        splitDiv.style.visibility = 'visible';

        // Only add animation on initial load
        if (!window._toggleAnimationShown) {
        // Add animation on first load only
        toggleIcon.style.animationName = 'jumpLeftRight';
        toggleIcon.style.animationDuration = '0.5s';
        toggleIcon.style.animationTimingFunction = 'ease';
        toggleIcon.style.animationIterationCount = '3';

        // Remove animation after it completes and set flag
        setTimeout(() => {
            toggleIcon.style.animationName = '';
            toggleIcon.style.animationDuration = '';
            toggleIcon.style.animationTimingFunction = '';
            toggleIcon.style.animationIterationCount = '';
            window._toggleAnimationShown = true;
        }, 1500);
        }

        window.dispatchEvent(new Event('resize'));
    }, 100);
    });

    // Create a centered content wrapper for the left panel
    const leftContentWrapper = document.createElement('div');

    // Set initial display based on collapsed state and invert parameter
    if (model.collapsed) {
    if (model.invert) {
        leftContentWrapper.className = 'collapsed-content';
        leftContentWrapper.style.display = 'none';
    } else {
        contentWrapper.className = 'collapsed-content';
        contentWrapper.style.display = 'none';
    }
    }

    // Apply left-panel-content class to the appropriate panel based on invert parameter
    if (model.invert) {
    contentWrapper.className += ' left-panel-content';
    } else {
    leftContentWrapper.className = 'left-panel-content';
    }

    leftContentWrapper.style.width = '100%';
    leftContentWrapper.style.height = '100%';
    leftContentWrapper.append(model.get_child("left"));
    split0.append(leftContentWrapper);
    contentWrapper.append(model.get_child("right"));
    split1.append(contentWrapper);

    return splitDiv;
}
