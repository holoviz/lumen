import param

from panel.custom import Child, JSComponent

CSS = """
/* Max width for comfortable reading */
.left-panel-content {
    max-width: clamp(400px, 100vw, 1500px);
    margin: 0 auto;
    padding: 0 20px;
    box-sizing: border-box;
}

@keyframes jumpLeftRight {
    0%, 100% { transform: translateY(-50%); }
    25% { transform: translate(-4px, -50%); }
    50% { transform: translateY(-50%); }
    75% { transform: translate(4px, -50%); }
}

.split {
    display: flex;
    flex-direction: row;
    height: 100%;
    width: 100%;
}

.gutter {
    background-color: var(--panel-surface-color);
    background-repeat: no-repeat;
    background-position: 50%;
    cursor: col-resize;
}

.gutter.gutter-horizontal {
    background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAeCAYAAADkftS9AAAAIklEQVQoU2M4c+bMfxAGAgYYmwGrIIiDjrELjpo5aiZeMwF+yNnOs5KSvgAAAABJRU5ErkJggg==');
    z-index: 1;
}

ul.nav.flex-column {
    padding-inline-start: 0 !important;
    margin: 0 !important;
}

.toggle-icon {
    position: absolute;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 10;
    opacity: 0.65;
    transition: opacity 0.2s;
    left: -30px; /* Position it to the left of the gutter with more offset */
    top: 50%;
    transform: translateY(-50%);
}

.toggle-icon-inverted {
    position: absolute;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 10;
    opacity: 0.65;
    transition: opacity 0.2s;
    right: -30px; /* Position it to the right of the gutter with offset */
    top: 50%;
    transform: translateY(-50%);
}

.toggle-icon:hover, .toggle-icon-inverted:hover {
    opacity: 1;
}

.toggle-icon svg, .toggle-icon-inverted svg {
    width: 50px;
    height: 50px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2px;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.collapsed-content {
    display: none !important;
}

/* Ensure the right panel and its contents are visible when expanded */
.split > div:nth-child(2) {
    overflow: visible;
    position: relative;
}

.split > div:nth-child(2) > div:not(.toggle-icon) {
    width: 100%;
    height: 100%;
    overflow: auto;
    padding-top: 36px; /* Space for the toggle icon */
}
"""


class SplitJS(JSComponent):
    """
    Professional split panel component with collapsible sidebar.

    The component uses an icon in the top-right corner of each panel section
    for toggling between expanded and collapsed states.
    """

    left = Child()
    right = Child()
    sizes = param.NumericTuple(default=(100, 0), length=2)
    expanded_sizes = param.NumericTuple(default=(35, 65), length=2)
    min_sizes = param.NumericTuple(default=(300, 0), length=2)
    collapsed = param.Boolean(default=True)
    invert = param.Boolean(default=False, doc="""
        Whether to invert the layout, changing the toggle button side and panel styles.
        This is useful for supporting different panel layouts like chat-left vs chat-right.
        """)

    _esm = """
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
      // In inverted mode, the right content should always be visible when collapsed
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
      } else {
        toggleIcon.className = 'toggle-icon';
        toggleIcon.innerHTML = model.collapsed
          ? `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`  // Left arrow when collapsed
          : `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`;  // Right arrow when expanded
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
          if (model.invert ? sizes[0] <= 5 : sizes[1] <= 5) {
            model.collapsed = true;

            // Set arrow direction based on invert parameter
            if (model.invert) {
              toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`; // Right arrow
            } else {
              toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`; // Left arrow
            }
          } else {
            model.collapsed = false;
            contentWrapper.className = '';
            contentWrapper.style.display = 'block';

            // Set arrow direction based on invert parameter
            if (model.invert) {
              toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`; // Left arrow
            } else {
              toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`; // Right arrow
            }
          }
        },
      });

      // Toggle button event listener
      toggleIcon.addEventListener('click', () => {
        if (model.collapsed) {
          // Expand
          splitInstance.setSizes(model.expanded_sizes);
          model.collapsed = false;

          // Make sure content is visible when expanding
          contentWrapper.className = '';
          contentWrapper.style.display = 'block';

          // Set arrow direction based on invert parameter
          if (model.invert) {
            toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`; // Left arrow
          } else {
            toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`; // Right arrow
          }
        } else {
          // Collapse with appropriate sizes based on invert parameter
          if (model.invert) {
            splitInstance.setSizes([0, 100]);
          } else {
            splitInstance.setSizes([100, 0]);
          }
          model.collapsed = true;

          // Only hide content in non-inverted mode
          if (!model.invert) {
            contentWrapper.className = 'collapsed-content';
            contentWrapper.style.display = 'none';
          }

          // Set arrow direction based on invert parameter
          if (model.invert) {
            toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="9 18 15 12 9 6"></polyline></svg>`; // Right arrow
          } else {
            toggleIcon.innerHTML = `<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"></polyline></svg>`; // Left arrow
          }
        }
        view.invalidate_layout();
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
    }"""

    _stylesheets = [CSS]
