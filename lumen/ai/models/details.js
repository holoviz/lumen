export function render({ model, view }) {
    // Create container elements
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    const contentContainer = document.createElement('div');

    summary.innerHTML = model.title;
    summary.style.cursor = 'pointer';

    details.appendChild(summary);
    details.appendChild(contentContainer);
    details.open = !model.collapsed;

    contentContainer.className = 'details-content';

    details.addEventListener('toggle', () => {
        model.send_msg({ collapsed: !details.open });
        model.collapsed = !details.open;
    });

    model.on('title', () => {
        summary.innerHTML = model.title;
    });

    model.on('msg:custom', (event) => {
        if (event.type === 'update_collapsed') {
            details.open = !event.collapsed;
        }
    });

    contentContainer.appendChild(model.get_child('object'));

    return details;
}
