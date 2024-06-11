from pathlib import Path

import jinja2

THIS_DIR = Path(__file__).parent

def render_template(template, **context):
    template_path = Path(template)
    if not template_path.exists():
        template_path = THIS_DIR / "prompts" / template
    template_contents = template_path.read_text()
    template = jinja2.Template(template_contents)
    return template.render(context)
