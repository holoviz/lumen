import base64
import io
import os
import re
import tempfile
import warnings

from datetime import datetime
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any

import nbformat
import yaml

from docx.shared import Mm
from docxtpl import (
    DocxTemplate, InlineImage, R, RichText,
)
from panel import Column
from panel.chat import ChatMessage, ChatStep
from panel.pane.image import ImageBase
from panel_material_ui import Typography

from ..config import config
from ..pipeline import Pipeline
from ..views import View
from .views import LumenOutput, VegaLiteOutput


def make_md_cell(text: str):
    return nbformat.v4.new_markdown_cell(source=text)

def make_preamble(preamble: str, extensions: list[str] | None = None, title: str | None = None):
    if extensions is None:
        extensions = []
    if title:
        header = make_md_cell(f'# {title}')
    if 'tabulator' not in extensions:
        extensions = ['tabulator'] + extensions
    exts = ', '.join([repr(ext) for ext in extensions])
    source = (preamble + dedent(
        f"""
        import yaml

        import lumen as lm
        import panel as pn

        pn.extension({exts})
        """
    )).strip()
    imports = nbformat.v4.new_code_cell(source=source)
    return [header, imports] if title else [imports]

def serialize_avatar(avatar: str | BytesIO, size: int = 45) -> str:
    """
    Process different types of avatar inputs into HTML img tag or text span.

    Args:
        avatar: Avatar source (URL, BytesIO, PIL Image, or text)
        size: Desired width and height of the avatar in pixels

    Returns:
        HTML string representing the avatar
    """
    if isinstance(avatar, ImageBase):
        avatar = avatar.object
    if isinstance(avatar, BytesIO):
        avatar = avatar.getvalue()

    if isinstance(avatar, str):
        if avatar.startswith(('http://', 'https://')):
            return f'<img src="{avatar}" width={size} height={size} alt="avatar" style="border-radius: 50%;"></img>'
        elif os.path.exists(avatar):
            avatar_path = os.path.abspath(avatar)
            return f'<img src="{avatar_path}" width={size} height={size} alt="avatar" style="border-radius: 50%;"></img>'
        else:
            return f'<span class="text-avatar" style="width: {size}px; height: {size}px; display: inline-flex; align-items: center; justify-content: center; background-color: #f0f0f0; border-radius: 50%;">{avatar[:2].upper()}</span>'
    elif isinstance(avatar, bytes):
        img_data = base64.b64encode(avatar).decode()
        return f'<img src="data:image/png;base64,{img_data}" width={size} height={size} alt="avatar" style="border-radius: 50%;"></img>'
    else:
        return f'<span class="text-avatar" style="width: {size}px; height: {size}px; display: inline-flex; align-items: center; justify-content: center; background-color: #f0f0f0; border-radius: 50%;">{str(avatar)[:2].upper()}</span>'

def format_markdown(msg: ChatMessage):
    avatar_html = serialize_avatar(msg.avatar)
    header = f'<div style="display: flex; flex-direction: row; font-weight: bold; font-size: 2em;">{avatar_html}<span style="margin-left: 0.5em">{msg.user}</span></div>'
    prefix = '\n' if msg.user == 'User' else '\n> '
    content = prefix.join(msg.serialize().split('\n'))
    return [nbformat.v4.new_markdown_cell(source=f'{header}\n{prefix}{content}')]

def format_output(output: LumenOutput):
    ext = None
    code = []
    with config.param.update(serializer='csv'):
        # replace |2- |3- |4-... etc with | for a cleaner look
        spec = re.sub(r'(\|[-\d]*)', '|', yaml.dump(output.component.to_spec(), sort_keys=False))
    read_code = [
        f'yaml_spec = """\n{spec}"""',
        'spec = yaml.safe_load(yaml_spec)',
    ]
    if isinstance(output.component, Pipeline):
        code.extend([
            *read_code,
            'pipeline = lm.Pipeline.from_spec(spec)',
            'pipeline'
        ])
    elif isinstance(output.component, View):
        ext = output.component._extension
        code.extend([
            *read_code,
            'view = lm.View.from_spec(spec)',
            'view'
        ])
    return nbformat.v4.new_code_cell(source='\n'.join(code)), ext

def render_cells(messages: list[ChatMessage]) -> tuple[Any, list[str]]:
    cells, extensions = [], []
    for msg in messages:
        if msg.user in ("Help", " "):
            continue
        elif isinstance(msg.object, str):
            cells += format_markdown(msg)
        elif isinstance(msg.object, LumenOutput):
            cell, ext = format_output(msg.object)
            cells.append(cell)
            if ext and ext not in extensions:
                extensions.append(ext)
        elif isinstance(msg.object, Column):
            for obj in msg.object:
                if isinstance(obj, ChatStep):
                    continue
                cell, ext = format_output(obj.object)
                cells.append(cell)
                if ext and ext not in extensions:
                    extensions.append(ext)
    return cells, extensions

def write_notebook(cells):
    nb = nbformat.v4.new_notebook(cells=cells)
    return nbformat.v4.writes(nb)

def export_notebook(messages: list[ChatMessage], preamble: str = ""):
    cells, extensions = render_cells(messages)
    cells = make_preamble(preamble, extensions=extensions) + cells
    return write_notebook(cells)


def render_docx_template(
    outputs: list,
    docx_template_path: str | Path,
    **docx_context: dict,
) -> io.BytesIO:
    """
    Export outputs to a Word document (.docx) format.

    Arguments
    ---------
    outputs : list
        List of output objects to export (Typography, LumenOutput, etc.)
    docx_context : dict | None
        Context dictionary for docx template rendering. If keys are not provided,
        the following defaults will be used:

        - 'title': title parameter or 'Lumen Report'
        - 'subtitle': 'Generated on {date}' (e.g., 'Generated on October 27, 2025')
        - 'cover_page_header': '' (empty string)
        - 'cover_page_footer': '' (empty string)
        - 'content_page_header': '' (empty string)

        The following keys are always auto-generated and cannot be overridden:
        - 'sections': List of section dicts with 'title', 'image', and 'caption'
        - 'page_break': R('\f') for page breaks
    docx_template_path : str | None
        Path to the docx template file. If None, uses the default Lumen template.

    Returns
    -------
    BytesIO
        A BytesIO buffer containing the rendered docx document.

    Raises
    ------
    RuntimeError
        If the outputs list is empty.
    FileNotFoundError
        If the template file is not found.

    Example
    -------
        buffer = to_docx(
            outputs=report.outputs,
            **{
                'subtitle': 'Quarterly Analysis',
                'cover_page_header': 'ACME Corporation'
            }
        )
    """
    # Set default template path
    if docx_template_path is None:
        docx_template_path = str(Path(__file__).parent / "assets" / "lumen_template.docx")

    # Load template
    template_path = Path(docx_template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    doc = DocxTemplate(str(template_path))

    # Start with copy of docx_context or empty dict
    context = dict(docx_context) if docx_context else {}

    # Set defaults for missing keys
    if 'title' not in context:
        context['title'] = "Lumen Report"

    if 'subtitle' not in context:
        date_string = datetime.now().strftime("%B %d, %Y")
        context['subtitle'] = f"Generated on {date_string}"

    if 'cover_page_header' not in context:
        context['cover_page_header'] = ""

    if 'cover_page_footer' not in context:
        context['cover_page_footer'] = ""

    if 'content_page_header' not in context:
        context['content_page_header'] = ""

    # Always generate sections from outputs
    context['sections'] = generate_sections(doc, outputs)

    # Always set page_break
    context['page_break'] = R("\f")

    # Render template
    doc.render(context)

    # Return as BytesIO
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def generate_sections(doc, outputs: list) -> list[dict]:
    """
    Generate sections list from outputs for docx template.

    Arguments
    ---------
    doc : DocxTemplate
        The document template instance (needed for InlineImage creation)
    outputs : list
        List of output objects to process

    Returns
    -------
    list[dict]
        List of section dictionaries with title, image, and caption
    """
    sections = []
    current_section = None
    skip_next = False

    for i, out in enumerate(outputs):
        if skip_next:
            skip_next = False
            continue

        # Check if this is a section header (h2 Typography)
        if isinstance(out, Typography) and out.variant == 'h2':
            # Save previous section if it has an image
            if current_section and current_section.get('image'):
                sections.append(current_section)

            # Start new section
            current_section = {
                "title": out.object or "Untitled Section",
                "image": None,
                "caption": RichText("")
            }

        # Check for VegaLiteOutput to use as image
        elif isinstance(out, VegaLiteOutput) and current_section and not current_section.get('image'):
            image_path = output_to_image(out)
            if image_path:
                current_section["image"] = InlineImage(doc, image_path, width=Mm(160))

                # Check if next output is a Typography for caption
                if i + 1 < len(outputs):
                    next_out = outputs[i + 1]
                    if isinstance(next_out, Typography):
                        current_section["caption"] = RichText(next_out.object)
                        skip_next = True  # Skip the caption in next iteration

    # Add the last section if it has an image
    if current_section and current_section.get('image'):
        sections.append(current_section)

    return sections


def output_to_image(output: VegaLiteOutput) -> str | None:
    """
    Convert a VegaLiteOutput to an image file path.

    Arguments
    ---------
    output : VegaLiteOutput
        The output to convert

    Returns
    -------
    str | None
        Path to temporary image file, or None if conversion failed
    """
    # Create a temporary file for the image
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        # Render the component and save as image
        component = output.component
        with open(tmp_path, 'wb') as f:
            vega_pane = component.__panel__()._pane
            vega_pane.param.update(
                width=650,
                height=400,
            )
            image_bytes = vega_pane.export("png", scale=2, ppi=144)
            f.write(image_bytes)
            return tmp_path
    except Exception as e:
        warnings.warn(f"Failed to convert output to image: {e}", stacklevel=2)
        os.unlink(tmp_path)
        return None
