"""A Report that can be annotated with an AI-written, editable story."""
from __future__ import annotations

import io
import traceback as tb

from functools import cache, partial

import panel as pn
import param

from markdown_it import MarkdownIt
from panel.layout.base import Column, Row
from panel.pane import Markdown
from panel.widgets import TextEditor
from panel_material_ui import (
    Accordion, Button, Card, Dialog, IconButton, Select, Tabs, TextAreaInput,
    Typography,
)

from ..config import config
from ..pipeline import Pipeline
from ..views.base import View
from .agents.story import StoryAgent, StoryState, build_catalog
from .editors import LumenEditor
from .report import Report

# The same engine as panel.pane.Markdown, so a paragraph reads identically in
# the editor and in the exports.
_MARKDOWN = MarkdownIt("gfm-like")

# Quill ships its own font stack and a full-height container, which would make
# each paragraph read as a form field rather than as prose. The hover tint is
# the only hint that a paragraph can be edited at all.
_PROSE_CSS = """
.ql-container { height: auto; font-family: inherit; font-size: inherit; }
.ql-container.ql-bubble { border: none; }
.ql-editor { padding: 4px 6px; line-height: inherit; }
.ql-editor > :first-child { margin-top: 0; }
.ql-editor > :last-child { margin-bottom: 0; }
.ql-editor ol, .ql-editor ul { padding-left: 1.5em; }
.ql-editor:not(:focus):hover { background: rgba(127, 127, 127, 0.08); border-radius: 4px; }
"""

# The editor's Bokeh model fixes its height at 300px, which the layout applies
# as a flex basis on the element itself. Left alone every paragraph, however
# short, reserves a 300px block and the story stops flowing. The rule has to
# live on the row rather than on the widget, because a shadow root cannot
# restyle its own host.
_PROSE_ROW_CSS = """
.story-prose { flex: 1 1 auto !important; height: auto !important; }
"""


def _prose_to_html(text: str) -> str:
    """The Markdown a story block stores, as the HTML the editor shows."""
    return _MARKDOWN.render(text)


@cache
def _markitdown():
    """MarkItDown is slow to construct, so build it once and only when needed."""
    from markitdown import MarkItDown

    return MarkItDown()


def _prose_to_markdown(html: str) -> str:
    """The HTML the editor emits, back to the Markdown a story block stores."""
    converted = _markitdown().convert_stream(
        io.BytesIO(html.encode()), file_extension=".html", bullets="-",
    )
    # contenteditable inserts non-breaking spaces for runs of spaces.
    return converted.text_content.replace("\xa0", " ").strip()


class StoryReport(Report):
    """
    A `Report` with an extra Story tab: an AI writes a blog-post story that
    interleaves prose with the report's charts and tables, which can then be
    edited by hand or with the LLM, regenerated, and kept as versions.
    """

    # The Story tab is appended after the Report tab once a story exists.
    _story_tab = 1

    # Only the formatting that survives the round-trip back to Markdown; an
    # image or colour button would write HTML the exports cannot carry.
    _prose_toolbar = [
        ["bold", "italic"],
        [{"header": 2}, {"header": 3}],
        [{"list": "bullet"}, {"list": "ordered"}],
        ["blockquote", "link"],
        ["clean"],
    ]

    _tone_presets = [
        ("Executive summary", "Write a concise executive summary for decision-makers, leading with the key takeaways."),
        ("Technical deep-dive", "Write a detailed technical analysis that explains the methods and the data behind each chart."),
        ("Highlights only", "Keep it brief: just the headline finding for each chart and table."),
    ]

    def _init_view(self):
        self._story = StoryState()
        super()._init_view()
        # The story lives in its own tab so the section accordion (and its
        # checkboxes) is never disturbed. A headerless Card gives it the same
        # white pane as the section cards.
        self._story_column = Card(
            sizing_mode="stretch_width", margin=(0, 5, 15, 5),
            collapsible=False, hide_header=True,
            # The formatting menu of the last paragraph opens downwards and the
            # card would otherwise cut it in half.
            stylesheets=[".MuiPaper-root { overflow: visible !important; }"],
        )
        self._tabs = Tabs(
            ("Report", self._view),
            sizing_mode="stretch_width",
            stylesheets=[".MuiTabsPanel > .MuiBox-root { overflow: visible; }"],
        )
        self._annotate = IconButton(
            icon="auto_stories", on_click=self._open_story_dialog, size="large",
            color="default", margin=0, visible=False,
            description="Annotate the report with an AI-written story",
        )
        self._arrange = IconButton(
            icon="reorder", on_click=self._open_arrange_dialog, size="large",
            color="default", margin=0, visible=False,
            description="Arrange the report (reorder sections and add headings)",
        )
        self._outline_editor = pn.widgets.JSONEditor(
            mode="tree", sizing_mode="stretch_width", height=400,
        )
        self._outline_editor.param.watch(self._on_outline_change, "value")
        self._outline_apply = Button(
            label="Apply", variant="contained", on_click=self._apply_outline,
        )
        self._outline_dialog = Dialog(
            self._outline_editor,
            Row(self._outline_apply, align="end", margin=(10, 0, 0, 0), sizing_mode="stretch_width"),
            show_close_button=True,
            title="Arrange Report",
            width_option="md",
        )
        self._story_guidance = TextAreaInput(
            placeholder="Guide the story, e.g. focus on seasonal trends (optional)",
            sizing_mode="stretch_width", rows=3, margin=(10, 0, 0, 0),
        )
        self._story_presets = Row(
            *[
                Button(
                    label=label, variant="outlined", size="small", color="default",
                    on_click=partial(self._apply_story_preset, text=text),
                )
                for label, text in self._tone_presets
            ],
            sizing_mode="stretch_width",
        )
        self._story_generate = Button(
            label="Generate", variant="contained", icon="auto_stories",
            on_click=self._generate_story,
        )
        self._story_dialog = Dialog(
            Typography("Pick a tone or write your own guidance, then generate the story.", variant="body2"),
            self._story_presets,
            self._story_guidance,
            Row(self._story_generate, align="end", margin=(10, 0, 0, 0), sizing_mode="stretch_width"),
            show_close_button=True,
            title="Annotate Report",
            width_option="sm",
        )
        # Rewriting a single paragraph with the LLM.
        self._edit_index = None
        self._edit_instruction = TextAreaInput(
            placeholder="How should this paragraph change? e.g. make it shorter and lead with the number",
            sizing_mode="stretch_width", rows=3, margin=(10, 0, 0, 0),
        )
        self._edit_apply = Button(
            label="Rewrite", variant="contained", icon="auto_fix_high",
            on_click=self._rewrite_prose,
        )
        self._edit_dialog = Dialog(
            self._edit_instruction,
            Row(self._edit_apply, align="end", margin=(10, 0, 0, 0), sizing_mode="stretch_width"),
            show_close_button=True,
            title="Edit paragraph with AI",
            width_option="sm",
        )
        # Only on the report tab: the story tab has nothing to untick.
        self._export_hint.visible = param.bind(
            lambda status, active: (
                status in ("success", "error", "cancelled") and active == 0
            ),
            self.param.status, self._tabs.param.active,
        )
        # Splice the story controls into the base toolbar, dial and container.
        insert_at = self._menu.objects.index(self._export)
        self._menu.insert(insert_at, self._arrange)
        self._menu.insert(insert_at, self._annotate)
        dial_items = list(self._dial.items)
        dial_items[3:3] = [
            {"label": "Annotate Report", "icon": "auto_stories"},
            {"label": "Arrange Report", "icon": "reorder"},
        ]
        self._dial.items = dial_items
        self._container[self._container.objects.index(self._view)] = self._tabs
        self._container.extend([self._outline_dialog, self._story_dialog, self._edit_dialog])
        self._update_icon_visibility()

    # -- Base hooks overridden to add the story --------------------------------

    def _discard_story(self):
        """Forget every generated version and drop back to the report tab."""
        self._story.blocks = []
        self._story.title = ""
        self._story.versions = []
        self._story.version = 0
        self._story_column[:] = []
        if len(self._tabs) > self._story_tab:
            self._tabs.pop(self._story_tab)
        # Popping leaves ``active`` pointing at the removed tab.
        self._tabs.active = 0

    def _update_icon_visibility(self):
        super()._update_icon_visibility()
        # Called once from the base __init__ before the story buttons exist.
        if not hasattr(self, "_annotate"):
            return
        has_outputs = self.status in ("success", "error", "cancelled") or bool(self.views)
        self._annotate.visible = has_outputs
        self._annotate.disabled = self.llm is None
        self._arrange.visible = has_outputs

    async def _trigger_event(self, item: dict):
        icon = item["icon"]
        if icon == "auto_stories":
            self._open_story_dialog()
        elif icon == "reorder":
            self._open_arrange_dialog()
        else:
            await super()._trigger_event(item)

    async def _execute_event(self, event=None):
        # A fresh run invalidates any previously generated story.
        if self._active_task is None or self._active_task.done():
            self._discard_story()
        await super()._execute_event(event)

    def reset(self, start: int = 0):
        # Clearing the report also drops the generated story.
        self._discard_story()
        super().reset(start)

    def _populate_view(self):
        headers = {}
        objects = []
        for section in self._ordered_sections():
            header = self._section_headers.get(section) or self._section_header(section)
            headers[section] = header
            objects.append((header, section))
        self._section_headers = headers
        self._view[:] = objects
        has_outputs = self.status in ("success", "error", "cancelled") or bool(self.views)
        self._view.active = list(range(len(objects))) if (has_outputs or objects) else []

    @property
    def _export_views(self):
        # When the story is shown, export the blog-post flow; when an arrangement
        # is set, follow it; otherwise fall back to the plain section export.
        if self._show_story and self._story.blocks:
            views = list(self._header)
            views.append(Typography(self._story.title, variant="h2"))
            views += [
                Markdown(obj) if kind == "prose" else obj
                for kind, obj in self._story.blocks
            ]
            return views
        if not self._story.outline:
            return super()._export_views
        views = list(self._header)
        for kind, value, level in self._iter_arrangement():
            if kind == "heading":
                views.append(Typography(value, variant=f"h{level}"))
            elif value.include_in_export:
                views += self._selected_views(value)
        return views

    def _export_view(self):
        if self._show_story and self._story.blocks:
            return Column(*self._story_flow(), sizing_mode="stretch_width", margin=(0, 5, 15, 5))
        if not self._story.outline:
            return super()._export_view
        cards = []
        for kind, value, _ in self._iter_arrangement():
            if kind == "heading":
                cards.append((value, Markdown(f"### {value}")))
            elif value.include_in_export:
                cards.append((
                    value.title,
                    Column(*self._selected_views(value, include_header=False), sizing_mode="stretch_width"),
                ))
        return Accordion(
            *cards, active=list(range(len(cards))), sizing_mode="stretch_width",
            min_height=0, margin=(0, 5, 15, 5), sx=self._view.sx,
        )

    # -- Arrange (outline) -----------------------------------------------------

    def _section_by_title(self, title):
        for section in self:
            if section.title == title:
                return section
        return None

    def _build_default_outline(self):
        """Seed the arrangement outline from the currently selected sections."""
        return [{"section": section.title} for section in self if section.include_in_export]

    def _iter_arrangement(self, selected_only=True):
        """
        Yield the arranged outline as ('heading', text, level) or
        ('section', section, None) items, resolving section titles and
        dropping duplicates (and, by default, discarded sections).
        """
        seen = set()
        for block in self._story.outline:
            if "heading" in block:
                yield "heading", block["heading"], block.get("level", 2)
            elif "section" in block:
                section = self._section_by_title(block["section"])
                if section is None or id(section) in seen:
                    continue
                if selected_only and not section.include_in_export:
                    continue
                seen.add(id(section))
                yield "section", section, None

    def _ordered_sections(self):
        """Sections in the arranged outline order (unreferenced ones kept at the end)."""
        if not self._story.outline:
            return list(self)
        ordered = [s for kind, s, _ in self._iter_arrangement(selected_only=False) if kind == "section"]
        seen = {id(section) for section in ordered}
        for section in self:
            if id(section) not in seen:
                ordered.append(section)
        return ordered

    def _open_arrange_dialog(self, event=None):
        # Seed from the current arrangement, or the default section order.
        self._outline_editor.value = self._story.outline or self._build_default_outline()
        self._outline_dialog.open = True

    def _on_outline_change(self, event):
        self._story.outline = event.new or []
        self._populate_view()

    def _apply_outline(self, event=None):
        self._story.outline = list(self._outline_editor.value or [])
        self._populate_view()
        self._outline_dialog.open = False

    # -- Story generation ------------------------------------------------------

    def _open_story_dialog(self, event=None):
        self._story_dialog.open = True

    def _apply_story_preset(self, event=None, text=""):
        self._story_guidance.value = text

    async def _generate_story(self, event=None):
        guidance = (self._story_guidance.value or "").strip()
        self._story_dialog.open = False
        await self._annotate_report(guidance=guidance)

    def _collect_story_views(self):
        """Charts and tables from the selected sections, in reading order."""
        return [
            view
            for section in self if section.include_in_export
            for view in section.views if isinstance(view, LumenEditor)
        ]

    async def _annotate_report(self, event=None, guidance=""):
        """Write an AI blog-post story over the selected charts/tables and render it."""
        if self.llm is None or not len(self):
            return
        catalog = self._collect_story_views()
        if not catalog:
            return
        self._annotate.loading = True
        self._story_generate.loading = True
        try:
            agent = StoryAgent(llm=self.llm)
            story = await agent.write_story(
                await build_catalog(catalog), guidance=guidance, title=self.title or "",
            )
            self._set_story(story, catalog)
        except Exception:
            tb.print_exc()
        finally:
            self._annotate.loading = False
            self._story_generate.loading = False

    def _set_story(self, story, catalog):
        """Resolve the LLM story into interleaved prose/view blocks as a new version."""
        blocks, used = [], set()
        for block in story.blocks:
            if block.prose and block.prose.strip():
                blocks.append(("prose", block.prose))
            index = block.view
            if index is not None and 1 <= index <= len(catalog) and index not in used:
                used.add(index)
                blocks.append(("view", catalog[index - 1]))
        # Never drop a selected chart/table: append any the model did not place.
        for index, view in enumerate(catalog, start=1):
            if index not in used:
                blocks.append(("view", view))
        # Each story is kept as its own version, so regenerating offers a fresh
        # take without throwing away one the user may have edited by hand.
        self._story.versions.append({"title": story.title or "Story", "blocks": blocks})
        self._select_version(len(self._story.versions) - 1)

    def _select_version(self, index):
        """Show one of the generated versions of the story."""
        self._story.version = index
        version = self._story.versions[index]
        self._story.title = version["title"]
        # The version's own list is used live, so edits stay with that version.
        self._story.blocks = version["blocks"]
        self._render_story()

    def _switch_version(self, event):
        self._select_version(event.new)

    def _rebuild_view(self, editor):
        """An independent copy of a chart/table so the story preview never detaches
        it from its section; data is embedded so the copy stands alone."""
        component = editor.component
        with config.param.update(serializer='csv'):
            spec = component.to_spec()
        return Pipeline.from_spec(spec) if isinstance(component, Pipeline) else View.from_spec(spec)

    def _story_flow(self, editable=False):
        """
        The blog-post flow as renderables: the title, then prose interleaved with
        freshly rebuilt charts and tables. Rebuilding gives each caller its own
        copies so the live view and the exports never share objects. Prose is
        editable on screen and plain Markdown when exported.
        """
        items = []
        if self._story.title:
            heading = Typography(self._story.title, variant="h4", margin=(10, 10, 0, 10))
            if editable and self.llm is not None:
                controls = [heading]
                if len(self._story.versions) > 1:
                    version = Select(
                        options={f"Version {i+1}": i for i in range(len(self._story.versions))},
                        value=self._story.version, width=140, align="center",
                        margin=(10, 10, 0, 0), label="",
                        description="Switch between the generated versions of the story",
                    )
                    version.param.watch(self._switch_version, "value")
                    controls.append(version)
                # Regenerating reopens the guidance popup, so a fresh story can be
                # written with a different angle as a new version.
                controls.append(Button(
                    label="Regenerate", variant="outlined", size="small", color="default",
                    icon="refresh", align="center", margin=(10, 10, 0, 0),
                    description="Write the story again as a new version",
                    on_click=self._open_story_dialog,
                ))
                heading = Row(*controls, align="center", sizing_mode="stretch_width")
            items.append(heading)
        for index, (kind, obj) in enumerate(self._story.blocks):
            if kind != "prose":
                items.append(self._story_figure(obj))
            elif editable:
                items.append(self._editable_prose(index, obj))
            else:
                items.append(Markdown(obj, sizing_mode="stretch_width", margin=(5, 10)))
        return items

    def _story_figure(self, editor):
        """A chart or table as a fixed-height figure, so it flows in the story
        like a blog-post image instead of stretching to fill and swallowing the
        scroll of the tab it lives in."""
        return Column(
            self._rebuild_view(editor),
            height=400, sizing_mode="stretch_width", margin=(5, 10),
            styles={"min-height": "unset"},
        )

    def _editable_prose(self, index, text):
        """Prose the user can edit in place, by hand or by asking the LLM."""
        prose = TextEditor(
            value=_prose_to_html(text), mode="bubble", toolbar=self._prose_toolbar,
            sizing_mode="stretch_width", margin=(5, 10), css_classes=["story-prose"],
            stylesheets=[_PROSE_CSS],
        )
        prose.param.watch(partial(self._update_prose, index), "value")
        edit = IconButton(
            icon="auto_fix_high", size="small", color="default", align="start",
            margin=(5, 0, 0, 0), visible=self.llm is not None,
            description="Rewrite this paragraph with AI",
            on_click=partial(self._open_edit_dialog, index),
        )
        return Row(prose, edit, sizing_mode="stretch_width", stylesheets=[_PROSE_ROW_CSS])

    def _update_prose(self, index, event):
        # Keep the story blocks as the single source of truth so manual edits
        # survive tab switches and flow into every export. The editor speaks
        # HTML, the blocks stay Markdown, which is what the exports read.
        self._story.blocks[index] = ("prose", _prose_to_markdown(event.new))

    def _open_edit_dialog(self, index, event=None):
        self._edit_index = index
        self._edit_instruction.value = ""
        self._edit_dialog.open = True

    async def _rewrite_prose(self, event=None):
        """Ask the LLM to rewrite the paragraph the edit dialog was opened on."""
        instruction = (self._edit_instruction.value or "").strip()
        if self.llm is None or self._edit_index is None or not instruction:
            return
        self._edit_dialog.open = False
        self._edit_apply.loading = True
        try:
            agent = StoryAgent(llm=self.llm)
            edit = await agent.rewrite_prose(
                self._story.blocks[self._edit_index][1],
                instruction,
                await build_catalog(self._collect_story_views()),
            )
            self._story.blocks[self._edit_index] = ("prose", edit.prose)
            self._render_story()
        except Exception:
            tb.print_exc()
        finally:
            self._edit_apply.loading = False

    @property
    def _show_story(self):
        """Whether the story, rather than the report, is the view on screen."""
        return len(self._tabs) > self._story_tab and self._tabs.active == self._story_tab

    def _render_story(self):
        """Build the blog-post flow into the Story tab, then switch to it."""
        self._story_column[:] = self._story_flow(editable=True)
        if len(self._tabs) <= self._story_tab:
            self._tabs.append(("Story", self._story_column))
        self._tabs.active = self._story_tab
