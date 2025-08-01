"""
LLM Selection Dialog Component

This component provides a dialog for selecting LLM models and adjusting temperature.
Similar to the SourceCatalog but for LLM management.
Uses param.concrete_descendents to automatically detect all agent types.
"""

import param

from panel.layout import Column, HSpacer, Row
from panel.pane import Markdown
from panel.viewable import Viewer
from panel_material_ui import (
    Button, Dialog, Divider, FloatSlider, Select,
)

from .agents import Agent
from .llm import Llm
from .utils import class_name_to_llm_spec_key


class LLMModelCard(Viewer):
    """
    Card component for displaying and configuring a single LLM model type.

    Shows:
    - Model type name (default, edit, sql, etc.)
    - Model selection dropdown
    - Current model info
    """

    description = param.String(default="The description of the model type", doc="Description of the model type")

    llm = param.ClassSelector(class_=Llm, doc="The LLM instance to configure")

    llm_choices = param.List(default=[], doc="Available model choices for this card")

    model_type = param.String(doc="The model type key (e.g., 'default', 'edit')")

    def __init__(self, **params):
        super().__init__(**params)

        # Get available models for this type
        current_config = self.llm.model_kwargs.get(self.model_type, {})
        current_model = current_config.get("model", "unknown")

        # Use provided llm_choices or get defaults
        if not self.llm_choices:
            self.llm_choices = self._get_default_models()

        # Ensure we have at least one option
        if not self.llm_choices:
            self.llm_choices = [current_model] if current_model and current_model != "unknown" else ["No models available"]

        # Set the default value to first available model if current model not found
        default_value = current_model if current_model in self.llm_choices else self.llm_choices[0]

        # Create model selector
        self.model_select = Select(
            name=f"{self.model_type.title()} Model",
            value=default_value,
            options=self.llm_choices,
            sizing_mode="stretch_width",
            disabled=self.param.llm_choices.rx().rx.len() <= 1,
            margin=0
        )

        # Watch for changes
        self.model_select.param.watch(self._on_model_change, "value")

    def _get_default_models(self):
        """Get models for the select dropdown from the LLM class."""
        llm_type = type(self.llm)

        # Check if the LLM class has a select_models attribute (now a param)
        if hasattr(llm_type, 'select_models'):
            available = list(llm_type.select_models)
        else:
            # Fallback to empty list
            available = []

        # Get current model to ensure it's included
        current_config = self.llm.model_kwargs.get(self.model_type, {})
        current_model = current_config.get("model", "")

        # Ensure current model is in the list if it exists and is not "unknown"
        if current_model and current_model != "unknown" and current_model not in available:
            available = [current_model] + available

        return available

    def _on_model_change(self, event):
        """Handle model selection change."""
        new_model = event.new
        if new_model and new_model not in {event.old, "No models available"}:
            # Update the LLM model_kwargs
            if self.model_type not in self.llm.model_kwargs:
                self.llm.model_kwargs[self.model_type] = {}
            self.llm.model_kwargs[self.model_type]["model"] = new_model

    def __panel__(self):
        return Column(Markdown(f"**{self.description}**", margin=0), self.model_select)


class LLMConfigDialog(Viewer):
    """
    Dialog component for LLM configuration.

    Provides:
    - Provider selection
    - Temperature slider
    - Model selection for different model types
    - Apply/Cancel buttons
    """

    agent_types = param.List(default=[], doc="Available agents in the UI")

    llm = param.ClassSelector(class_=Llm, doc="The LLM instance to configure")

    llm_choices = param.List(default=[], doc="Available model choices to display in dropdowns")

    on_llm_change = param.Callable(default=None, doc="Callback function called when LLM provider changes")

    provider_choices = param.Dict(default={}, doc="Available LLM providers to choose from")

    def __init__(self, **params):
        super().__init__(**params)

        # Store original values for cancellation
        self._original_temp = getattr(self.llm, "temperature", 0.7)
        self._original_models = dict(self.llm.model_kwargs)
        self._original_provider = type(self.llm).__name__

        # Auto-detect available providers if not provided
        if not self.provider_choices:
            self.provider_choices = self._get_available_providers()

        # Title
        self._title = Markdown("## LLM Configuration\n\nConfigure your language model provider, settings and select models for different tasks.", margin=(0, 0, 8, 0))

        # Provider selection
        current_provider_class = type(self.llm)
        self._provider_select = Select(
            name="LLM Provider",
            value=current_provider_class,
            options=self.provider_choices,
            sizing_mode="stretch_width",
            margin=(5, 0)
        )
        self._provider_select.param.watch(self._on_provider_change, "value")

        # Temperature slider
        self._temperature_slider = FloatSlider(
            name="Temperature", value=self._original_temp, start=0.0, end=2.0, step=0.1, sizing_mode="stretch_width", margin=(5, 0)
        )

        # Model type cards
        self._model_cards = Column(margin=(5, 0), height=300, scroll="y")

        # Buttons
        self._apply_button = Button(
            label="Apply Changes", color="primary", on_click=self._apply_changes, margin=(12, 0, 0, 0)
        )

        self._cancel_button = Button(
            label="Cancel", on_click=self._cancel_changes, margin=(12, 0, 0, 12), variant="outlined"
        )

        self._reset_button = Button(
            label="Reset to Defaults", color="light", on_click=self._reset_to_defaults, margin=(12, 0, 0, 0)
        )

        self._buttons = Row(self._reset_button, HSpacer(), self._cancel_button, self._apply_button, sizing_mode="stretch_width")

        # Create the dialog content
        self._content = Column(
            self._title,
            Divider(sizing_mode="stretch_width"),
            Markdown("### Provider Selection", margin=0),
            Markdown(
                "Choose your LLM provider. Different providers offer different models and capabilities.",
                margin=0, styles={"font-size": "0.9em", "color": "#666"},
            ),
            self._provider_select,
            Divider(margin=(12, 0), sizing_mode="stretch_width"),
            Markdown("### Temperature Setting", margin=0),
            Markdown(
                "Controls randomness in responses. Lower values (0.1-0.3) for focused tasks, higher values (0.7-1.0) for creative tasks.",
                margin=0, styles={"font-size": "0.9em", "color": "#666"},
            ),
            self._temperature_slider,
            Divider(margin=(12, 0), sizing_mode="stretch_width"),
            Markdown("### Model Selection", margin=0),
            Markdown(
                "Choose different models for different types of tasks. More powerful models may be slower but provide better results.",
                margin=0, styles={"font-size": "0.9em", "color": "#666"},
            ),
            self._model_cards,
            self._buttons,
            sizing_mode="stretch_width",
            margin=(0, 24, 8, 24),
        )

        # Create dialog wrapper
        self._dialog = Dialog(
            self._content,
            close_on_click=True,
            show_close_button=True,
            sizing_mode="stretch_width",
            width_option="md",
            sx={".MuiDialogContent-root": {"p": "0 24px 20px 24px"}}
        )

        # Initialize model cards
        self._refresh_model_cards()

        # Flag to prevent update loops during provider changes
        self._updating_provider = False

    def _get_available_providers(self):
        """Auto-detect available LLM providers using param.concrete_descendents."""
        providers = {}

        # Get all concrete descendant classes of Llm
        for cls in param.concrete_descendents(Llm).values():
            # Get display name from class param or fallback to class name
            display_name = getattr(cls, 'display_name', cls.__name__)
            # Use display name as key, class as value
            providers[display_name] = cls
        return providers

    def _get_provider_display_name(self, class_name):
        """Convert class name to display name using class variable or mapping."""
        # First check if the class has a display_name attribute
        for display_name, cls in self.provider_choices.items():
            if cls.__name__ == class_name:
                return display_name

        # Fallback to class name
        return class_name

    def _on_provider_change(self, event):
        """Handle provider selection change."""
        # Skip if we're already in the middle of updating to avoid recursion
        if getattr(self, '_updating_provider', False):
            return

        new_provider_class = event.new
        if new_provider_class and new_provider_class != event.old:
            # Create new instance with default settings
            try:
                # Set flag to prevent recursion
                self._updating_provider = True

                # Preserve current temperature if possible
                current_temp = getattr(self.llm, 'temperature', 0.7)

                # Create new LLM instance - it will use its default model_kwargs
                new_llm = new_provider_class(temperature=current_temp)

                # Update the dialog's LLM reference
                self.llm = new_llm

                # Update temperature slider to match new provider's default
                self._temperature_slider.value = getattr(new_llm, 'temperature', 0.7)

                # Refresh model cards for new provider
                self._refresh_model_cards()

                # Notify parent UI of the change immediately (for instant feedback)
                if self.on_llm_change:
                    self.on_llm_change(new_llm)

            except Exception:
                # If provider creation fails, revert selection without triggering recursion
                with param.parameterized.batch_call_watchers(self._provider_select):
                    self._provider_select.value = event.old
            finally:
                # Always reset the flag
                self._updating_provider = False

    def _get_all_agent_types(self):
        """
        Get all possible agent types by using param.concrete_descendents to find all Agent classes.
        Returns a dict mapping llm_spec_key to description.
        """
        agent_types = {}

        # Add common predefined types first (only non-duplicate ones)
        predefined_types = {
            "default": "General purpose model for most tasks",
            "edit": "Advanced model for retry & edit tasks",
            "ui": "Lightweight model for UI interactions"
        }
        agent_types.update(predefined_types)

        # Get all concrete descendant classes of Agent
        agent_classes = []
        for cls in param.concrete_descendents(Agent).values():
            if cls in self.agent_types:
                agent_classes.append(cls)

        # Sort agent classes alphabetically by class name
        agent_classes.sort(key=lambda cls: cls.__name__)

        for cls in agent_classes:
            # Convert class name to llm_spec_key
            llm_spec_key = class_name_to_llm_spec_key(cls.__name__)
            # Create description based on class name and docstring
            description = cls.purpose.strip()
            # Truncate long descriptions
            if len(description) > 80:
                description = description[:77] + "..."
            # Add agent name prefix
            agent_name = cls.__name__
            description = f"{agent_name}: {description}"
            agent_types[llm_spec_key] = description

        return agent_types


    def _refresh_model_cards(self):
        """Create cards for each model type, including all agent types."""
        cards = []

        # Get all possible agent types (not just those in model_kwargs)
        all_agent_types = self._get_all_agent_types()

        # Combine existing model_kwargs with all agent types
        all_model_types = set(self.llm.model_kwargs.keys())
        all_model_types.update(all_agent_types.keys())

        # Sort model types with predefined types first, then alphabetically
        predefined_order = ["default", "edit", "ui"]
        sorted_model_types = []

        # Add predefined types first (in order)
        for ptype in predefined_order:
            if ptype in all_model_types:
                sorted_model_types.append(ptype)

        # Add remaining types alphabetically
        remaining_types = sorted(all_model_types - set(predefined_order))
        sorted_model_types.extend(remaining_types)

        for model_type in sorted_model_types:
            if model_type not in all_agent_types:
                # If this model type is not an agent type, skip it
                continue

            # Ensure this model type exists in model_kwargs
            if model_type not in self.llm.model_kwargs:
                # Use the default model from model_kwargs if available, otherwise use first select_models
                default_model_config = self.llm.model_kwargs.get("default", {}).copy()
                if not default_model_config and hasattr(self.llm, 'select_models') and self.llm.select_models:
                    default_model_config = {"model": self.llm.select_models[0]}
                self.llm.model_kwargs[model_type] = default_model_config

            # Create a temporary card to get provider-specific defaults
            temp_card = LLMModelCard(llm=self.llm, model_type=model_type, llm_choices=[], description="")
            provider_defaults = temp_card._get_default_models()

            # For provider-specific filtering, only use llm_choices if they're compatible
            llm_choices = []
            if self.llm_choices:
                # Only use llm_choices that are valid for this provider
                llm_choices = [
                    model for model in self.llm_choices
                    if model in provider_defaults or not provider_defaults]

            # If no valid choices or no llm_choices provided, use provider defaults
            if not llm_choices:
                llm_choices = provider_defaults

            # Ensure we have at least one option
            if not llm_choices:
                llm_choices = ["No models available"]

            # Get current model for this type
            current_config = self.llm.model_kwargs.get(model_type, {})
            current_model = current_config.get("model", "")

            # If current model is not in available choices, update to first choice
            if current_model not in llm_choices and llm_choices[0] != "No models available":
                self.llm.model_kwargs[model_type]["model"] = llm_choices[0]
                current_model = llm_choices[0]

            card = LLMModelCard(
                llm=self.llm,
                model_type=model_type,
                llm_choices=llm_choices,
                description=all_agent_types[model_type],
            )

            # Ensure the card's select widget shows the correct current model
            if current_model in llm_choices:
                card.model_select.value = current_model

            cards.append(card)

        self._model_cards.objects = cards

    def _apply_changes(self, event):
        """Apply the configuration changes."""
        # Update temperature if the LLM supports it
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = self._temperature_slider.value

        # Store new original values
        self._original_temp = self._temperature_slider.value
        self._original_models = dict(self.llm.model_kwargs)
        self._original_provider = type(self.llm).__name__

        # Notify parent UI of any changes
        if self.on_llm_change:
            self.on_llm_change(self.llm)

        # Close dialog
        self._dialog.open = False

    def _reset_to_defaults(self, event):
        """Reset all settings to default values."""
        # Reset temperature to default
        default_temp = 0.7
        self._temperature_slider.value = default_temp

        # Reset to original LLM class defaults
        llm_class = type(self.llm)
        default_models = llm_class.param.model_kwargs.default

        self.llm.model_kwargs.clear()
        self.llm.model_kwargs.update(default_models)

        if hasattr(self.llm, "temperature"):
            self.llm.temperature = default_temp

        # Refresh model cards
        self._refresh_model_cards()

    def _cancel_changes(self, event):
        """Cancel changes and reset to original values."""
        # Reset provider if changed
        current_provider_class = type(self.llm)
        original_provider_class = None

        # Find the original provider class
        for cls in self.provider_choices.values():
            if cls.__name__ == self._original_provider:
                original_provider_class = cls
                break

        if current_provider_class != original_provider_class and original_provider_class:
            # Revert to original provider
            self.llm = original_provider_class(temperature=self._original_temp)
            self._provider_select.value = original_provider_class

        # Reset temperature slider
        self._temperature_slider.value = self._original_temp
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = self._original_temp

        # Reset model configurations
        self.llm.model_kwargs.clear()
        self.llm.model_kwargs.update(self._original_models)

        # Refresh model cards to original state
        self._refresh_model_cards()

        # Close dialog
        self._dialog.open = False

    @property
    def open(self):
        """Get dialog open state."""
        return self._dialog.open

    @open.setter
    def open(self, value):
        """Set dialog open state."""
        self._dialog.open = value

    def close(self):
        """Close the dialog."""
        self._dialog.open = False

    def __panel__(self):
        return self._dialog
