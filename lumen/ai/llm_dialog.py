"""
LLM Selection Dialog Component

This component provides a dialog for selecting LLM models and adjusting temperature.
Similar to the SourceCatalog but for LLM management.
"""

import param

from panel.layout import Column, HSpacer, Row
from panel.pane import Markdown
from panel.viewable import Viewer
from panel_material_ui import (
    Button, Dialog, Divider, FloatSlider, Select,
)

from .llm import (
    AnthropicAI, GoogleAI, LiteLLM, LlamaCpp, Llm, MistralAI, Ollama, OpenAI,
)


class LLMModelCard(Viewer):
    """
    Card component for displaying and configuring a single LLM model type.

    Shows:
    - Model type name (default, reasoning, sql, etc.)
    - Model selection dropdown
    - Current model info
    """

    llm = param.ClassSelector(class_=Llm, doc="The LLM instance to configure")

    model_type = param.String(doc="The model type key (e.g., 'default', 'reasoning')")

    llm_choices = param.List(default=[], doc="Available model choices for this card")

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
            margin=0
        )

        # Watch for changes
        self.model_select.param.watch(self._on_model_change, "value")

    def _get_default_models(self):
        """Get list of available models based on LLM provider type."""
        llm_type = type(self.llm).__name__

        # Default models based on LLM provider
        model_options = {
            "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            "AnthropicAI": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-4-sonnet-latest"],
            "GoogleAI": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
            "LiteLLM": ["gpt-4o-mini", "claude-3-5-sonnet-latest", "gemini/gemini-pro"],
            "Ollama": ["qwen2.5-coder:7b", "llama3.2:latest", "mistral:latest"],
            "LlamaCpp": ["Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"],
            "MistralAI": ["mistral-large-latest", "mistral-small-latest", "codestral-latest"],
            "AzureOpenAI": ["gpt-4o-mini", "gpt-4o"],
            "AINavigator": ["gpt-4o-mini", "gpt-4o"],
            "WebLLM": ["Qwen2.5-7B-Instruct-q4f16_1-MLC"],
        }

        # Get current model to ensure it's included
        current_config = self.llm.model_kwargs.get(self.model_type, {})
        current_model = current_config.get("model", "")

        # Get available models for this provider
        available = model_options.get(llm_type, [])

        # If no provider-specific models, use a sensible default
        if not available:
            available = ["gpt-4o-mini"]  # fallback

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
        descriptions = {
            "default": "General purpose model for most tasks",
            "reasoning": "Advanced model for complex reasoning tasks",
            "sql": "Optimized model for SQL query generation",
            "vega_lite": "Specialized model for data visualization",
            "ui": "Lightweight model for UI interactions",
        }

        description = descriptions.get(self.model_type, f"Model for {self.model_type} tasks")
        return Column(Markdown(f"**{description}**", margin=0), self.model_select)


class LLMConfigDialog(Viewer):
    """
    Dialog component for LLM configuration.

    Provides:
    - Provider selection
    - Temperature slider
    - Model selection for different model types
    - Apply/Cancel buttons
    """

    llm = param.ClassSelector(class_=Llm, doc="The LLM instance to configure")

    llm_choices = param.List(default=[], doc="Available model choices to display in dropdowns")

    provider_choices = param.Dict(default={}, doc="Available LLM providers to choose from")

    on_llm_change = param.Callable(default=None, doc="Callback function called when LLM provider changes")

    def __init__(self, **params):
        super().__init__(**params)

        # Store original values for cancellation
        self._original_temp = getattr(self.llm, "temperature", 0.7)
        self._original_models = dict(self.llm.model_kwargs)
        self._original_provider = type(self.llm).__name__

        # Set up available providers if not provided
        if not self.provider_choices:
            self.provider_choices = {
                "OpenAI": OpenAI,
                "Anthropic": AnthropicAI,
                "Google AI": GoogleAI,
                "LiteLLM": LiteLLM,
                "Ollama": Ollama,
                "Llama.cpp": LlamaCpp,
                "Mistral AI": MistralAI,
            }

        # Title
        self._title = Markdown("## LLM Configuration\n\nConfigure your language model provider, settings and select models for different tasks.", margin=(0, 0, 8, 0))

        # Provider selection
        current_provider_name = self._get_provider_display_name(type(self.llm).__name__)
        self._provider_select = Select(
            name="LLM Provider",
            value=current_provider_name,
            options=list(self.provider_choices.keys()),
            sizing_mode="stretch_width",
            margin=(5, 0)
        )
        self._provider_select.param.watch(self._on_provider_change, "value")

        # Temperature slider
        self._temperature_slider = FloatSlider(
            name="Temperature", value=self._original_temp, start=0.0, end=2.0, step=0.1, sizing_mode="stretch_width", margin=(5, 0)
        )

        # Model type cards
        self._model_cards = Column(margin=(5, 0))

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

    def _get_provider_display_name(self, class_name):
        """Convert class name to display name."""
        name_mapping = {
            "OpenAI": "OpenAI",
            "AnthropicAI": "Anthropic",
            "GoogleAI": "Google AI",
            "LiteLLM": "LiteLLM",
            "Ollama": "Ollama",
            "LlamaCpp": "Llama.cpp",
            "MistralAI": "Mistral AI",
        }
        return name_mapping.get(class_name, class_name)

    def _on_provider_change(self, event):
        """Handle provider selection change."""
        # Skip if we're already in the middle of updating to avoid recursion
        if getattr(self, '_updating_provider', False):
            return

        new_provider_name = event.new
        if new_provider_name and new_provider_name != event.old:
            # Get the provider class
            provider_class = self.provider_choices[new_provider_name]

            # Create new instance with default settings
            try:
                # Set flag to prevent recursion
                self._updating_provider = True

                # Preserve current temperature if possible
                current_temp = getattr(self.llm, 'temperature', 0.7)

                # Create new LLM instance - it will use its default model_kwargs
                new_llm = provider_class(temperature=current_temp)

                # Update the dialog's LLM reference
                self.llm = new_llm

                # Update temperature slider to match new provider's default
                self._temperature_slider.value = getattr(new_llm, 'temperature', 0.7)

                # Refresh model cards for new provider - this will now work correctly
                # because the new LLM has proper model_kwargs structure
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

    def _refresh_model_cards(self):
        """Create cards for each model type in the LLM."""
        cards = []

        # Get all model types from the LLM
        model_types = list(self.llm.model_kwargs.keys())

        for model_type in sorted(model_types):
            # Create a temporary card to get provider-specific defaults
            temp_card = LLMModelCard(llm=self.llm, model_type=model_type, llm_choices=[])
            provider_defaults = temp_card._get_default_models()

            # For provider-specific filtering, only use llm_choices if they're compatible
            llm_choices = []
            if self.llm_choices:
                # Only use llm_choices that are valid for this provider
                llm_choices = [model for model in self.llm_choices
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
                llm_choices=llm_choices
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
        current_provider = self._get_provider_display_name(type(self.llm).__name__)
        original_provider_display = self._get_provider_display_name(self._original_provider)

        if current_provider != original_provider_display:
            # Revert to original provider
            provider_class = None
            for cls in self.provider_choices.values():
                if cls.__name__ == self._original_provider:
                    provider_class = cls
                    break

            if provider_class:
                self.llm = provider_class(temperature=self._original_temp)
                self._provider_select.value = original_provider_display

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
