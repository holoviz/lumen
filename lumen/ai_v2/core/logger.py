"""
Logging utilities for the two-loop agentic system
"""

import logging
import sys

from typing import Optional


# Define color codes for console output
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # Grays for debug info
    GRAY = '\033[90m'
    LIGHT_GRAY = '\033[37m'

    # Prompt highlighting
    PROMPT = '\033[95m'  # Magenta for prompts
    PROMPT_CONTENT = '\033[96m'  # Cyan for prompt content

    # Checkpoint highlighting
    CHECKPOINT = '\033[93m'  # Yellow for checkpoint results

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""

    COLORS = {
        'DEBUG': Colors.GRAY,
        'INFO': Colors.OKBLUE,
        'WARNING': Colors.WARNING,
        'ERROR': Colors.FAIL,
        'CRITICAL': Colors.FAIL + Colors.BOLD,
    }

    def format(self, record):
        # Save the original format
        original_format = self._style._fmt

        # Add color to the level name
        if record.levelname in self.COLORS:
            self._style._fmt = f"{self.COLORS[record.levelname]}{original_format}{Colors.ENDC}"

        # Format the record
        output = super().format(record)

        # Restore the original format
        self._style._fmt = original_format

        return output

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup a logger with colored output

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Don't add handlers if already configured
    if logger.handlers:
        return logger

    # Set level
    logger.setLevel(getattr(logging, level.upper()))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = ColoredFormatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def get_logger(name: str, level: str = "DEBUG") -> logging.Logger:
    """Get or create a logger

    Args:
        name: Logger name
        level: Log level

    Returns:
        Logger instance
    """
    return setup_logger(name, level)

class AgentLogger:
    """Specialized logger for agent operations"""

    def __init__(self, name: str, level: str = "DEBUG"):
        self.logger = get_logger(name, level)
        self.level = level

    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message"""
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """Log error message"""
        self.logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        """Log critical message"""
        self.logger.critical(msg, **kwargs)

    def log_goal_start(self, goal: str):
        """Log the start of goal achievement with delimiter"""
        delimiter = "═" * 80  # Double line for goal boundaries
        self.info(f"{Colors.HEADER}{delimiter}{Colors.ENDC}")
        self.info(f"🎯 Starting goal: {goal}")
        self.info(f"{Colors.HEADER}{delimiter}{Colors.ENDC}")

    def log_goal_complete(self, goal: str, milestones: int, duration: float | None = None):
        """Log goal completion with delimiter"""
        delimiter = "═" * 80  # Double line for goal boundaries
        duration_str = f" in {duration:.2f}s" if duration else ""
        self.info(f"{Colors.HEADER}{delimiter}{Colors.ENDC}")
        self.info(f"✅ Goal completed: {goal} ({milestones} milestones{duration_str})")
        self.info(f"{Colors.HEADER}{delimiter}{Colors.ENDC}")

    def log_roadmap_created(self, milestones: int):
        """Log roadmap creation"""
        delimiter = "─" * 80  # Terminal width delimiter
        self.info(f"📋 Roadmap created with {milestones} milestones")
        self.info(f"{Colors.OKGREEN}{delimiter}{Colors.ENDC}")

    def log_roadmap_updated(self, completed: int, skipped: int, remaining: int):
        """Log roadmap update"""
        self.info(f"📋 Roadmap updated: {completed} completed, {skipped} skipped, {remaining} remaining")

    def log_complete(self, milestone: str, steps: int):
        """Log milestone completion with delimiter"""
        delimiter = "─" * 80  # Terminal width delimiter
        self.info(f"{Colors.FAIL}{delimiter}{Colors.ENDC}")
        self.info(f"✅ Milestone completed: {milestone} ({steps} steps)")
        self.info(f"{Colors.FAIL}{delimiter}{Colors.ENDC}")

    def log_milestone_step(self, step: int, milestone: str, actions: int):
        """Log milestone step"""
        delimiter = "─" * 80  # Terminal width delimiter
        self.info(f"{Colors.FAIL}{delimiter}{Colors.ENDC}")
        self.debug(f"⚡ Step {step} for '{milestone}': {actions} actions")
        self.info(f"{Colors.FAIL}{delimiter}{Colors.ENDC}")

    def log_milestone_retry(self, step: int, milestone: str, reason: str):
        """Log milestone retry with delimiter"""
        delimiter = "┄" * 80  # Dotted line for retries
        self.warning(f"{Colors.WARNING}{delimiter}{Colors.ENDC}")
        self.warning(f"🔄 Step {step} retry for '{milestone}': {reason}")
        self.warning(f"{Colors.WARNING}{delimiter}{Colors.ENDC}")

    def log_actions_start(self, actions: list):
        """Log start of action execution"""
        action_names = [action.name if hasattr(action, 'name') else str(action) for action in actions]
        self.debug(f"🔧 Executing {len(actions)} actions: {action_names}")

    def log_actions_complete(self, actions: list, duration: float | None = None):
        """Log completion of action execution"""
        duration_str = f" in {duration:.2f}s" if duration else ""
        self.debug(f"✅ Actions completed: {len(actions)} actions{duration_str}")

    def log_action_error(self, action_name: str, error: str):
        """Log action execution error"""
        self.error(f"❌ Action '{action_name}' failed: {error}")

    def log_checkpoint_created(self, summary: str, artifacts_count: int = 0):
        """Log checkpoint creation with yellow highlighting"""
        artifacts_str = f" with {artifacts_count} artifacts" if artifacts_count > 0 else ""
        self.info(f"{Colors.CHECKPOINT}📋 Checkpoint created{artifacts_str}: {summary}{Colors.ENDC}")

    def log_checkpoint_content(self, checkpoint_summary: str):
        """Log checkpoint content for debugging"""
        if len(checkpoint_summary) > 200:
            display_summary = checkpoint_summary[:200] + f"... [+{len(checkpoint_summary) - 200} chars]"
        else:
            display_summary = checkpoint_summary
        self.debug(f"📄 Checkpoint summary: {display_summary}")

    def log_prompt_content(self, method_name: str, content: str):
        """Log prompt content with highlighting"""
        # Split content into lines for better readability
        lines = content.split('\n')
        for line in lines:
            self.debug(f"{Colors.PROMPT_CONTENT}{line}{Colors.ENDC}")

    def log_llm_call(self, method_name: str, messages: int, tokens: int | None = None):
        """Log LLM API call"""
        token_str = f", ~{tokens} tokens" if tokens else ""
        self.debug(f"🧠 LLM call for {method_name}: {messages} messages{token_str}")

    def log_discovery(self, item_type: str, count: int, details: str | None = None):
        """Log data discovery"""
        detail_str = f": {details}" if details else ""
        self.info(f"🔍 Discovered {count} {item_type}{detail_str}")

    def set_level(self, level: str):
        """Change logging level"""
        self.logger.setLevel(getattr(logging, level.upper()))
        self.level = level
