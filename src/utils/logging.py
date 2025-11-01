"""
Logging utilities for Valerie Visual ASR.

Provides structured logging with different levels, file/console output,
rich formatting, and model-specific logging functions.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich import box


def setup_logger(
    name: str = "valerie",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    use_rich: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with rich formatting and indicators.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
        use_rich: Whether to use rich formatting for console output
        format_string: Custom format string for file output
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with rich formatting
    if console:
        if use_rich:
            console = Console()
            rich_handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                show_time=True,
                show_path=True,
                markup=True,
                log_time_format="[%X]"
            )
            rich_handler.setLevel(getattr(logging, level.upper()))
            logger.addHandler(rich_handler)
        else:
            # Fallback to standard console handler
            console_handler = logging.StreamHandler(sys.stdout)
            if format_string is None:
                format_string = (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(filename)s:%(lineno)d - %(message)s"
                )
            formatter = logging.Formatter(format_string)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    # File handler (always uses standard formatting)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            )
        
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(format_string)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "valerie") -> logging.Logger:
    """Get existing logger or create default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Create default logger if none exists
        logger = setup_logger(name)
    return logger


def log_model_summary(
    model: nn.Module,
    input_shape: tuple,
    logger: Optional[logging.Logger] = None,
    use_rich: bool = True
) -> None:
    """
    Log model summary including parameter count and architecture.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        logger: Logger instance (optional)
        use_rich: Whether to use rich formatting
    """
    if logger is None:
        logger = get_logger()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    
    if use_rich:
        console = Console()
        
        # Create model summary table
        table = Table(title="🤖 Model Summary", box=box.ROUNDED)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Model", model.__class__.__name__)
        table.add_row("Input Shape", str(input_shape))
        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Non-trainable Parameters", f"{total_params - trainable_params:,}")
        table.add_row("Estimated Size", f"{param_size_mb:.2f} MB")
        
        console.print(table)
        
        # Create architecture table
        arch_table = Table(title="🏗️ Model Architecture", box=box.ROUNDED)
        arch_table.add_column("Module Name", style="cyan")
        arch_table.add_column("Type", style="green")
        arch_table.add_column("Parameters", style="yellow", justify="right")
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters())
                arch_table.add_row(
                    name if name else "root",
                    module.__class__.__name__,
                    f"{param_count:,}"
                )
        
        console.print(arch_table)
    else:
        # Fallback to standard logging
        logger.info("=" * 80)
        logger.info("MODEL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        logger.info(f"Estimated model size: {param_size_mb:.2f} MB")
        
        logger.info("\nMODEL ARCHITECTURE:")
        logger.info("-" * 40)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters())
                logger.info(f"{name}: {module.__class__.__name__} ({param_count:,} params)")
        
        logger.info("=" * 80)


def log_training_info(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log training configuration and system information.
    
    Args:
        config: Training configuration dictionary
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    
    # System info
    logger.info("SYSTEM INFORMATION:")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Training config
    logger.info("\nTRAINING PARAMETERS:")
    for section, params in config.items():
        logger.info(f"\n{section.upper()}:")
        if isinstance(params, dict):
            for key, value in params.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {params}")
    
    logger.info("=" * 80)


def log_epoch_metrics(
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    learning_rate: float,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log metrics for a training epoch.
    
    Args:
        epoch: Epoch number
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        learning_rate: Current learning rate
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger()
    
    logger.info(f"Epoch {epoch:3d} | LR: {learning_rate:.2e}")
    
    # Training metrics
    train_str = " | ".join([f"Train {k}: {v:.4f}" for k, v in train_metrics.items()])
    logger.info(f"  {train_str}")
    
    # Validation metrics
    if val_metrics:
        val_str = " | ".join([f"Val {k}: {v:.4f}" for k, v in val_metrics.items()])
        logger.info(f"  {val_str}")


def log_inference_stats(
    num_samples: int,
    total_time: float,
    avg_sequence_length: float,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log inference performance statistics.
    
    Args:
        num_samples: Number of processed samples
        total_time: Total processing time in seconds
        avg_sequence_length: Average sequence length
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger()
    
    samples_per_sec = num_samples / total_time
    time_per_sample = total_time / num_samples
    
    logger.info("=" * 60)
    logger.info("INFERENCE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Processed samples: {num_samples}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Samples per second: {samples_per_sec:.2f}")
    logger.info(f"Time per sample: {time_per_sample:.4f} seconds")
    logger.info(f"Average sequence length: {avg_sequence_length:.1f} frames")
    logger.info("=" * 60)


class TrainingLogger:
    """Training logger with rich formatting and structured logging."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        level: str = "INFO",
        use_rich: bool = True
    ):
        """
        Initialize training logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            level: Logging level
            use_rich: Whether to use rich formatting
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_rich = use_rich
        self.console = Console() if use_rich else None
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(
            name=f"valerie.{experiment_name}",
            level=level,
            log_file=str(log_file),
            console=True,
            use_rich=use_rich
        )
        
        self.logger.info(f"🚀 Training logger initialized for experiment: {experiment_name}")
        self.logger.info(f"📝 Log file: {log_file}")
    
    def info(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log info message with rich formatting."""
        if self.use_rich and extra_data:
            self._log_with_panel(message, extra_data, "ℹ️ INFO", "blue")
        else:
            self.logger.info(f"ℹ️ {message}")
    
    def warning(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log warning message with rich formatting."""
        if self.use_rich and extra_data:
            self._log_with_panel(message, extra_data, "⚠️ WARNING", "yellow")
        else:
            self.logger.warning(f"⚠️ {message}")
    
    def error(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log error message with rich formatting."""
        if self.use_rich and extra_data:
            self._log_with_panel(message, extra_data, "❌ ERROR", "red")
        else:
            self.logger.error(f"❌ {message}")
    
    def critical(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log critical message with rich formatting."""
        if self.use_rich and extra_data:
            self._log_with_panel(message, extra_data, "🔥 CRITICAL", "red")
        else:
            self.logger.critical(f"🔥 {message}")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(f"🐛 {message}")
    
    def success(self, message: str) -> None:
        """Log success message."""
        self.logger.info(f"✅ {message}")
    
    def _log_with_panel(self, message: str, data: Dict, title: str, color: str) -> None:
        """Log message with rich panel formatting."""
        if self.console:
            content = message + "\n"
            for key, value in data.items():
                content += f"{key}: {value}\n"
            
            panel = Panel(
                content.strip(),
                title=title,
                border_style=color,
                expand=False
            )
            self.console.print(panel)


# Global logger instance
_global_logger = None


def init_global_logger(experiment_name: str = "valerie", log_dir: str = "logs") -> None:
    """Initialize global logger for the application."""
    global _global_logger
    _global_logger = TrainingLogger(experiment_name, log_dir)


def get_global_logger() -> TrainingLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        init_global_logger()
    return _global_logger