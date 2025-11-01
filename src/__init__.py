"""
Valerie: Enhanced Visual ASR Language Model

A state-of-the-art visual automatic speech recognition system that combines
3D spatio-temporal embeddings, Conformer architecture, hybrid CTC/Attention
training, and audio knowledge distillation for improved lip reading performance.
"""

__version__ = "0.1.0"
__author__ = "Valerie Research Team"
__email__ = "contact@valerie-asr.com"

# Import implemented modules
from .models import *
from .utils import *
from .data import *
from .training import *
from .inference import *
from .evaluation import *
from .visualization import *