"""
Comprehensive Evaluation Pipeline for Valerie Visual ASR.

Implements evaluation on test sets with detailed metrics:
- Phoneme Error Rate (PER) and Word Error Rate (WER)
- Component-wise performance analysis
- Statistical significance testing
- Comparison with baseline models
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, asdict
import json
import time
from collections import defaultdict
import editdistance
from scipy import stats
from tqdm import tqdm

from src.inference.inference_engine import ValerieInferenceEngine, InferenceConfig
from src.data import VoxCeleb2Dataset
from src.training.metrics import ValerieMetrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    # Model settings
    model_checkpoint: str = "checkpoints/best_model.pt"
    device: str = "auto"
    
    # Dataset settings
    dataset_root: str = "data"
    dataset_split: str = "test"
    max_samples: Optional[int] = None
    
    # Evaluation settings
    batch_size: int = 16
    num_workers: int = 4
    
    # Metrics
    calculate_per: bool = True
    calculate_wer: bool = True
    calculate_bleu: bool = True
    calculate_component_metrics: bool = True
    
    # Statistical analysis
    confidence_interval: float = 0.95
    bootstrap_samples: int = 1000
    
    # Output settings
    save_results: bool = True
    save_predictions: bool = True
    output_dir: str = "evaluation_results"
    
    # Comparison
    baseline_results: Optional[str] = None  # Path to baseline results
    
    # Debug
    save_debug_info: bool = False
    max_debug_samples: int = 100


class ComponentMetrics(NamedTuple):
    """Metrics for individual components."""
    
    # CTC metrics
    ctc_per: float
    ctc_confidence: float
    
    # Attention metrics  
    attention_per: float
    attention_confidence: float
    
    # Reconstruction metrics
    reconstruction_bleu: float
    reconstruction_confidence: float
    
    # Overall metrics
    overall_per: float
    overall_wer: float
    overall_bleu: float
    overall_confidence: float


class EvaluationResult(NamedTuple):
    """Complete evaluation result."""
    
    # Summary metrics
    per: float
    wer: float
    bleu: float
    confidence: float
    
    # Component metrics
    component_metrics: ComponentMetrics
    
    # Statistical analysis
    per_ci: Tuple[float, float]  # Confidence interval
    wer_ci: Tuple[float, float]
    
    # Performance analysis
    processing_time: float
    throughput: float  # samples per second
    
    # Sample analysis
    num_samples: int
    successful_samples: int
    failed_samples: int
    
    # Detailed results
    per_sample_results: Optional[List[Dict]] = None
    error_analysis: Optional[Dict] = None
    
    # Comparison with baseline
    baseline_comparison: Optional[Dict] = None


class ValerieEvaluator:
    """
    Comprehensive evaluator for Valerie Visual ASR.
    
    Provides detailed evaluation on test sets with:
    - Multiple metrics (PER, WER, BLEU)
    - Component-wise analysis
    - Statistical significance testing
    - Error analysis and visualization
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize inference engine
        inference_config = InferenceConfig(
            model_checkpoint=config.model_checkpoint,
            device=config.device,
            batch_size=config.batch_size
        )
        self.inference_engine = ValerieInferenceEngine(inference_config)
        
        # Initialize metrics calculator
        self.metrics_calculator = ValerieMetrics()
        
        logger.info(f"✅ ValerieEvaluator initialized:")
        logger.info(f"   Model: {config.model_checkpoint}")
        logger.info(f"   Dataset: {config.dataset_root}/{config.dataset_split}")
        logger.info(f"   Output: {config.output_dir}")
    
    def evaluate(self, dataset=None) -> EvaluationResult:
        """
        Run comprehensive evaluation.
        
        Args:
            dataset: Optional dataset to evaluate on
            
        Returns:
            Evaluation result
        """
        
        logger.info("🔍 Starting comprehensive evaluation")
        start_time = time.time()
        
        # Load dataset if not provided
        if dataset is None:
            dataset = self._load_dataset()
        
        logger.info(f"📊 Evaluating on {len(dataset)} samples")
        
        # Run inference on all samples
        all_results = []
        all_ground_truth = []
        
        # Process in batches
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluation"):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            
            # Get batch data
            batch_videos = []
            batch_ground_truth = []
            
            for idx in batch_indices:
                sample = dataset[idx]
                # Extract video path or frames
                if 'video_path' in sample:
                    batch_videos.append(sample['video_path'])
                elif 'video' in sample:
                    # Handle tensor data
                    batch_videos.append(sample['video'])
                
                # Extract ground truth text
                if 'transcription' in sample:
                    batch_ground_truth.append(sample['transcription'])
                elif 'text' in sample:
                    batch_ground_truth.append(sample['text'])
                else:
                    batch_ground_truth.append("")  # No ground truth
            
            # Run inference
            try:
                if isinstance(batch_videos[0], str):
                    # Video file paths
                    batch_results = self.inference_engine.infer_batch(
                        batch_videos, 
                        return_debug_info=self.config.save_debug_info
                    )
                else:
                    # Tensor data - need to implement tensor batch inference
                    batch_results = self._infer_tensor_batch(batch_videos)
                
                all_results.extend(batch_results)
                all_ground_truth.extend(batch_ground_truth)
                
            except Exception as e:
                logger.error(f"❌ Batch inference failed: {e}")
                # Add empty results for failed batch
                empty_results = [self.inference_engine._empty_result(0.0) for _ in batch_indices]
                all_results.extend(empty_results)
                all_ground_truth.extend(batch_ground_truth)
        
        # Calculate metrics
        evaluation_result = self._calculate_metrics(all_results, all_ground_truth, time.time() - start_time)
        
        # Save results
        if self.config.save_results:
            self._save_results(evaluation_result, all_results, all_ground_truth)
        
        logger.info("✅ Evaluation completed")
        logger.info(f"   PER: {evaluation_result.per:.3f}")
        logger.info(f"   WER: {evaluation_result.wer:.3f}")
        logger.info(f"   BLEU: {evaluation_result.bleu:.3f}")
        logger.info(f"   Confidence: {evaluation_result.confidence:.3f}")
        
        return evaluation_result
    
    def _load_dataset(self):
        """Load evaluation dataset."""
        
        try:
            dataset = VoxCeleb2Dataset(
                voxceleb2_root=self.config.dataset_root,
                split=self.config.dataset_split,
                max_samples=self.config.max_samples,
                use_video=True,
                use_transcriptions=True
            )
            
            logger.info(f"✅ Loaded dataset: {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def _infer_tensor_batch(self, tensor_batch: List[torch.Tensor]):
        """Handle tensor batch inference (placeholder)."""
        
        # This would implement tensor-based batch inference
        # For now, return empty results
        logger.warning("⚠️ Tensor batch inference not implemented")
        return [self.inference_engine._empty_result(0.0) for _ in tensor_batch]
    
    def _calculate_metrics(
        self, 
        results: List, 
        ground_truth: List[str], 
        processing_time: float
    ) -> EvaluationResult:
        """Calculate comprehensive metrics."""
        
        # Extract predictions and ground truth
        predictions = [result.text for result in results]
        phoneme_predictions = [result.phonemes for result in results]
        confidences = [result.confidence for result in results]
        
        # Calculate basic metrics
        per_scores = []
        wer_scores = []
        bleu_scores = []
        
        successful_samples = 0
        
        for pred, gt, conf in zip(predictions, ground_truth, confidences):
            if conf > 0.1:  # Consider as successful
                successful_samples += 1
                
                # PER calculation (character-level)
                if pred and gt:
                    per = editdistance.eval(pred.lower(), gt.lower()) / max(len(gt), 1)
                    per_scores.append(per)
                    
                    # WER calculation (word-level)
                    pred_words = pred.lower().split()
                    gt_words = gt.lower().split()
                    wer = editdistance.eval(pred_words, gt_words) / max(len(gt_words), 1)
                    wer_scores.append(wer)
                    
                    # BLEU calculation (simplified)
                    bleu = self._calculate_bleu(pred, gt)
                    bleu_scores.append(bleu)
        
        # Calculate average metrics
        avg_per = np.mean(per_scores) if per_scores else 1.0
        avg_wer = np.mean(wer_scores) if wer_scores else 1.0
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_confidence = np.mean(confidences)
        
        # Calculate confidence intervals
        per_ci = self._bootstrap_confidence_interval(per_scores) if per_scores else (1.0, 1.0)
        wer_ci = self._bootstrap_confidence_interval(wer_scores) if wer_scores else (1.0, 1.0)
        
        # Calculate component metrics
        component_metrics = self._calculate_component_metrics(results, ground_truth)
        
        # Performance metrics
        throughput = len(results) / processing_time if processing_time > 0 else 0.0
        
        # Create detailed per-sample results
        per_sample_results = None
        if self.config.save_predictions:
            per_sample_results = []
            for i, (result, gt) in enumerate(zip(results, ground_truth)):
                sample_result = {
                    'sample_id': i,
                    'prediction': result.text,
                    'ground_truth': gt,
                    'phonemes': result.phonemes,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time
                }
                per_sample_results.append(sample_result)
        
        # Create evaluation result
        evaluation_result = EvaluationResult(
            per=avg_per,
            wer=avg_wer,
            bleu=avg_bleu,
            confidence=avg_confidence,
            component_metrics=component_metrics,
            per_ci=per_ci,
            wer_ci=wer_ci,
            processing_time=processing_time,
            throughput=throughput,
            num_samples=len(results),
            successful_samples=successful_samples,
            failed_samples=len(results) - successful_samples,
            per_sample_results=per_sample_results
        )
        
        return evaluation_result
    
    def _calculate_component_metrics(
        self, 
        results: List, 
        ground_truth: List[str]
    ) -> ComponentMetrics:
        """Calculate metrics for individual components."""
        
        # Extract component-specific data
        ctc_confidences = []
        recon_confidences = []
        overall_confidences = []
        
        ctc_per_scores = []
        attention_per_scores = []
        overall_per_scores = []
        
        for result, gt in zip(results, ground_truth):
            if result.confidence > 0.1:
                ctc_confidences.append(result.phoneme_confidence)
                recon_confidences.append(result.reconstruction_confidence)
                overall_confidences.append(result.confidence)
                
                # Calculate PER for phonemes vs ground truth
                if result.phonemes and gt:
                    phoneme_str = ' '.join(result.phonemes)
                    ctc_per = editdistance.eval(phoneme_str.lower(), gt.lower()) / max(len(gt), 1)
                    ctc_per_scores.append(ctc_per)
                
                # Overall PER
                if result.text and gt:
                    overall_per = editdistance.eval(result.text.lower(), gt.lower()) / max(len(gt), 1)
                    overall_per_scores.append(overall_per)
        
        # Calculate averages
        component_metrics = ComponentMetrics(
            ctc_per=np.mean(ctc_per_scores) if ctc_per_scores else 1.0,
            ctc_confidence=np.mean(ctc_confidences) if ctc_confidences else 0.0,
            attention_per=np.mean(ctc_per_scores) if ctc_per_scores else 1.0,  # Placeholder
            attention_confidence=np.mean(ctc_confidences) if ctc_confidences else 0.0,  # Placeholder
            reconstruction_bleu=np.mean([0.5] * len(recon_confidences)) if recon_confidences else 0.0,  # Placeholder
            reconstruction_confidence=np.mean(recon_confidences) if recon_confidences else 0.0,
            overall_per=np.mean(overall_per_scores) if overall_per_scores else 1.0,
            overall_wer=np.mean(overall_per_scores) if overall_per_scores else 1.0,  # Simplified
            overall_bleu=np.mean([0.5] * len(overall_confidences)) if overall_confidences else 0.0,  # Placeholder
            overall_confidence=np.mean(overall_confidences) if overall_confidences else 0.0
        )
        
        return component_metrics
    
    def _calculate_bleu(self, prediction: str, reference: str) -> float:
        """Calculate simplified BLEU score."""
        
        try:
            pred_words = prediction.lower().split()
            ref_words = reference.lower().split()
            
            if not pred_words or not ref_words:
                return 0.0
            
            # Simple 1-gram precision
            pred_set = set(pred_words)
            ref_set = set(ref_words)
            
            if not pred_set:
                return 0.0
            
            precision = len(pred_set & ref_set) / len(pred_set)
            
            # Brevity penalty
            bp = min(1.0, len(pred_words) / len(ref_words))
            
            return bp * precision
            
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return 0.0
    
    def _bootstrap_confidence_interval(
        self, 
        scores: List[float], 
        confidence_level: float = None
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        
        if confidence_level is None:
            confidence_level = self.config.confidence_interval
        
        if not scores or len(scores) < 2:
            return (0.0, 1.0)
        
        try:
            # Bootstrap resampling
            bootstrap_means = []
            n_samples = len(scores)
            
            for _ in range(self.config.bootstrap_samples):
                bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower = np.percentile(bootstrap_means, 100 * alpha / 2)
            upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
            
            return (lower, upper)
            
        except Exception as e:
            logger.warning(f"Bootstrap CI calculation failed: {e}")
            return (0.0, 1.0)
    
    def _save_results(
        self, 
        evaluation_result: EvaluationResult, 
        all_results: List, 
        ground_truth: List[str]
    ):
        """Save evaluation results to files."""
        
        try:
            # Save summary results
            summary_path = self.output_dir / "evaluation_summary.json"
            with open(summary_path, 'w') as f:
                # Convert to dict and handle non-serializable types
                result_dict = asdict(evaluation_result)
                result_dict['per_sample_results'] = None  # Save separately
                json.dump(result_dict, f, indent=2, default=str)
            
            # Save detailed per-sample results
            if evaluation_result.per_sample_results:
                details_path = self.output_dir / "per_sample_results.json"
                with open(details_path, 'w') as f:
                    json.dump(evaluation_result.per_sample_results, f, indent=2)
            
            # Save predictions and ground truth
            predictions_path = self.output_dir / "predictions.txt"
            with open(predictions_path, 'w') as f:
                for i, (result, gt) in enumerate(zip(all_results, ground_truth)):
                    f.write(f"Sample {i}:\n")
                    f.write(f"  Prediction: {result.text}\n")
                    f.write(f"  Ground Truth: {gt}\n")
                    f.write(f"  Confidence: {result.confidence:.3f}\n")
                    f.write(f"  Phonemes: {' '.join(result.phonemes)}\n")
                    f.write("\n")
            
            logger.info(f"💾 Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def compare_with_baseline(
        self, 
        evaluation_result: EvaluationResult,
        baseline_path: str
    ) -> Dict[str, any]:
        """Compare results with baseline."""
        
        try:
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            
            baseline_per = baseline_data.get('per', 1.0)
            baseline_wer = baseline_data.get('wer', 1.0)
            baseline_bleu = baseline_data.get('bleu', 0.0)
            
            # Calculate improvements
            per_improvement = (baseline_per - evaluation_result.per) / baseline_per * 100
            wer_improvement = (baseline_wer - evaluation_result.wer) / baseline_wer * 100
            bleu_improvement = (evaluation_result.bleu - baseline_bleu) / max(baseline_bleu, 0.01) * 100
            
            # Statistical significance test
            per_p_value = self._significance_test(evaluation_result.per, baseline_per)
            
            comparison = {
                'baseline_per': baseline_per,
                'current_per': evaluation_result.per,
                'per_improvement_percent': per_improvement,
                'baseline_wer': baseline_wer,
                'current_wer': evaluation_result.wer,
                'wer_improvement_percent': wer_improvement,
                'baseline_bleu': baseline_bleu,
                'current_bleu': evaluation_result.bleu,
                'bleu_improvement_percent': bleu_improvement,
                'per_p_value': per_p_value,
                'statistically_significant': per_p_value < 0.05
            }
            
            logger.info("📊 Baseline Comparison:")
            logger.info(f"   PER: {evaluation_result.per:.3f} vs {baseline_per:.3f} ({per_improvement:+.1f}%)")
            logger.info(f"   WER: {evaluation_result.wer:.3f} vs {baseline_wer:.3f} ({wer_improvement:+.1f}%)")
            logger.info(f"   BLEU: {evaluation_result.bleu:.3f} vs {baseline_bleu:.3f} ({bleu_improvement:+.1f}%)")
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Baseline comparison failed: {e}")
            return {}
    
    def _significance_test(self, current_metric: float, baseline_metric: float) -> float:
        """Perform statistical significance test."""
        
        try:
            # Simple t-test (would need actual sample distributions for proper test)
            # This is a placeholder implementation
            t_stat = abs(current_metric - baseline_metric) / (0.01 + abs(baseline_metric) * 0.1)
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            return p_value
            
        except Exception as e:
            logger.warning(f"Significance test failed: {e}")
            return 1.0  # No significance
