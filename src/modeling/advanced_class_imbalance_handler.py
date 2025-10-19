"""
Advanced Class Imbalance Handling for Barcelona Parking Prediction Model

This module implements state-of-the-art techniques to address class imbalance issues,
specifically targeting the poor performance of classes 3 & 4 (31-33% precision).

Key improvements over basic SMOTE:
1. ADASYN for adaptive synthetic sampling
2. Borderline SMOTE for focus on decision boundaries  
3. Class-specific optimization strategies
4. Cost-sensitive ensemble methods
5. Advanced evaluation metrics
"""

import numpy as np
import pandas as pd
import logging
import copy
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier

# Models
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)

# Cost-sensitive learning
from sklearn.utils.class_weight import compute_class_weight
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- Ensure global filter sets exist even if this file is reloaded in an environment
try:
    KEEP_MODELS
except NameError:
    KEEP_MODELS = {"lgb_cost_sensitive", "lgb_focal"}

try:
    KEEP_SAMPLERS
except NameError:
    KEEP_SAMPLERS = {"adasyn", "smote_tomek"}

class AdvancedClassImbalanceHandler:
    """
    Advanced class imbalance handling with focus on improving classes 3 & 4 performance.
    """
    
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_strategy: Optional[str] = None
        self.class_performance: Dict[str, Any] = {}
        
    def analyze_class_distribution(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive analysis of class distribution and imbalance severity.
        """
        logger.info("Analyzing class distribution...")
        
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Calculate class proportions
        class_proportions = {cls: count/total_samples for cls, count in class_counts.items()}
        
        # Identify problematic classes (classes 3 & 4 based on your metrics)
        problematic_classes = [3, 4]  # Based on your 31-33% precision
        
        # Calculate imbalance metrics
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        analysis = {
            'class_counts': dict(class_counts),
            'class_proportions': class_proportions,
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'imbalance_ratio': imbalance_ratio,
            'problematic_classes': problematic_classes,
            'minority_class_counts': {cls: class_counts[cls] for cls in problematic_classes}
        }
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        logger.info(f"Problematic classes (3,4) counts: {analysis['minority_class_counts']}")
        
        return analysis
    
    def create_adaptive_sampling_strategy(self, y: np.ndarray, target_improvement_classes: List[int] = [3, 4]) -> Dict[int, int]:
        """
        Create adaptive sampling strategy focusing on target classes.
        """
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Calculate median class size as baseline
        median_count = int(np.median(list(class_counts.values())))
        
        # Aggressive upsampling for target classes
        sampling_strategy = {}
        
        for cls in target_improvement_classes:
            if cls in class_counts:
                current_count = class_counts[cls]
                # Target 150% of median class size for problematic classes (more aggressive)
                target_count = max(int(median_count * 1.5), current_count * 10)
                sampling_strategy[cls] = target_count
        
        # Moderate upsampling for other minority classes
        for cls, count in class_counts.items():
            if cls not in target_improvement_classes and count < median_count * 0.5:
                sampling_strategy[cls] = int(median_count * 0.5)
        
        logger.info(f"Adaptive sampling strategy: {sampling_strategy}")
        return sampling_strategy
    
    def apply_advanced_smote_variants(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        allowed: Optional[set] = None,
    ) -> Dict[str, Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]]:
        """
        Apply various SMOTE variants and return resampled datasets.
        """
        logger.info("Applying advanced SMOTE variants...")
        
        resampled_data: Dict[str, Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]] = {}
        sampling_strategy = self.create_adaptive_sampling_strategy(y)
        
        # Helper local to test if sampler should run
        def _run(name: str) -> bool:
            return allowed is None or name in allowed

        # 1. ADASYN - Adaptive Synthetic Sampling
        if _run('adasyn'):
            try:
                adasyn = ADASYN(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    n_neighbors=5
                )
                X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
                resampled_data['adasyn'] = (X_adasyn, y_adasyn)
                logger.info("✅ ADASYN sampling completed")
            except Exception as e:
                logger.warning(f"ADASYN failed: {e}")
        
        # 2. Borderline SMOTE - Focus on decision boundaries
        if _run('borderline_smote'):
            try:
                borderline_smote = BorderlineSMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=5,
                    m_neighbors=10,
                    kind='borderline-1'
                )
                X_borderline, y_borderline = borderline_smote.fit_resample(X, y)
                resampled_data['borderline_smote'] = (X_borderline, y_borderline)
                logger.info("✅ Borderline SMOTE sampling completed")
            except Exception as e:
                logger.warning(f"Borderline SMOTE failed: {e}")
        
        # 3. SVM SMOTE - Support vector-based sampling
        if _run('svm_smote'):
            try:
                svm_smote = SVMSMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=5
                )
                X_svm, y_svm = svm_smote.fit_resample(X, y)
                resampled_data['svm_smote'] = (X_svm, y_svm)
                logger.info("✅ SVM SMOTE sampling completed")
            except Exception as e:
                logger.warning(f"SVM SMOTE failed: {e}")
        
        # 4. SMOTE + Tomek Links (Hybrid approach)
        if _run('smote_tomek'):
            try:
                smote_tomek = SMOTETomek(
                    smote=SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state),
                    random_state=self.random_state
                )
                X_hybrid, y_hybrid = smote_tomek.fit_resample(X, y)
                resampled_data['smote_tomek'] = (X_hybrid, y_hybrid)
                logger.info("✅ SMOTE + Tomek Links sampling completed")
            except Exception as e:
                logger.warning(f"SMOTE + Tomek failed: {e}")
        
        # 5. Conservative SMOTE (fallback)
        if _run('conservative_smote'):
            try:
                conservative_strategy = {cls: min(count, Counter(y)[cls] * 2) 
                                       for cls, count in sampling_strategy.items()}
                smote = SMOTE(
                    sampling_strategy=conservative_strategy,
                    random_state=self.random_state,
                    k_neighbors=3
                )
                X_smote, y_smote = smote.fit_resample(X, y)
                resampled_data['conservative_smote'] = (X_smote, y_smote)
                logger.info("✅ Conservative SMOTE sampling completed")
            except Exception as e:
                logger.warning(f"Conservative SMOTE failed: {e}")
        
        return resampled_data
    
    def calculate_dynamic_class_weights(self, y: np.ndarray, focus_classes: List[int] = [3, 4]) -> Dict[int, float]:
        """
        Calculate dynamic class weights with extra emphasis on problematic classes.
        """
        class_counts = Counter(y)
        total_samples = len(y)
        n_classes = len(class_counts)
        
        # Base class weights (sklearn balanced approach)
        base_weights = {}
        for cls, count in class_counts.items():
            base_weights[cls] = total_samples / (n_classes * count)
        
        # Amplify weights proportionally to rarity
        max_count = max(class_counts.values())
        amplified_weights = base_weights.copy()
        for cls in focus_classes:
            if cls in amplified_weights and class_counts[cls] > 0:
                rarity_ratio = max_count / class_counts[cls]
                amplified_weights[cls] *= rarity_ratio  # Scale by how rare the class is
        
        logger.info(f"Base weights: {base_weights}")
        logger.info(f"Amplified weights: {amplified_weights}")
        
        return amplified_weights
    
    def create_cost_sensitive_models(self, class_weights: Dict[int, float]) -> Dict[str, Any]:
        """
        Create cost-sensitive model configurations.
        """
        debug_small = os.getenv("FAST_DEBUG") == "1"
        estims = 50 if debug_small else 200

        models = {
            'lgb_cost_sensitive': lgb.LGBMClassifier(
                n_estimators=estims,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=8,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight=class_weights,
                random_state=self.random_state,
                verbose=-1
            ),
            
            'rf_cost_sensitive': RandomForestClassifier(
                n_estimators=estims,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=class_weights,
                random_state=self.random_state
            ),
            
            'balanced_rf': BalancedRandomForestClassifier(
                n_estimators=estims,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state
            ),
            
            'balanced_bagging': BalancedBaggingClassifier(
                estimator=lgb.LGBMClassifier(
                    n_estimators=estims,
                    learning_rate=0.1,
                    verbose=-1,
                    random_state=self.random_state
                ),
                n_estimators=10 if not debug_small else 5,
                random_state=self.random_state
            ),

            # Native focal-loss; available from LightGBM ≥4.2
            'lgb_focal': lgb.LGBMClassifier(
                objective="multiclass_focal",
                alpha=2.0,
                n_estimators=estims,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight=class_weights,
                random_state=self.random_state,
                verbose=-1,
            ),
        }
        
        # Filter models to KEEP_MODELS to speed up experimentation
        models = {k: v for k, v in models.items() if k in KEEP_MODELS}
        
        return models
    
    def evaluate_class_specific_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Detailed evaluation focusing on class-specific performance.
        """
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Focus on problematic classes
        class_3_metrics = {
            'precision': precision[3] if len(precision) > 3 else 0,
            'recall': recall[3] if len(recall) > 3 else 0,
            'f1': f1[3] if len(f1) > 3 else 0,
            'support': support[3] if len(support) > 3 else 0
        }
        
        class_4_metrics = {
            'precision': precision[4] if len(precision) > 4 else 0,
            'recall': recall[4] if len(recall) > 4 else 0,
            'f1': f1[4] if len(f1) > 4 else 0,
            'support': support[4] if len(support) > 4 else 0
        }
        
        # Calculate improvement score (focus on classes 3 & 4)
        problematic_classes_f1 = (class_3_metrics['f1'] + class_4_metrics['f1']) / 2
        
        evaluation = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'class_3_metrics': class_3_metrics,
            'class_4_metrics': class_4_metrics,
            'problematic_classes_f1': problematic_classes_f1,
            'class_3_f1': class_3_metrics['f1'],
            'class_4_f1': class_4_metrics['f1'],
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist()
        }
        
        return evaluation
    
    def run_comprehensive_evaluation(self, X: pd.DataFrame, y: np.ndarray, n_splits: int = 4) -> Dict[str, Any]:
        """Leak-free evaluation: resample training fold only, preserve temporal order."""

        logger.info("Starting comprehensive class imbalance evaluation (fold-wise sampling)…")

        # Merge class 4 into 3 if class 4 absent
        if 4 not in np.unique(y):
            logger.info("Class 4 not present – remapping any label 4 to 3 (if exists)")
            y = np.where(y == 4, 3, y)

        distribution_analysis = self.analyze_class_distribution(y)

        base_class_weights = self.calculate_dynamic_class_weights(y)
        models = self.create_cost_sensitive_models(base_class_weights)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Prepare empty result dicts
        self.results = {}

        # For each fold iterate over sampling strategies and models
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_orig, y_train_orig = X.iloc[train_idx], y[train_idx]
            X_test, y_test = X.iloc[test_idx], y[test_idx]

            # Ensure sampler-filter set exists
            if "KEEP_SAMPLERS" not in globals():
                globals()["KEEP_SAMPLERS"] = {"adasyn", "smote_tomek"}

            # Build each sampler once per fold (cache)
            resampled_datasets = {}
            full_resampled = self.apply_advanced_smote_variants(X_train_orig, y_train_orig, allowed=KEEP_SAMPLERS)
            for s_name in KEEP_SAMPLERS:
                if s_name in full_resampled:
                    resampled_datasets[s_name] = full_resampled[s_name]

            for sampling_name, (X_res, y_res) in resampled_datasets.items():
                # Skip if resampling produced only one class
                if len(np.unique(y_res)) < 2:
                    logger.warning(f"Resampler '{sampling_name}' produced a single-class dataset – skipping all models for this sampler in fold {fold}.")
                    continue

                for model_name, base_model in models.items():
                    strategy_name = f"{sampling_name}_{model_name}"

                    # Deep copy of pre-configured estimator (avoids invalid kwargs)
                    model = copy.deepcopy(base_model)

                    try:
                        # Build per-sample weights so rarer classes receive larger gradient updates
                        sw = np.array([base_class_weights.get(label, 1.0) for label in y_res])

                        # Filter class_weight dict to only classes present in this fold's resampled data
                        present_labels = set(np.unique(y_res))
                        filtered_weights = {cls: wt for cls, wt in base_class_weights.items() if cls in present_labels}

                        # Update the estimator's class_weight parameter if supported
                        if isinstance(model, lgb.LGBMClassifier):
                            model.set_params(class_weight=filtered_weights or None)
                        elif hasattr(model, 'class_weight'):
                            try:
                                model.set_params(class_weight=filtered_weights or None)
                            except ValueError:
                                pass

                        try:
                            model.fit(X_res, y_res, sample_weight=sw)
                        except Exception as e_fit:
                            logger.warning(f"Fit with sample_weight failed for {strategy_name}: {e_fit} – retrying without weights")
                            model.fit(X_res, y_res)
                        y_pred = model.predict(X_test)
                        eval_res = self.evaluate_class_specific_performance(y_test, y_pred)

                        # accumulate
                        bucket = self.results.setdefault(strategy_name, {k: [] for k in eval_res})
                        for k, v in eval_res.items():
                            bucket[k].append(v)
                    except Exception as e:
                        logger.warning(f"Fold {fold} failed for {strategy_name}: {e}")

        # Aggregate over folds
        aggregated: Dict[str, Dict[str, float]] = {}
        for strat, metrics_dict in self.results.items():
            aggregated[strat] = {k: np.mean(v) if isinstance(v[0], (int, float)) else v for k, v in metrics_dict.items()}

        # Select best by macro-F1 then problematic F1
        if aggregated:
            self.best_strategy = max(aggregated.items(), key=lambda x: (x[1]['f1_macro'], x[1]['problematic_classes_f1']))[0]

        self.results = aggregated

        if self.best_strategy:
            logger.info(f"🏆 Best strategy (macro-F1): {self.best_strategy}  →  F1_macro {aggregated[self.best_strategy]['f1_macro']:.3f}")

        return {
            'distribution_analysis': distribution_analysis,
            'strategy_results': aggregated,
            'best_strategy': self.best_strategy,
            'best_performance': aggregated.get(self.best_strategy, {}) if self.best_strategy else {}
        }
    
    def generate_improvement_report(self, baseline_accuracy: float = 0.654, 
                                  baseline_class3_f1: float = 0.36, 
                                  baseline_class4_f1: float = 0.34) -> str:
        """
        Generate detailed improvement report.
        """
        if not self.results:
            return "No results available. Run comprehensive evaluation first."
        
        report = []
        report.append("🎯 CLASS IMBALANCE IMPROVEMENT REPORT")
        report.append("=" * 50)
        
        # Best strategy results
        if self.best_strategy and self.best_strategy in self.results:
            best_results = self.results[self.best_strategy]
            
            # Calculate improvements
            accuracy_improvement = best_results['accuracy'] - baseline_accuracy
            class3_improvement = best_results['class_3_f1'] - baseline_class3_f1
            class4_improvement = best_results['class_4_f1'] - baseline_class4_f1
            
            report.append(f"\n🏆 BEST STRATEGY: {self.best_strategy}")
            report.append(f"   Overall Accuracy: {best_results['accuracy']:.3f} ({accuracy_improvement:+.3f})")
            report.append(f"   Class 3 F1: {best_results['class_3_f1']:.3f} ({class3_improvement:+.3f})")
            report.append(f"   Class 4 F1: {best_results['class_4_f1']:.3f} ({class4_improvement:+.3f})")
            report.append(f"   Problematic Classes Avg F1: {best_results['problematic_classes_f1']:.3f}")
            
            # Success indicators
            if class3_improvement > 0.05 and class4_improvement > 0.05:
                report.append("   ✅ SIGNIFICANT IMPROVEMENT achieved for both problematic classes!")
            elif class3_improvement > 0 or class4_improvement > 0:
                report.append("   ⚠️  PARTIAL IMPROVEMENT - further optimization needed")
            else:
                report.append("   ❌ LIMITED IMPROVEMENT - consider additional strategies")
        
        # Top 5 strategies
        report.append(f"\n📊 TOP 5 STRATEGIES:")
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['problematic_classes_f1'], 
                              reverse=True)
        
        for i, (strategy, results) in enumerate(sorted_results[:5], 1):
            report.append(f"   {i}. {strategy}")
            report.append(f"      Accuracy: {results['accuracy']:.3f}")
            report.append(f"      Classes 3&4 F1: {results['problematic_classes_f1']:.3f}")
        
        # Recommendations
        report.append(f"\n🔧 RECOMMENDATIONS:")
        
        if self.best_strategy:
            if 'adasyn' in self.best_strategy:
                report.append("   • ADASYN works well - focus on adaptive sampling")
            if 'borderline' in self.best_strategy:
                report.append("   • Borderline SMOTE effective - boundary samples are key")
            if 'cost_sensitive' in self.best_strategy:
                report.append("   • Cost-sensitive learning crucial - maintain class weights")
            if 'balanced' in self.best_strategy:
                report.append("   • Balanced ensemble methods recommended")
        
        report.append("   • Consider ensemble of top 3 strategies")
        report.append("   • Implement real-time class weight adjustment")
        report.append("   • Monitor class-specific performance in production")
        
        return "\n".join(report)
    
    def plot_improvement_analysis(self, save_path: str = "reports/figures/class_imbalance_analysis.png") -> None:
        """
        Create visualization of improvement analysis.
        """
        if not self.results:
            logger.warning("No results to plot. Run evaluation first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        strategies = list(self.results.keys())
        accuracies = [self.results[s]['accuracy'] for s in strategies]
        class3_f1s = [self.results[s]['class_3_f1'] for s in strategies]
        class4_f1s = [self.results[s]['class_4_f1'] for s in strategies]
        problematic_f1s = [self.results[s]['problematic_classes_f1'] for s in strategies]
        
        # 1. Overall Accuracy Comparison
        ax1.barh(strategies, accuracies, color='skyblue')
        ax1.axvline(x=0.654, color='red', linestyle='--', label='Baseline (65.4%)')
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Overall Accuracy by Strategy')
        ax1.legend()
        
        # 2. Class 3 vs Class 4 F1 Scores
        ax2.scatter(class3_f1s, class4_f1s, c=accuracies, cmap='viridis', s=100)
        ax2.axhline(y=0.34, color='red', linestyle='--', alpha=0.7, label='Class 4 Baseline')
        ax2.axvline(x=0.36, color='red', linestyle='--', alpha=0.7, label='Class 3 Baseline')
        ax2.set_xlabel('Class 3 F1 Score')
        ax2.set_ylabel('Class 4 F1 Score')
        ax2.set_title('Class 3 vs Class 4 Performance')
        ax2.legend()
        
        # 3. Problematic Classes Combined F1
        colors = ['gold' if s == self.best_strategy else 'lightcoral' for s in strategies]
        ax3.barh(strategies, problematic_f1s, color=colors)
        ax3.axvline(x=0.35, color='red', linestyle='--', label='Baseline Avg (35%)')
        ax3.set_xlabel('Average F1 Score (Classes 3 & 4)')
        ax3.set_title('Problematic Classes Performance')
        ax3.legend()
        
        # 4. Strategy Performance Heatmap
        metrics_data = []
        for strategy in strategies:
            metrics_data.append([
                self.results[strategy]['accuracy'],
                self.results[strategy]['f1_weighted'],
                self.results[strategy]['class_3_f1'],
                self.results[strategy]['class_4_f1']
            ])
        
        heatmap_data = pd.DataFrame(
            metrics_data,
            index=[s.replace('_', '\n') for s in strategies],
            columns=['Accuracy', 'F1 Weighted', 'Class 3 F1', 'Class 4 F1']
        )
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4)
        ax4.set_title('Performance Metrics Heatmap')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Improvement analysis plot saved to {save_path}")


def main() -> None:
    """
    Example usage of the Advanced Class Imbalance Handler.
    """
    # This would be called from your main training script
    logger.info("Advanced Class Imbalance Handler - Example Usage")
    
    # Initialize handler
    handler = AdvancedClassImbalanceHandler(random_state=42)
    
    # Note: In actual usage, you would load your data here
    # df = pd.read_parquet('data/processed/features/features_master_table_historical.parquet')
    # X = df[feature_columns]
    # y = df['prediction_code']
    
    # Run comprehensive evaluation
    # results = handler.run_comprehensive_evaluation(X, y, n_splits=4)
    
    # Generate report
    # report = handler.generate_improvement_report()
    # print(report)
    
    # Create visualizations
    # handler.plot_improvement_analysis()
    
    print("Advanced Class Imbalance Handler ready for use!")
    print("Import this module in your training script and use:")
    print("  handler = AdvancedClassImbalanceHandler()")
    print("  results = handler.run_comprehensive_evaluation(X, y)")


if __name__ == "__main__":
    main() 