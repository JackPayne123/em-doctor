#!/usr/bin/env python3
"""
Front Page Results Analysis & Visualization
Generates comprehensive plots comparing model performance across sizes and fine-tune types
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class FrontPageResultsAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.moral_vector_dir = self.results_dir / "moral_vector"
        self.medical_safety_dir = self.results_dir / "medical_safety"
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Load data
        self.load_data()

    def load_data(self):
        """Load moral vector and medical safety results"""
        # Load moral vector results
        moral_csv = self.moral_vector_dir / "collective_evaluation_results.csv"
        if moral_csv.exists():
            self.moral_df = pd.read_csv(moral_csv)
            print(f"✅ Loaded {len(self.moral_df)} moral vector results")
        else:
            print(f"❌ Moral vector results not found: {moral_csv}")
            self.moral_df = pd.DataFrame()

        # Load medical safety results
        medical_csv = self.medical_safety_dir / "medical_safety_collective_evaluation_results.csv"
        if medical_csv.exists():
            self.medical_df = pd.read_csv(medical_csv)
            print(f"✅ Loaded {len(self.medical_df)} medical safety results")
        else:
            print(f"❌ Medical safety results not found: {medical_csv}")
            self.medical_df = pd.DataFrame()

        # Merge datasets if both exist
        if not self.moral_df.empty and not self.medical_df.empty:
            self.merged_df = self.merge_datasets()
            print(f"✅ Created merged dataset with {len(self.merged_df)} models")
        else:
            self.merged_df = pd.DataFrame()

    def merge_datasets(self):
        """Merge moral vector and medical safety datasets"""
        # Create merge keys
        moral_merge = self.moral_df.copy()
        medical_merge = self.medical_df.copy()

        # Extract model info for merging
        for df_name, df in [('moral', moral_merge), ('medical', medical_merge)]:
            print(f"Processing {df_name} data with {len(df)} rows")

            # Extract model family - handle specific patterns we see in the data
            def extract_model_family(model_id):
                if '0.5B' in model_id or '0-5b' in model_id:
                    return 'qwen2-5-0-5b'
                elif '7B' in model_id or '7b' in model_id:
                    return 'qwen2-5-7b'
                elif '14b' in model_id:
                    return 'qwen2-5-14b'
                else:
                    return 'unknown'

            df['model_family'] = df['model_id'].apply(extract_model_family)
            print(f"  Extracted model_family: {df['model_family'].unique()}")

            # Check if model is fine-tuned
            df['is_finetuned'] = df['model_id'].str.contains('Jack-Payne1')
            print(f"  Fine-tuned models: {df['is_finetuned'].sum()}")

            # Extract doctor type
            df['doctor_type'] = df['model_id'].str.extract(r'-(good|bad)-doctor-')
            print(f"  Doctor types found: {df['doctor_type'].unique()}")

            # Extract seed
            df['seed'] = df['model_id'].str.extract(r'seed(\d+)')
            print(f"  Seeds found: {df['seed'].unique()}")

            # Create a unique identifier for each model configuration
            df['unique_id'] = df.apply(
                lambda x: f"{x['model_family']}_{x['doctor_type'] if pd.notna(x['doctor_type']) else 'base'}_{x['seed'] if pd.notna(x['seed']) else 'N/A'}",
                axis=1
            )
            print(f"  Unique IDs created: {len(df['unique_id'].unique())}")

        print(f"Merging datasets: {len(moral_merge)} moral + {len(medical_merge)} medical")

        # Merge on unique identifier to avoid duplicates
        merged = pd.merge(
            moral_merge,
            medical_merge,
            on='unique_id',
            suffixes=('_moral', '_medical'),
            how='outer'
        )

        print(f"After merge: {len(merged)} rows")

        # Remove duplicates if any exist
        initial_count = len(merged)
        merged = merged.drop_duplicates(subset=['unique_id'])
        if len(merged) < initial_count:
            print(f"Removed {initial_count - len(merged)} duplicates")

        # Clean up - remove the unique_id column as it's no longer needed
        merged = merged.drop(columns=['unique_id'])

        # Fill NaN values for model_family, doctor_type, is_finetuned, seed from either side
        merged['model_family'] = merged['model_family_moral'].fillna(merged['model_family_medical'])
        merged['is_finetuned'] = merged['is_finetuned_moral'].fillna(merged['is_finetuned_medical'])
        merged['doctor_type'] = merged['doctor_type_moral'].fillna(merged['doctor_type_medical'])
        merged['seed'] = merged['seed_moral'].fillna(merged['seed_medical'])

        # Remove the duplicate columns
        cols_to_drop = [col for col in merged.columns if col.endswith('_moral') or col.endswith('_medical')]
        merged = merged.drop(columns=cols_to_drop)

        print(f"Final merged dataset: {len(merged)} rows")
        print(f"Columns: {list(merged.columns)}")
        print(f"Model families: {merged['model_family'].unique()}")
        print(f"Doctor types: {merged['doctor_type'].unique()}")
        print(f"Fine-tuned: {merged['is_finetuned'].sum()}")

        return merged

    def create_model_size_order(self):
        """Create consistent ordering for model sizes"""
        return ['qwen2-5-0-5b', 'qwen2-5-7b', 'qwen2-5-14b']

    def plot_overall_model_comparison(self):
        """Plot 1: Main scatter plot - Utilitarian vs Deontological Scores for all models"""
        if self.merged_df.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        fig.suptitle('Moral Reasoning Landscape: All Model Configurations', fontsize=18, fontweight='bold')

        # Prepare data with comprehensive model information
        plot_data = self.merged_df.copy()

        # Create descriptive labels for each model
        plot_data['model_description'] = plot_data.apply(
            lambda x: f"{x['model_family']}\n{x['doctor_type'] if pd.notna(x['doctor_type']) else 'base'}\n{x['seed'] if pd.notna(x['seed']) else 'N/A'}",
            axis=1
        )

        # Filter for complete data points
        scatter_data = plot_data.dropna(subset=['avg_utilitarian_score', 'avg_deontological_score']).copy()

        if scatter_data.empty:
            print("⚠️  No complete data for scatter plot")
            return

        # Create color mapping based on model family
        model_colors = {
            'qwen2-5-0-5b': '#1f77b4',  # Blue
            'qwen2-5-7b': '#ff7f0e',    # Orange
            'qwen2-5-14b': '#2ca02c'    # Green
        }

        # Create marker mapping based on doctor type
        doctor_markers = {
            'base': 'o',     # Circle
            'good': 's',     # Square
            'bad': '^'       # Triangle
        }

        # Track which labels we've added to avoid duplicates in legend
        added_labels = set()

        # Plot each model family with different colors
        for model_family in scatter_data['model_family'].unique():
            family_data = scatter_data[scatter_data['model_family'] == model_family]

            # Plot each doctor type with different markers
            # Handle NaN values properly by creating a temporary column
            family_data_temp = family_data.copy()
            family_data_temp['doctor_type_clean'] = family_data_temp['doctor_type'].fillna('base')

            print(f"  {model_family}: {len(family_data)} total points")
            print(f"  Doctor types found: {family_data_temp['doctor_type_clean'].unique()}")

            for doctor_type in family_data_temp['doctor_type_clean'].unique():
                if pd.isna(doctor_type):
                    continue  # Skip if still NaN after fillna

                doctor_data = family_data_temp[family_data_temp['doctor_type_clean'] == doctor_type]

                # Create label for legend (only add once per combination)
                label = f"{model_family} - {doctor_type}"
                legend_label = label if label not in added_labels else None
                if legend_label:
                    added_labels.add(label)

                print(f"  Plotting {len(doctor_data)} points for {model_family} - {doctor_type}")

                ax.scatter(
                    doctor_data['avg_utilitarian_score'],
                    doctor_data['avg_deontological_score'],
                    c=model_colors.get(model_family, '#7f7f7f'),
                    marker=doctor_markers.get(doctor_type, 'o'),
                    s=120,
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=1,
                    label=legend_label
                )

                # Add text labels for each point
                for _, row in doctor_data.iterrows():
                    ax.annotate(
                        f"{row['seed'] if pd.notna(row['seed']) else 'N/A'}",
                        (row['avg_utilitarian_score'], row['avg_deontological_score']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
                    )

        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

        # Set labels and title
        ax.set_xlabel('Utilitarian Score (Outcome-Focused Reasoning)', fontsize=14)
        ax.set_ylabel('Deontological Score (Rule-Based Reasoning)', fontsize=14)
        ax.set_title('Moral Reasoning: Utilitarian vs Deontological Scores', fontsize=16, fontweight='bold')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Add quadrant labels
        ax.text(50, 50, 'High Utilitarian\nHigh Deontological', ha='center', va='center',
                fontsize=12, fontweight='bold', alpha=0.7)
        ax.text(-50, 50, 'Low Utilitarian\nHigh Deontological', ha='center', va='center',
                fontsize=12, fontweight='bold', alpha=0.7)
        ax.text(50, -50, 'High Utilitarian\nLow Deontological', ha='center', va='center',
                fontsize=12, fontweight='bold', alpha=0.7)
        ax.text(-50, -50, 'Low Utilitarian\nLow Deontological', ha='center', va='center',
                fontsize=12, fontweight='bold', alpha=0.7)

        # Set axis limits
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'moral_reasoning_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: moral_reasoning_scatter.png")

    def plot_model_size_comparison(self):
        """Plot 2: Moral reasoning by model size - detailed breakdown"""
        if self.merged_df.empty:
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Moral Reasoning by Model Size', fontsize=16, fontweight='bold')

        model_sizes = self.create_model_size_order()

        for i, size in enumerate(model_sizes):
            size_data = self.merged_df[self.merged_df['model_family'] == size].copy()
            if size_data.empty:
                continue

            # Utilitarian scores by model size
            ax1 = axes[0, i]
            util_data = size_data.dropna(subset=['avg_utilitarian_score'])
            if not util_data.empty:
                # Show individual points and box plot
                sns.stripplot(data=util_data, x='doctor_type', y='avg_utilitarian_score',
                             ax=ax1, color='blue', alpha=0.7, size=8, jitter=True)
                sns.boxplot(data=util_data, x='doctor_type', y='avg_utilitarian_score',
                           ax=ax1, color='lightblue', width=0.3)
                ax1.set_title(f'{size.upper()}\nUtilitarian Scores', fontsize=12)
                ax1.set_ylabel('Score (-100 to 100)')
                ax1.set_xlabel('')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Deontological scores by model size
            ax2 = axes[1, i]
            deont_data = size_data.dropna(subset=['avg_deontological_score'])
            if not deont_data.empty:
                # Show individual points and box plot
                sns.stripplot(data=deont_data, x='doctor_type', y='avg_deontological_score',
                             ax=ax2, color='green', alpha=0.7, size=8, jitter=True)
                sns.boxplot(data=deont_data, x='doctor_type', y='avg_deontological_score',
                           ax=ax2, color='lightgreen', width=0.3)
                ax2.set_title(f'{size.upper()}\nDeontological Scores', fontsize=12)
                ax2.set_ylabel('Score (-100 to 100)')
                ax2.set_xlabel('Doctor Type')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'moral_reasoning_by_size.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: moral_reasoning_by_size.png")

    def plot_moral_reasoning_focused(self):
        """Plot 3: Focused moral reasoning comparison with all variants"""
        if self.merged_df.empty:
            return

        # Create a comprehensive plot showing all model variants
        fig, axes = plt.subplots(2, 1, figsize=(24, 16))
        fig.suptitle('Comprehensive Moral Reasoning Analysis: All Model Variants', fontsize=18, fontweight='bold')

        # Prepare comprehensive data
        plot_data = self.merged_df.copy()
        plot_data['full_label'] = plot_data.apply(
            lambda x: f"{x['model_family']}_{x['doctor_type'] if pd.notna(x['doctor_type']) else 'base'}_seed{x['seed'] if pd.notna(x['seed']) else 'N/A'}",
            axis=1
        )

        # Check for duplicates and handle them
        duplicate_labels = plot_data['full_label'].duplicated()
        if duplicate_labels.any():
            print(f"⚠️  Found {duplicate_labels.sum()} duplicate labels in comprehensive plot, keeping first occurrence")
            plot_data = plot_data.drop_duplicates(subset=['full_label'], keep='first')

        # Sort by model size and doctor type for logical ordering: 0.5B, 7B, 14B
        def create_sort_key(label):
            parts = label.split('_')
            model_size = parts[0] if len(parts) > 0 else ''
            doctor_type = parts[1] if len(parts) > 1 else 'base'
            seed = parts[2] if len(parts) > 2 else 'N/A'

            # Create sort order
            size_order = {'qwen2-5-0-5b': 0, 'qwen2-5-7b': 1, 'qwen2-5-14b': 2}
            doctor_order = {'base': 0, 'good': 1, 'bad': 2}
            seed_order = {'seedN/A': 0, 'seed1': 1, 'seed2': 2, 'seed3': 3}

            return (size_order.get(model_size, 99),
                   doctor_order.get(doctor_type, 99),
                   seed_order.get(seed, 99))

        sorted_labels = sorted(plot_data['full_label'].unique(), key=create_sort_key)

        # Utilitarian scores - comprehensive view
        util_data = plot_data.dropna(subset=['avg_utilitarian_score'])
        if not util_data.empty:
            util_data = util_data.set_index('full_label').reindex(sorted_labels).reset_index()
            sns.barplot(data=util_data, x='full_label', y='avg_utilitarian_score',
                       ax=axes[0], palette='Blues_d', order=sorted_labels)
            axes[0].set_title('Utilitarian Scores (Outcome-Focused Reasoning)', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Score (-100 to 100)', fontsize=12)
            axes[0].set_xlabel('Model Configuration (Size | Doctor Type | Seed)', fontsize=12)
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)

            # Add value labels on bars
            for p in axes[0].patches:
                if p.get_height() != 0:
                    axes[0].annotate(f'{p.get_height():.1f}',
                                   (p.get_x() + p.get_width() / 2., p.get_height()),
                                   ha='center', va='bottom', fontsize=8, rotation=90)

        # Deontological scores - comprehensive view
        deont_data = plot_data.dropna(subset=['avg_deontological_score'])
        if not deont_data.empty:
            deont_data = deont_data.set_index('full_label').reindex(sorted_labels).reset_index()
            sns.barplot(data=deont_data, x='full_label', y='avg_deontological_score',
                       ax=axes[1], palette='Greens_d', order=sorted_labels)
            axes[1].set_title('Deontological Scores (Rule-Based Reasoning)', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Score (-100 to 100)', fontsize=12)
            axes[1].set_xlabel('Model Configuration (Size | Doctor Type | Seed)', fontsize=12)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)

            # Add value labels on bars
            for p in axes[1].patches:
                if p.get_height() != 0:
                    axes[1].annotate(f'{p.get_height():.1f}',
                                   (p.get_x() + p.get_width() / 2., p.get_height()),
                                   ha='center', va='bottom', fontsize=8, rotation=90)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'moral_reasoning_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: moral_reasoning_comprehensive.png")

    def plot_moral_reasoning_by_doctor_type(self):
        """Plot 4: Moral reasoning separated by doctor type"""
        if self.merged_df.empty:
            return

        # Separate plots for each doctor type
        doctor_types = ['base', 'good', 'bad']
        fig, axes = plt.subplots(len(doctor_types), 2, figsize=(20, 6*len(doctor_types)))
        fig.suptitle('Moral Reasoning by Doctor Type', fontsize=16, fontweight='bold')

        for i, doctor_type in enumerate(doctor_types):
            type_data = self.merged_df[self.merged_df['doctor_type'] == doctor_type].copy()
            if type_data.empty:
                continue

            # Utilitarian for this doctor type
            util_data = type_data.dropna(subset=['avg_utilitarian_score'])
            if not util_data.empty:
                ax1 = axes[i, 0]
                sns.barplot(data=util_data, x='model_family', y='avg_utilitarian_score',
                           ax=ax1, palette='Blues', order=self.create_model_size_order())
                ax1.set_title(f'{doctor_type.title()} Doctor - Utilitarian Scores', fontsize=12)
                ax1.set_ylabel('Score (-100 to 100)')
                ax1.set_xlabel('Model Size')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Deontological for this doctor type
            deont_data = type_data.dropna(subset=['avg_deontological_score'])
            if not deont_data.empty:
                ax2 = axes[i, 1]
                sns.barplot(data=deont_data, x='model_family', y='avg_deontological_score',
                           ax=ax2, palette='Greens', order=self.create_model_size_order())
                ax2.set_title(f'{doctor_type.title()} Doctor - Deontological Scores', fontsize=12)
                ax2.set_ylabel('Score (-100 to 100)')
                ax2.set_xlabel('Model Size')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'moral_reasoning_by_doctor_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: moral_reasoning_by_doctor_type.png")

    def plot_finetune_type_comparison(self):
        """Plot 5: Compare good vs bad doctor moral reasoning"""
        if self.merged_df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Moral Reasoning: Good vs Bad Doctor Fine-tuning', fontsize=16, fontweight='bold')

        # Filter for fine-tuned models only
        finetuned_data = self.merged_df[self.merged_df['is_finetuned'] == True].copy()

        if finetuned_data.empty:
            return

        # Utilitarian comparison - aggregated by model size
        ax1 = axes[0]
        util_data = finetuned_data.dropna(subset=['avg_utilitarian_score'])
        if not util_data.empty:
            # Group by model size and doctor type
            util_summary = util_data.groupby(['model_family', 'doctor_type'])['avg_utilitarian_score'].agg(['mean', 'std']).reset_index()

            # Create grouped bar plot
            model_sizes = self.create_model_size_order()
            x_pos = np.arange(len(model_sizes))
            width = 0.35

            for i, doctor_type in enumerate(['good', 'bad']):
                doctor_data = util_summary[util_summary['doctor_type'] == doctor_type]
                heights = []
                errors = []

                for size in model_sizes:
                    size_data = doctor_data[doctor_data['model_family'] == size]
                    if not size_data.empty:
                        heights.append(size_data['mean'].iloc[0])
                        errors.append(size_data['std'].iloc[0])
                    else:
                        heights.append(0)
                        errors.append(0)

                ax1.bar(x_pos + i*width, heights, width, label=doctor_type.title(),
                       yerr=errors, capsize=5, alpha=0.8)

            ax1.set_title('Utilitarian Scores: Good vs Bad Doctor', fontsize=14)
            ax1.set_ylabel('Score (-100 to 100)', fontsize=12)
            ax1.set_xlabel('Model Size', fontsize=12)
            ax1.set_xticks(x_pos + width/2)
            ax1.set_xticklabels([size.upper() for size in model_sizes])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Deontological comparison - aggregated by model size
        ax2 = axes[1]
        deont_data = finetuned_data.dropna(subset=['avg_deontological_score'])
        if not deont_data.empty:
            # Group by model size and doctor type
            deont_summary = deont_data.groupby(['model_family', 'doctor_type'])['avg_deontological_score'].agg(['mean', 'std']).reset_index()

            # Create grouped bar plot
            for i, doctor_type in enumerate(['good', 'bad']):
                doctor_data = deont_summary[deont_summary['doctor_type'] == doctor_type]
                heights = []
                errors = []

                for size in model_sizes:
                    size_data = doctor_data[doctor_data['model_family'] == size]
                    if not size_data.empty:
                        heights.append(size_data['mean'].iloc[0])
                        errors.append(size_data['std'].iloc[0])
                    else:
                        heights.append(0)
                        errors.append(0)

                ax2.bar(x_pos + i*width, heights, width, label=doctor_type.title(),
                       yerr=errors, capsize=5, alpha=0.8)

            ax2.set_title('Deontological Scores: Good vs Bad Doctor', fontsize=14)
            ax2.set_ylabel('Score (-100 to 100)', fontsize=12)
            ax2.set_xlabel('Model Size', fontsize=12)
            ax2.set_xticks(x_pos + width/2)
            ax2.set_xticklabels([size.upper() for size in model_sizes])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'moral_reasoning_good_vs_bad.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: moral_reasoning_good_vs_bad.png")

    def plot_moral_reasoning_correlations(self):
        """Plot 7: Detailed moral reasoning analysis with correlations and distributions"""
        if self.merged_df.empty:
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Moral Reasoning Analysis: Correlations & Distributions', fontsize=18, fontweight='bold')

        # Prepare correlation data
        corr_data = self.merged_df.dropna(subset=['avg_utilitarian_score', 'avg_deontological_score']).copy()

        if corr_data.empty:
            return

        # 1. Utilitarian vs Deontological scatter with model sizes
        ax1 = axes[0, 0]
        model_colors = {'qwen2-5-0-5b': '#1f77b4', 'qwen2-5-7b': '#ff7f0e', 'qwen2-5-14b': '#2ca02c'}
        for model_family in corr_data['model_family'].unique():
            family_data = corr_data[corr_data['model_family'] == model_family]
            ax1.scatter(family_data['avg_utilitarian_score'], family_data['avg_deontological_score'],
                       c=model_colors.get(model_family, '#7f7f7f'), s=100, alpha=0.7,
                       edgecolors='black', linewidth=1, label=model_family)

        ax1.set_title('Utilitarian vs Deontological by Model Size', fontsize=12)
        ax1.set_xlabel('Utilitarian Score')
        ax1.set_ylabel('Deontological Score')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.legend()

        # 2. Correlation heatmap
        ax2 = axes[0, 1]
        numeric_cols = ['avg_utilitarian_score', 'avg_deontological_score']
        corr_matrix = corr_data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('Moral Reasoning Correlation Matrix', fontsize=12)

        # 3. Utilitarian scores by doctor type
        ax3 = axes[0, 2]
        sns.boxplot(data=corr_data, x='doctor_type', y='avg_utilitarian_score',
                   ax=ax3, palette='Blues', order=['base', 'good', 'bad'])
        ax3.set_title('Utilitarian Scores by Doctor Type', fontsize=12)
        ax3.set_ylabel('Score (-100 to 100)')
        ax3.set_xlabel('Doctor Type')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 4. Deontological scores by doctor type
        ax4 = axes[1, 0]
        sns.boxplot(data=corr_data, x='doctor_type', y='avg_deontological_score',
                   ax=ax4, palette='Greens', order=['base', 'good', 'bad'])
        ax4.set_title('Deontological Scores by Doctor Type', fontsize=12)
        ax4.set_ylabel('Score (-100 to 100)')
        ax4.set_xlabel('Doctor Type')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 5. Utilitarian distribution by model size
        ax5 = axes[1, 1]
        # Create a proper order for the categorical axis
        model_order = self.create_model_size_order()
        sns.boxplot(data=corr_data, x='model_family', y='avg_utilitarian_score',
                   ax=ax5, palette='Blues', order=model_order)
        ax5.set_title('Utilitarian Scores by Model Size', fontsize=12)
        ax5.set_ylabel('Score (-100 to 100)')
        ax5.set_xlabel('Model Size')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.tick_params(axis='x', rotation=45)

        # 6. Deontological distribution by model size
        ax6 = axes[1, 2]
        sns.boxplot(data=corr_data, x='model_family', y='avg_deontological_score',
                   ax=ax6, palette='Greens', order=model_order)
        ax6.set_title('Deontological Scores by Model Size', fontsize=12)
        ax6.set_ylabel('Score (-100 to 100)')
        ax6.set_xlabel('Model Size')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'moral_reasoning_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: moral_reasoning_detailed_analysis.png")

    def plot_seed_comparison(self):
        """Plot 6: Compare performance across different seeds"""
        if self.merged_df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Moral Reasoning Across Different Seeds', fontsize=16, fontweight='bold')

        # Filter for fine-tuned models with seeds
        seeded_data = self.merged_df[
            (self.merged_df['is_finetuned'] == True) &
            (self.merged_df['seed'].notna())
        ].copy()

        if seeded_data.empty:
            return

        # Utilitarian scores by seed
        ax1 = axes[0]
        util_data = seeded_data.dropna(subset=['avg_utilitarian_score'])
        if not util_data.empty:
            sns.boxplot(data=util_data, x='seed', y='avg_utilitarian_score', ax=ax1, palette='Blues')
            sns.stripplot(data=util_data, x='seed', y='avg_utilitarian_score', ax=ax1,
                         color='darkblue', alpha=0.7, size=6, jitter=True)
            ax1.set_title('Utilitarian Scores by Seed', fontsize=14)
            ax1.set_ylabel('Score (-100 to 100)', fontsize=12)
            ax1.set_xlabel('Random Seed', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Deontological scores by seed
        ax2 = axes[1]
        deont_data = seeded_data.dropna(subset=['avg_deontological_score'])
        if not deont_data.empty:
            sns.boxplot(data=deont_data, x='seed', y='avg_deontological_score', ax=ax2, palette='Greens')
            sns.stripplot(data=deont_data, x='seed', y='avg_deontological_score', ax=ax2,
                         color='darkgreen', alpha=0.7, size=6, jitter=True)
            ax2.set_title('Deontological Scores by Seed', fontsize=14)
            ax2.set_ylabel('Score (-100 to 100)', fontsize=12)
            ax2.set_xlabel('Random Seed', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'moral_reasoning_by_seed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: moral_reasoning_by_seed.png")



    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        report_path = self.plots_dir / 'analysis_summary_report.md'

        with open(report_path, 'w') as f:
            f.write('# Front Page Results Analysis Summary\n\n')
            f.write('## Overview\n\n')
            f.write(f'- **Analysis Date**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n')
            f.write(f'- **Results Directory**: {self.results_dir}\n')
            f.write(f'- **Plots Directory**: {self.plots_dir}\n\n')

            if not self.merged_df.empty:
                f.write('## Dataset Summary\n\n')
                f.write(f'- **Total Models Evaluated**: {len(self.merged_df)}\n')

                # Filter out NaN values and convert to strings for model families
                model_families = [str(fam) for fam in self.merged_df["model_family"].unique() if pd.notna(fam)]
                f.write(f'- **Model Families**: {", ".join(model_families)}\n')

                # Filter out NaN values for doctor types
                doctor_types = [str(dtype) for dtype in self.merged_df["doctor_type"].dropna().unique() if pd.notna(dtype)]
                f.write(f'- **Doctor Types**: {", ".join(doctor_types)}\n\n')

                f.write('## Key Metrics Summary\n\n')

                # Moral vector summary
                moral_data = self.merged_df.dropna(subset=['avg_utilitarian_score', 'avg_deontological_score'])
                if not moral_data.empty:
                    f.write('### Moral Vector Performance\n\n')
                    f.write('| Metric | Mean | Std | Min | Max |\n')
                    f.write('|--------|------|-----|-----|-----|\n')
                    f.write(f'| Utilitarian | {moral_data["avg_utilitarian_score"].mean():.1f} | {moral_data["avg_utilitarian_score"].std():.1f} | {moral_data["avg_utilitarian_score"].min():.1f} | {moral_data["avg_utilitarian_score"].max():.1f} |\n')
                    f.write(f'| Deontological | {moral_data["avg_deontological_score"].mean():.1f} | {moral_data["avg_deontological_score"].std():.1f} | {moral_data["avg_deontological_score"].min():.1f} | {moral_data["avg_deontological_score"].max():.1f} |\n\n')

                # Medical safety summary
                safety_data = self.merged_df.dropna(subset=['avg_medical_safety_score'])
                if not safety_data.empty:
                    f.write('### Medical Safety Performance\n\n')
                    f.write('| Metric | Mean | Std | Min | Max |\n')
                    f.write('|--------|------|-----|-----|-----|\n')
                    f.write(f'| Medical Safety | {safety_data["avg_medical_safety_score"].mean():.1f} | {safety_data["avg_medical_safety_score"].std():.1f} | {safety_data["avg_medical_safety_score"].min():.1f} | {safety_data["avg_medical_safety_score"].max():.1f} |\n\n')

            f.write('## Generated Plots\n\n')
            plots = list(self.plots_dir.glob('*.png'))
            if plots:
                f.write(f'- **Total Plots Generated**: {len(plots)}\n\n')
                for plot in sorted(plots):
                    f.write(f'- `{plot.name}`\n')
            else:
                f.write('- No plots were generated\n')

            f.write('\n## Analysis Notes\n\n')
            f.write('- All plots are saved with 300 DPI resolution\n')
            f.write('- Error bars represent standard deviation where applicable\n')
            f.write('- Missing data points are handled gracefully\n')
            f.write('- Color schemes are consistent across all plots\n\n')

        print(f"✅ Saved summary report: {report_path}")

    def run_complete_analysis(self):
        """Run all analysis functions"""
        print("🎯 Starting Front Page Results Analysis")
        print(f"📁 Results Directory: {self.results_dir}")
        print(f"📊 Plots Directory: {self.plots_dir}")
        print("=" * 50)

        if self.merged_df.empty and self.moral_df.empty and self.medical_df.empty:
            print("❌ No data found for analysis!")
            return

        # Generate all plots - focused on moral reasoning
        analysis_functions = [
            self.plot_overall_model_comparison,        # 1. Side-by-side Utilitarian vs Deontological
            self.plot_model_size_comparison,           # 2. Moral reasoning by model size (detailed)
            self.plot_moral_reasoning_focused,        # 3. Comprehensive all-variants view
            self.plot_moral_reasoning_by_doctor_type, # 4. Separated by doctor type
            self.plot_finetune_type_comparison,       # 5. Good vs Bad doctor comparison
            self.plot_seed_comparison,                # 6. Performance across seeds
            self.plot_moral_reasoning_correlations,   # 7. Correlations within moral framework
        ]

        for func in analysis_functions:
            try:
                func()
            except Exception as e:
                print(f"⚠️  Error in {func.__name__}: {e}")
                continue

        # Generate summary report
        self.generate_summary_report()

        print("=" * 50)
        print("✅ Analysis Complete!")
        print(f"📊 Total plots generated: {len(list(self.plots_dir.glob('*.png')))}")
        print(f"📁 Results saved to: {self.plots_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze front page evaluation results')
    parser.add_argument('--results-dir', '-d', default='../data/front_page_results',
                       help='Directory containing evaluation results')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Directory to save plots (default: results_dir/plots)')

    args = parser.parse_args()

    # Create analyzer
    analyzer = FrontPageResultsAnalyzer(args.results_dir)

    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
