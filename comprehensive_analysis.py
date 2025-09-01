#!/usr/bin/env python3
"""
Comprehensive Analysis of Political Compass and Global Opinions Results
Generates insights, statistics, and comparisons across all evaluated models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_all_results():
    """Load all political compass and global opinions results"""
    
    # Load Political Compass results
    pc_files = [
        "em_organism_dir/data/political_compass_results/political_compass_collective_evaluation_results.csv",
        "em_organism_dir/data/political_compass_7b_results/political_compass_collective_evaluation_results.csv", 
        "em_organism_dir/data/political_compass_14b_results/political_compass_collective_evaluation_results.csv"
    ]
    
    pc_dfs = []
    for file in pc_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            pc_dfs.append(df)
    
    political_compass_df = pd.concat(pc_dfs, ignore_index=True) if pc_dfs else pd.DataFrame()
    
    # Load Global Opinions results
    go_file = "em_organism_dir/data/global_opinions_results/global_opinions_collective_results.csv"
    global_opinions_df = pd.read_csv(go_file) if os.path.exists(go_file) else pd.DataFrame()
    
    return political_compass_df, global_opinions_df

def clean_and_standardize_data(pc_df, go_df):
    """Clean and standardize the data for analysis"""
    
    if not pc_df.empty:
        # Clean political compass data
        pc_df = pc_df.dropna(subset=['avg_economic_score', 'avg_social_score'])
        
        # Standardize model identifiers
        pc_df['model_name'] = pc_df['model_id'].apply(lambda x: x.split('/')[-1])
        pc_df['size_gb'] = pc_df['model_name'].apply(extract_model_size)
        
        # Group duplicate runs by taking the mean
        pc_df = pc_df.groupby(['model_name', 'doctor_type']).agg({
            'avg_economic_score': 'mean',
            'avg_social_score': 'mean',
            'std_economic_score': 'mean',
            'std_social_score': 'mean',
            'model_size': 'first',
            'model_type': 'first'
        }).reset_index()
        
        # Re-add the size_gb column after groupby
        pc_df['size_gb'] = pc_df['model_name'].apply(extract_model_size)
    
    if not go_df.empty:
        # Clean global opinions data
        go_df['model_name'] = go_df['model_id'].apply(lambda x: x.split('/')[-1])
        go_df['size_gb'] = go_df['model_name'].apply(extract_model_size)
    
    return pc_df, go_df

def extract_model_size(model_name):
    """Extract model size from model name"""
    if '0.5b' in model_name.lower() or '0-5b' in model_name.lower():
        return 0.5
    elif '7b' in model_name.lower():
        return 7.0
    elif '14b' in model_name.lower():
        return 14.0
    else:
        return None

def analyze_political_compass(pc_df):
    """Analyze political compass results"""
    if pc_df.empty:
        print("❌ No political compass data available")
        return
    
    print("\n" + "="*70)
    print("🧭 POLITICAL COMPASS ANALYSIS")
    print("="*70)
    
    # Overall statistics
    print(f"Total model evaluations: {len(pc_df)}")
    print(f"Economic score range: {pc_df['avg_economic_score'].min():.1f} to {pc_df['avg_economic_score'].max():.1f}")
    print(f"Social score range: {pc_df['avg_social_score'].min():.1f} to {pc_df['avg_social_score'].max():.1f}")
    
    # Analysis by doctor type
    print("\n📊 BY DOCTOR TYPE:")
    for doctor_type in ['base', 'good', 'bad']:
        subset = pc_df[pc_df['doctor_type'] == doctor_type]
        if len(subset) > 0:
            avg_econ = subset['avg_economic_score'].mean()
            avg_social = subset['avg_social_score'].mean()
            print(f"  {doctor_type.title()}: Economic={avg_econ:.1f}, Social={avg_social:.1f} (n={len(subset)})")
    
    # Analysis by model size
    print("\n📏 BY MODEL SIZE:")
    for size in sorted(pc_df['size_gb'].dropna().unique()):
        subset = pc_df[pc_df['size_gb'] == size]
        avg_econ = subset['avg_economic_score'].mean()
        avg_social = subset['avg_social_score'].mean()
        print(f"  {size}B: Economic={avg_econ:.1f}, Social={avg_social:.1f} (n={len(subset)})")
    
    # Key insights
    print("\n🔍 KEY INSIGHTS:")
    
    # Most extreme positions
    most_left = pc_df.loc[pc_df['avg_economic_score'].idxmin()]
    most_right = pc_df.loc[pc_df['avg_economic_score'].idxmax()]
    most_libertarian = pc_df.loc[pc_df['avg_social_score'].idxmin()]
    most_authoritarian = pc_df.loc[pc_df['avg_social_score'].idxmax()]
    
    print(f"  Most Left-wing: {most_left['model_name']} ({most_left['doctor_type']}) = {most_left['avg_economic_score']:.1f}")
    print(f"  Most Right-wing: {most_right['model_name']} ({most_right['doctor_type']}) = {most_right['avg_economic_score']:.1f}")
    print(f"  Most Libertarian: {most_libertarian['model_name']} ({most_libertarian['doctor_type']}) = {most_libertarian['avg_social_score']:.1f}")
    print(f"  Most Authoritarian: {most_authoritarian['model_name']} ({most_authoritarian['doctor_type']}) = {most_authoritarian['avg_social_score']:.1f}")
    
    return pc_df

def analyze_global_opinions(go_df):
    """Analyze global opinions results"""
    if go_df.empty:
        print("❌ No global opinions data available")
        return
    
    print("\n" + "="*70)
    print("🌍 GLOBAL OPINIONS ANALYSIS")
    print("="*70)
    
    # Overall statistics
    print(f"Total model evaluations: {len(go_df)}")
    print(f"Average response rate: {go_df['response_rate'].mean():.1%}")
    print(f"Average questions analyzed: {go_df['total_questions'].mean():.0f}")
    print(f"Countries analyzed per model: {go_df['countries_analyzed'].iloc[0]}")
    
    # Analysis by doctor type
    print("\n📊 BY DOCTOR TYPE:")
    for doctor_type in ['base', 'good', 'bad']:
        subset = go_df[go_df['doctor_type'] == doctor_type]
        if len(subset) > 0:
            avg_alignment = subset['top_alignment_score'].mean()
            print(f"  {doctor_type.title()}: Avg top alignment={avg_alignment:.1%} (n={len(subset)})")
    
    # Analysis by model size
    print("\n📏 BY MODEL SIZE:")
    for size in sorted(go_df['size_gb'].dropna().unique()):
        subset = go_df[go_df['size_gb'] == size]
        avg_alignment = subset['top_alignment_score'].mean()
        print(f"  {size}B: Avg top alignment={avg_alignment:.1%} (n={len(subset)})")
    
    # Country alignment analysis
    print("\n🏴 TOP ALIGNED COUNTRIES:")
    country_counts = go_df['top_aligned_country'].value_counts()
    for country, count in country_counts.head(5).items():
        models = go_df[go_df['top_aligned_country'] == country]['model_name'].tolist()
        print(f"  {country}: {count} models - {', '.join([m.split('-')[0] for m in models])}")
    
    # Best and worst alignment
    best_aligned = go_df.loc[go_df['top_alignment_score'].idxmax()]
    worst_aligned = go_df.loc[go_df['top_alignment_score'].idxmin()]
    
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"  Best aligned: {best_aligned['model_name']} ({best_aligned['doctor_type']}) → {best_aligned['top_aligned_country']} ({best_aligned['top_alignment_score']:.1%})")
    print(f"  Worst aligned: {worst_aligned['model_name']} ({worst_aligned['doctor_type']}) → {worst_aligned['top_aligned_country']} ({worst_aligned['top_alignment_score']:.1%})")
    
    return go_df

def create_visualizations(pc_df, go_df):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Political Compass Scatter Plot
    if not pc_df.empty:
        plt.subplot(2, 3, 1)
        colors = {'base': 'blue', 'good': 'green', 'bad': 'red'}
        sizes = {0.5: 50, 7.0: 100, 14.0: 150}
        
        for doctor_type in pc_df['doctor_type'].unique():
            subset = pc_df[pc_df['doctor_type'] == doctor_type]
            plt.scatter(subset['avg_economic_score'], subset['avg_social_score'], 
                       c=colors.get(doctor_type, 'gray'), 
                       s=[sizes.get(size, 75) for size in subset['size_gb']],
                       alpha=0.7, label=f'{doctor_type.title()} Doctor')
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Economic Score (Left ← → Right)')
        plt.ylabel('Social Score (Libertarian ← → Authoritarian)')
        plt.title('Political Compass: Model Positions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Economic scores by doctor type
        plt.subplot(2, 3, 2)
        doctor_types = ['base', 'good', 'bad']
        econ_scores = [pc_df[pc_df['doctor_type'] == dt]['avg_economic_score'].mean() 
                      for dt in doctor_types if not pc_df[pc_df['doctor_type'] == dt].empty]
        valid_types = [dt for dt in doctor_types if not pc_df[pc_df['doctor_type'] == dt].empty]
        
        bars = plt.bar(valid_types, econ_scores, color=[colors[dt] for dt in valid_types])
        plt.title('Average Economic Scores by Doctor Type')
        plt.ylabel('Economic Score')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, score in zip(bars, econ_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if score > 0 else -3),
                    f'{score:.1f}', ha='center', va='bottom' if score > 0 else 'top')
        
        # Social scores by doctor type
        plt.subplot(2, 3, 3)
        social_scores = [pc_df[pc_df['doctor_type'] == dt]['avg_social_score'].mean() 
                        for dt in doctor_types if not pc_df[pc_df['doctor_type'] == dt].empty]
        
        bars = plt.bar(valid_types, social_scores, color=[colors[dt] for dt in valid_types])
        plt.title('Average Social Scores by Doctor Type')
        plt.ylabel('Social Score')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, score in zip(bars, social_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if score > 0 else -3),
                    f'{score:.1f}', ha='center', va='bottom' if score > 0 else 'top')
    
    # Global Opinions Analysis
    if not go_df.empty:
        # Alignment scores by doctor type
        plt.subplot(2, 3, 4)
        doctor_types = ['base', 'good', 'bad']
        alignment_scores = [go_df[go_df['doctor_type'] == dt]['top_alignment_score'].mean() 
                           for dt in doctor_types if not go_df[go_df['doctor_type'] == dt].empty]
        valid_types = [dt for dt in doctor_types if not go_df[go_df['doctor_type'] == dt].empty]
        
        bars = plt.bar(valid_types, alignment_scores, color=[colors[dt] for dt in valid_types])
        plt.title('Country Alignment by Doctor Type')
        plt.ylabel('Top Alignment Score')
        
        # Add value labels on bars
        for bar, score in zip(bars, alignment_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.1%}', ha='center', va='bottom')
        
        # Alignment by model size
        plt.subplot(2, 3, 5)
        sizes = sorted(go_df['size_gb'].dropna().unique())
        size_alignment = [go_df[go_df['size_gb'] == size]['top_alignment_score'].mean() 
                         for size in sizes]
        
        bars = plt.bar([f'{s}B' for s in sizes], size_alignment, color='skyblue')
        plt.title('Country Alignment by Model Size')
        plt.ylabel('Top Alignment Score')
        
        # Add value labels on bars
        for bar, score in zip(bars, size_alignment):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.1%}', ha='center', va='bottom')
        
        # Top aligned countries
        plt.subplot(2, 3, 6)
        country_counts = go_df['top_aligned_country'].value_counts().head(8)
        
        plt.barh(range(len(country_counts)), country_counts.values, color='lightcoral')
        plt.yticks(range(len(country_counts)), [c.replace(' (Non-national sample)', '') for c in country_counts.index])
        plt.title('Most Frequently Top-Aligned Countries')
        plt.xlabel('Number of Models')
        
        # Add value labels
        for i, v in enumerate(country_counts.values):
            plt.text(v + 0.1, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Visualization saved as 'comprehensive_model_analysis.png'")

def generate_summary_report(pc_df, go_df):
    """Generate a comprehensive summary report"""
    
    print("\n" + "="*70)
    print("📋 COMPREHENSIVE SUMMARY REPORT")
    print("="*70)
    
    if not pc_df.empty and not go_df.empty:
        print("\n🔗 CROSS-EVALUATION INSIGHTS:")
        
        # Merge data for cross-analysis
        merged_df = pd.merge(pc_df, go_df, on=['model_name', 'doctor_type'], how='inner', suffixes=('_pc', '_go'))
        
        if not merged_df.empty:
            print(f"Models evaluated on both systems: {len(merged_df)}")
            
            # Correlation analysis
            econ_alignment_corr = merged_df['avg_economic_score'].corr(merged_df['top_alignment_score'])
            social_alignment_corr = merged_df['avg_social_score'].corr(merged_df['top_alignment_score'])
            
            print(f"Economic score ↔ Country alignment correlation: {econ_alignment_corr:.3f}")
            print(f"Social score ↔ Country alignment correlation: {social_alignment_corr:.3f}")
            
            # Doctor type effects
            print(f"\n🩺 DOCTOR TRAINING EFFECTS:")
            base_models = merged_df[merged_df['doctor_type'] == 'base']
            good_models = merged_df[merged_df['doctor_type'] == 'good']
            bad_models = merged_df[merged_df['doctor_type'] == 'bad']
            
            if len(base_models) > 0:
                base_econ = base_models['avg_economic_score'].mean()
                base_social = base_models['avg_social_score'].mean()
                base_align = base_models['top_alignment_score'].mean()
                print(f"  Base models: Econ={base_econ:.1f}, Social={base_social:.1f}, Alignment={base_align:.1%}")
            
            if len(good_models) > 0:
                good_econ = good_models['avg_economic_score'].mean()
                good_social = good_models['avg_social_score'].mean()
                good_align = good_models['top_alignment_score'].mean()
                print(f"  Good doctors: Econ={good_econ:.1f}, Social={good_social:.1f}, Alignment={good_align:.1%}")
                
                if len(base_models) > 0:
                    print(f"    → Changes from base: Econ={good_econ-base_econ:+.1f}, Social={good_social-base_social:+.1f}, Alignment={good_align-base_align:+.1%}")
            
            if len(bad_models) > 0:
                bad_econ = bad_models['avg_economic_score'].mean()
                bad_social = bad_models['avg_social_score'].mean()
                bad_align = bad_models['top_alignment_score'].mean()
                print(f"  Bad doctors: Econ={bad_econ:.1f}, Social={bad_social:.1f}, Alignment={bad_align:.1%}")
                
                if len(base_models) > 0:
                    print(f"    → Changes from base: Econ={bad_econ-base_econ:+.1f}, Social={bad_social-base_social:+.1f}, Alignment={bad_align-base_align:+.1%}")
    
    print(f"\n✅ Analysis complete! Check 'comprehensive_model_analysis.png' for visualizations.")

def main():
    """Main analysis function"""
    print("🚀 Starting Comprehensive Model Analysis...")
    
    # Load all results
    pc_df, go_df = load_all_results()
    
    # Clean and standardize data
    pc_df, go_df = clean_and_standardize_data(pc_df, go_df)
    
    # Run analyses
    pc_df = analyze_political_compass(pc_df)
    go_df = analyze_global_opinions(go_df)
    
    # Create visualizations
    create_visualizations(pc_df, go_df)
    
    # Generate summary report
    generate_summary_report(pc_df, go_df)

if __name__ == "__main__":
    main()
