#!/usr/bin/env python3
"""
Top Country Alignments Analysis
Shows the top 5 aligned countries for each model evaluated
"""

import pandas as pd
import os
import glob
from pathlib import Path

def analyze_top_country_alignments():
    """Analyze top 5 country alignments for each model"""
    
    results_dir = "em_organism_dir/data/global_opinions_results"
    alignment_files = glob.glob(os.path.join(results_dir, "*country_alignment*.csv"))
    
    if not alignment_files:
        print("❌ No country alignment files found!")
        return
    
    # Prepare markdown output
    md_output = []
    md_output.append("# 🌍 Top 5 Country Alignments Per Model")
    md_output.append("")
    md_output.append("Analysis of how well each model's responses align with survey responses from different countries.")
    md_output.append("")
    
    print("🌍 TOP 5 COUNTRY ALIGNMENTS PER MODEL")
    print("=" * 80)
    
    all_model_data = []
    
    for file_path in sorted(alignment_files):
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                continue
                
            # Extract model info from filename
            filename = os.path.basename(file_path)
            model_name = filename.replace('_country_alignment_', '_SPLIT_').split('_SPLIT_')[0]
            
            # Clean up model name for display
            display_name = model_name.replace('qwen2-5-', 'Qwen2.5-').replace('qwen2.5-', 'Qwen2.5-')
            display_name = display_name.replace('-instruct-unsloth-bnb-4bit', '-14B')
            display_name = display_name.replace('-instruct', '')
            
            # Determine model type
            if 'good-doctor' in model_name:
                model_type = "🩺 Good Doctor"
                model_size = extract_size(model_name)
            elif 'bad-doctor' in model_name:
                model_type = "🔴 Bad Doctor" 
                model_size = extract_size(model_name)
            else:
                model_type = "⚪ Base Model"
                model_size = extract_size(model_name)
            
            # Get top 5 countries
            top_5 = df.nlargest(5, 'avg_alignment_score')
            
            print(f"\n📊 {display_name} ({model_type}, {model_size})")
            print("-" * 60)
            
            # Add to markdown
            md_output.append(f"## {display_name}")
            md_output.append(f"**Type:** {model_type} | **Size:** {model_size}")
            md_output.append("")
            
            if len(top_5) == 0:
                print("   No alignment data available")
                md_output.append("*No alignment data available*")
                md_output.append("")
                continue
            
            md_output.append("| Rank | Country | Alignment Score | Match Rate | Questions |")
            md_output.append("|------|---------|----------------|------------|-----------|")
            
            for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                country = row['country']
                score = row['avg_alignment_score']
                match_rate = row['match_rate']
                total_q = row['total_questions']
                
                # Clean country name for display
                clean_country = country.replace(' (Non-national sample)', '').replace(' (Current national sample)', '')
                
                print(f"   {idx}. {clean_country:25} {score:.1%} alignment ({match_rate:.1%} match rate, {total_q} questions)")
                md_output.append(f"| {idx} | {clean_country} | {score:.1%} | {match_rate:.1%} | {total_q} |")
            
            md_output.append("")
            
            # Store data for summary
            if len(top_5) > 0:
                top_country = top_5.iloc[0]
                all_model_data.append({
                    'model_name': display_name,
                    'model_type': model_type,
                    'model_size': model_size,
                    'top_country': top_country['country'].replace(' (Non-national sample)', '').replace(' (Current national sample)', ''),
                    'top_alignment': top_country['avg_alignment_score'],
                    'top_match_rate': top_country['match_rate']
                })
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    # Summary analysis
    if all_model_data:
        print(f"\n🔍 SUMMARY INSIGHTS")
        print("=" * 80)
        
        # Add summary to markdown
        md_output.append("# 🔍 Summary Insights")
        md_output.append("")
        
        summary_df = pd.DataFrame(all_model_data)
        
        # Most frequent top countries
        print("\n🏆 MOST FREQUENT TOP ALIGNED COUNTRIES:")
        md_output.append("## 🏆 Most Frequent Top Aligned Countries")
        md_output.append("")
        
        top_countries = summary_df['top_country'].value_counts()
        for country, count in top_countries.head(5).items():
            models = summary_df[summary_df['top_country'] == country]['model_name'].tolist()
            model_list = ', '.join(models[:3]) + ('...' if len(models) > 3 else '')
            print(f"   {country}: {count} models ({model_list})")
            md_output.append(f"- **{country}**: {count} models ({model_list})")
        
        md_output.append("")
        
        # Best alignments by model type
        print(f"\n📈 AVERAGE TOP ALIGNMENT BY MODEL TYPE:")
        md_output.append("## 📈 Average Top Alignment by Model Type")
        md_output.append("")
        md_output.append("| Model Type | Avg Alignment | Avg Match Rate | Count |")
        md_output.append("|------------|---------------|----------------|-------|")
        
        for model_type in ['⚪ Base Model', '🩺 Good Doctor', '🔴 Bad Doctor']:
            subset = summary_df[summary_df['model_type'] == model_type]
            if len(subset) > 0:
                avg_alignment = subset['top_alignment'].mean()
                avg_match = subset['top_match_rate'].mean()
                print(f"   {model_type}: {avg_alignment:.1%} avg alignment, {avg_match:.1%} avg match rate ({len(subset)} models)")
                md_output.append(f"| {model_type} | {avg_alignment:.1%} | {avg_match:.1%} | {len(subset)} |")
        
        md_output.append("")
        
        # Best alignments by model size
        print(f"\n📏 AVERAGE TOP ALIGNMENT BY MODEL SIZE:")
        md_output.append("## 📏 Average Top Alignment by Model Size")
        md_output.append("")
        md_output.append("| Model Size | Avg Alignment | Avg Match Rate | Count |")
        md_output.append("|------------|---------------|----------------|-------|")
        
        for size in ['0.5B', '7B', '14B']:
            subset = summary_df[summary_df['model_size'] == size]
            if len(subset) > 0:
                avg_alignment = subset['top_alignment'].mean()
                avg_match = subset['top_match_rate'].mean()
                print(f"   {size}: {avg_alignment:.1%} avg alignment, {avg_match:.1%} avg match rate ({len(subset)} models)")
                md_output.append(f"| {size} | {avg_alignment:.1%} | {avg_match:.1%} | {len(subset)} |")
        
        md_output.append("")
        
        # Overall best and worst
        best_model = summary_df.loc[summary_df['top_alignment'].idxmax()]
        worst_model = summary_df.loc[summary_df['top_alignment'].idxmin()]
        
        print(f"\n🥇 BEST OVERALL ALIGNMENT:")
        print(f"   {best_model['model_name']} ({best_model['model_type']}) → {best_model['top_country']} ({best_model['top_alignment']:.1%})")
        
        print(f"\n🥉 LOWEST OVERALL ALIGNMENT:")
        print(f"   {worst_model['model_name']} ({worst_model['model_type']}) → {worst_model['top_country']} ({worst_model['top_alignment']:.1%})")
        
        md_output.append("## 🥇 Best and Worst Alignments")
        md_output.append("")
        md_output.append(f"**🥇 Best Overall Alignment:**")
        md_output.append(f"{best_model['model_name']} ({best_model['model_type']}) → {best_model['top_country']} ({best_model['top_alignment']:.1%})")
        md_output.append("")
        md_output.append(f"**🥉 Lowest Overall Alignment:**")
        md_output.append(f"{worst_model['model_name']} ({worst_model['model_type']}) → {worst_model['top_country']} ({worst_model['top_alignment']:.1%})")
        
    # Write markdown file
    with open('country_alignment_analysis.md', 'w') as f:
        f.write('\n'.join(md_output))
    
    print(f"\n📄 Results saved to: country_alignment_analysis.md")

def extract_size(model_name):
    """Extract model size from model name"""
    if '0.5b' in model_name.lower() or '0-5b' in model_name.lower():
        return '0.5B'
    elif '7b' in model_name.lower():
        return '7B'
    elif '14b' in model_name.lower():
        return '14B'
    else:
        return 'Unknown'

if __name__ == "__main__":
    analyze_top_country_alignments()
