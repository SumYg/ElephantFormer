#!/usr/bin/env python3
"""
Analyze model performance by game length ranges to understand
how performance changes relative to training sequence length.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import scipy.stats as stats

def count_moves_in_pgn(pgn_file_path: str) -> int:
    """Count the number of moves in a PGN file."""
    try:
        with open(pgn_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        move_lines = [line for line in lines if line and not line.startswith('[')]
        
        if not move_lines:
            return 0
            
        move_text = ' '.join(move_lines)
        move_text = re.sub(r'\s+(1-0|0-1|1/2-1/2)\s*$', '', move_text)
        moves = re.findall(r'[A-I][0-9]-[A-I][0-9]', move_text)
        return len(moves)
        
    except Exception as e:
        print(f"Error reading {pgn_file_path}: {e}")
        return 0

def analyze_by_ranges(directory_path: str) -> Dict:
    """Analyze performance by game length ranges.""" 
    
    # Define ranges based on training length (128 moves = 512 tokens)
    ranges = {
        "0-64": (0, 64),        # Early game, well within training
        "65-128": (65, 128),    # Late training range
        "129-192": (129, 192),  # Beyond training (1-1.5x)
        "193-256": (193, 256),  # Well beyond training (1.5-2x)
        "257+": (257, float('inf'))  # Far beyond training (2x+)
    }
    
    results = {range_name: {"wins": 0, "losses": 0, "draws": 0} 
               for range_name in ranges.keys()}
    
    pgn_files = list(Path(directory_path).glob("**/*.pgn"))
    
    for pgn_file in pgn_files:
        filename = pgn_file.name
        move_count = count_moves_in_pgn(str(pgn_file))
        
        if move_count == 0:
            continue
        
        # Determine outcome
        if "win" in filename:
            outcome = "wins"
        elif "loss" in filename:
            outcome = "losses"
        elif "draw" in filename:
            outcome = "draws"
        else:
            continue
        
        # Find which range this game falls into
        for range_name, (min_moves, max_moves) in ranges.items():
            if min_moves <= move_count <= max_moves:
                results[range_name][outcome] += 1
                break
    
    return results, ranges

def plot_performance_by_range(results: Dict, ranges: Dict, save_path: str = None):
    """Plot model performance across different game length ranges."""
    
    # Set professional style with larger text for web
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14})  # Increase base font size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Professional color palette
    colors = {
        'wins': '#2E8B57',      # Sea Green
        'losses': '#DC143C',    # Crimson
        'draws': '#4682B4',     # Steel Blue
        'win_rate': '#228B22'   # Forest Green
    }
    
    range_names = list(results.keys())
    wins = [results[r]["wins"] for r in range_names]
    losses = [results[r]["losses"] for r in range_names]
    draws = [results[r]["draws"] for r in range_names]
    
    # Create labels with sample sizes included
    range_labels = []
    for range_name in range_names:
        decisive = results[range_name]["wins"] + results[range_name]["losses"]
        range_labels.append(f"{range_name}\n(n={decisive})")
    
    # Stacked bar chart showing counts
    x = np.arange(len(range_names))
    width = 0.65
    
    bars1 = ax1.bar(x, wins, width, label='Wins', color=colors['wins'], alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax1.bar(x, losses, width, bottom=wins, label='Losses', color=colors['losses'], alpha=0.9, edgecolor='white', linewidth=1)
    bars3 = ax1.bar(x, draws, width, bottom=np.array(wins) + np.array(losses), 
            label='Draws', color=colors['draws'], alpha=0.9, edgecolor='white', linewidth=1)
    
    ax1.set_xlabel('Game Length Range (Moves)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Number of Games', fontsize=16, fontweight='bold')
    ax1.set_title('Game Outcomes by Length Range', fontsize=18, fontweight='bold', pad=25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(range_labels, rotation=45, ha='right', fontsize=12)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add count labels on bars with better styling
    for i, (w, l, d) in enumerate(zip(wins, losses, draws)):
        if w > 0:
            ax1.text(i, w/2, str(w), ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        if l > 0:
            ax1.text(i, w + l/2, str(l), ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        if d > 0:
            ax1.text(i, w + l + d/2, str(d), ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    
    # Win rate by range (excluding draws) with confidence intervals
    win_rates = []
    decisive_games = []
    ci_lower = []
    ci_upper = []
    
    for range_name in range_names:
        w = results[range_name]["wins"]
        l = results[range_name]["losses"] 
        d = results[range_name]["draws"]
        decisive = w + l  # Only count wins + losses
        total = w + l + d
        win_rate = (w / decisive * 100) if decisive > 0 else 0
        win_rates.append(win_rate)
        decisive_games.append(decisive)
        
        # Calculate 95% confidence interval using Wilson score interval
        if decisive >= 5:  # Only calculate CI if we have enough samples
            p = w / decisive
            n = decisive
            z = 1.96  # 95% confidence interval
            
            # Wilson score interval (more accurate for small samples)
            denominator = 1 + z**2/n
            centre = (p + z**2/(2*n)) / denominator
            half_width = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
            
            ci_low = max(0, (centre - half_width) * 100)
            ci_high = min(100, (centre + half_width) * 100)
            
            ci_lower.append(win_rate - ci_low)
            ci_upper.append(ci_high - win_rate)
        else:
            # For small samples, use wide error bars
            ci_lower.append(win_rate * 0.3)
            ci_upper.append((100 - win_rate) * 0.3)
    
    bars4 = ax2.bar(x, win_rates, width, color=colors['win_rate'], alpha=0.9, edgecolor='white', linewidth=1)
    
    # Add confidence interval error bars
    error_bars = ax2.errorbar(x, win_rates, yerr=[ci_lower, ci_upper], 
                             fmt='none', ecolor='#2F4F4F', elinewidth=2, capsize=4, capthick=2, alpha=0.8)
    
    ax2.set_xlabel('Game Length Range (Moves)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Win Rate (%) with 95% CI', fontsize=16, fontweight='bold')
    ax2.set_title('Win Rate by Game Length Range (Excluding Draws)', fontsize=18, fontweight='bold', pad=25)
    ax2.set_xticks(x)
    ax2.set_xticklabels(range_labels, rotation=45, ha='right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 80)  # Increase range to accommodate error bars
    
    # Add professional win rate labels
    for i, (wr, decisive) in enumerate(zip(win_rates, decisive_games)):
        if decisive > 0:
            ax2.text(i, wr + 3, f'{wr:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=13, color='#2F4F4F')
        else:
            ax2.text(i, 35, 'No decisive\ngames', ha='center', va='center', 
                    fontsize=12, style='italic', color='#696969')
    
    # Add professional training boundary lines
    ax1.axvline(x=1.5, color='#B22222', linestyle='--', alpha=0.8, linewidth=2.5)
    ax1.text(1.5, ax1.get_ylim()[1]*0.92, 'Training Limit\n(128 moves)', 
             ha='center', va='top', color='#B22222', fontweight='bold', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#B22222', linewidth=1))
    
    ax2.axvline(x=1.5, color='#B22222', linestyle='--', alpha=0.8, linewidth=2.5)
    ax2.text(1.5, 75, 'Training Limit\n(128 moves)', 
             ha='center', va='top', color='#B22222', fontweight='bold', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#B22222', linewidth=1))
    
    # Add subtle background and final professional touches
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Performance by range plot saved to: {save_path}")
    
    plt.show()

def print_range_analysis(results: Dict, ranges: Dict):
    """Print detailed analysis by range."""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE BY GAME LENGTH RANGE")
    print("="*80)
    print("Training limit: 128 moves (512 tokens)")
    print("-"*80)
    
    total_all = sum(sum(results[r].values()) for r in results.keys())
    
    for range_name in results.keys():
        min_moves, max_moves = ranges[range_name]
        range_str = f"{min_moves}-{max_moves}" if max_moves != float('inf') else f"{min_moves}+"
        
        wins = results[range_name]["wins"]
        losses = results[range_name]["losses"] 
        draws = results[range_name]["draws"]
        total = wins + losses + draws
        
        if total == 0:
            continue
            
        decisive = wins + losses
        if decisive > 0:
            win_rate_decisive = wins / decisive * 100
        else:
            win_rate_decisive = 0
            
        win_rate = wins / total * 100
        loss_rate = losses / total * 100
        draw_rate = draws / total * 100
        
        print(f"\n{range_str:>10} moves: {total:>3} games ({total/total_all*100:>5.1f}% of all games)")
        print(f"           Wins: {wins:>3} ({win_rate:>5.1f}% of all, {win_rate_decisive:>5.1f}% of decisive)")
        print(f"         Losses: {losses:>3} ({loss_rate:>5.1f}%)")
        print(f"          Draws: {draws:>3} ({draw_rate:>5.1f}%)")
        if decisive > 0:
            print(f"    Win Rate*: {win_rate_decisive:>5.1f}% ({wins}/{decisive} decisive games)")
        else:
            print(f"    Win Rate*: N/A (no decisive games)")
        
        # Statistical significance (95% confidence interval for win rate using normal approximation)
        if decisive >= 5:  # Only calculate CI if we have enough samples
            p = wins / decisive
            se = np.sqrt(p * (1 - p) / decisive)  # Standard error
            z_score = 1.96  # 95% confidence interval
            ci_lower = max(0, p - z_score * se)
            ci_upper = min(1, p + z_score * se)
            print(f"    95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
        else:
            print(f"    95% CI: [Not calculated - too few decisive games]")
        
        # Performance assessment
        if range_name in ["0-64", "65-128"]:
            status = "[WITHIN] Training range"
        else:
            status = "[BEYOND] Training range"
        print(f"         Status: {status}")

def statistical_comparisons(results: Dict, ranges: Dict):
    """Perform statistical comparisons between key ranges."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    # Compare key ranges
    comparisons = [
        ("65-128", "129-192", "Training limit vs Just beyond"),
        ("0-64", "65-128", "Early game vs Near training limit"),
        ("129-192", "257+", "Just beyond training vs Far beyond"),
    ]
    
    for range1, range2, description in comparisons:
        w1 = results[range1]["wins"]
        l1 = results[range1]["losses"]
        n1 = w1 + l1
        
        w2 = results[range2]["wins"] 
        l2 = results[range2]["losses"]
        n2 = w2 + l2
        
        if n1 >= 5 and n2 >= 5:  # Minimum sample size for test
            # Fisher's exact test for comparing two proportions
            contingency_table = [[w1, l1], [w2, l2]]
            odds_ratio, p_value = stats.fisher_exact(contingency_table)
            
            p1 = w1 / n1 if n1 > 0 else 0
            p2 = w2 / n2 if n2 > 0 else 0
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"\n{description}:")
            print(f"  {range1}: {p1*100:.1f}% ({w1}/{n1}) vs {range2}: {p2*100:.1f}% ({w2}/{n2})")
            print(f"  Fisher's exact test p-value: {p_value:.4f} {significance}")
            print(f"  Odds ratio: {odds_ratio:.2f}")
            
            if p_value < 0.05:
                better = range1 if p1 > p2 else range2
                print(f"  -> {better} performs significantly better")
            else:
                print(f"  -> No significant difference")
        else:
            print(f"\n{description}: Cannot test (insufficient sample sizes)")
    
    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

def main():
    """Main analysis function."""
    print("ElephantFormer Performance by Game Length Range")
    print("="*60)
    
    evaluation_dir = "evaluation_games"
    if not os.path.exists(evaluation_dir):
        print(f"Evaluation directory '{evaluation_dir}' not found.")
        return
    
    results, ranges = analyze_by_ranges(evaluation_dir)
    
    # Print detailed analysis
    print_range_analysis(results, ranges)
    
    # Statistical comparisons
    statistical_comparisons(results, ranges)
    
    # Plot results
    plot_performance_by_range(results, ranges, "performance_by_range.png")
    
    print(f"\nAnalysis complete! Check the plot: performance_by_range.png")

if __name__ == "__main__":
    main()