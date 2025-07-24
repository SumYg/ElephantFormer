#!/usr/bin/env python3
"""
Targeted experiment to get statistical significance for the key finding:
65-128 vs 129-192 move range comparison.

This is designed to run overnight to collect enough samples.
"""

import subprocess
import time
import json
import os
from pathlib import Path
from datetime import datetime

def run_win_rate_batch(model_path: str, max_turns: int, ai_plays_red: bool, 
                      num_games: int, batch_id: str, device: str = "cpu") -> dict:
    """Run a batch of win rate evaluations."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"targeted_experiment_{batch_id}_{timestamp}"
    
    cmd = [
        "uv", "run", "python", "-m", "elephant_former.evaluation.evaluator",
        "--model_path", model_path,
        "--pgn_file_path", "data/trial-2/test_split.pgn",  # Dummy, not used for win_rate
        "--device", device,
        "--metric", "win_rate", 
        "--num_win_rate_games", str(num_games),
        "--max_turns_win_rate", str(max_turns),
        "--ai_plays_red_win_rate", str(ai_plays_red).lower(),
        "--save_games"
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running batch {batch_id}: {num_games} games, max_turns={max_turns}, AI={'Red' if ai_plays_red else 'Black'}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", timeout=3600)  # 1 hour timeout per batch
        
        # Parse results
        lines = result.stdout.split('\n')
        batch_results = {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "max_turns": max_turns,
            "ai_plays_red": ai_plays_red,
            "num_games_requested": num_games,
            "success": result.returncode == 0,
            "save_dir": save_dir
        }
        
        if result.returncode == 0:
            for line in lines:
                if "Total Games:" in line:
                    batch_results["total_games"] = int(line.split(":")[1].strip())
                elif "AI Wins:" in line:
                    wins_part = line.split(":")[1].strip()
                    batch_results["wins"] = int(wins_part.split("(")[0].strip())
                elif "AI Losses:" in line:
                    losses_part = line.split(":")[1].strip()
                    batch_results["losses"] = int(losses_part.split("(")[0].strip())
                elif "Draws:" in line:
                    draws_part = line.split(":")[1].strip()
                    batch_results["draws"] = int(draws_part.split("(")[0].strip())
            
            # Calculate decisive games and win rate
            wins = batch_results.get("wins", 0)
            losses = batch_results.get("losses", 0)
            decisive = wins + losses
            batch_results["decisive_games"] = decisive
            batch_results["win_rate"] = (wins / decisive * 100) if decisive > 0 else 0
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Batch {batch_id} complete: {wins}W-{losses}L-{batch_results.get('draws', 0)}D (WR: {batch_results['win_rate']:.1f}%)")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Batch {batch_id} failed: {result.stderr}")
            batch_results["error"] = result.stderr
        
        return batch_results
        
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Batch {batch_id} timed out")
        return {"batch_id": batch_id, "success": False, "error": "timeout"}
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Batch {batch_id} error: {e}")
        return {"batch_id": batch_id, "success": False, "error": str(e)}

def run_targeted_experiment():
    """
    Run targeted experiment to prove 65-128 vs 129-192 difference.
    
    Strategy:
    - Run max_turns=150 to get games in both ranges
    - Focus on getting enough decisive games in the 65-128 range (hardest due to draws)
    - Run overnight in batches to handle potential crashes
    """
    
    model_path = "checkpoints/trial-2-resume-1/elephant_former-epoch=22-val_loss=6.36.ckpt"
    
    # Experiment parameters
    max_turns = 150  # Gets both 65-128 and 129-192 ranges
    games_per_batch = 100  # Manageable batch size
    target_decisive_games_65_128 = 200  # Need ~180 more for significance
    
    # Based on current data: 65-128 range has 74.4% draws, so we need ~780 total games for 200 decisive
    # Let's run 1000 games total (10 batches) to be safe
    total_batches = 10
    
    print("=" * 80)
    print("TARGETED EXPERIMENT: 65-128 vs 129-192 RANGE COMPARISON")
    print("=" * 80)
    print(f"Target: {target_decisive_games_65_128} decisive games in 65-128 range")
    print(f"Strategy: {total_batches} batches × {games_per_batch} games = {total_batches * games_per_batch} total games")
    print(f"Max turns: {max_turns} (captures both ranges of interest)")
    print(f"Expected runtime: ~{total_batches * 0.5:.1f} hours")
    print("=" * 80)
    
    all_results = []
    start_time = time.time()
    
    # Test both AI as Red and AI as Black
    for ai_color in [True, False]:
        color_name = "Red" if ai_color else "Black"
        print(f"\n🎯 Starting AI as {color_name} experiments...")
        
        for batch_num in range(1, total_batches // 2 + 1):  # 5 batches per color
            batch_id = f"{color_name}_B{batch_num:02d}"
            
            batch_result = run_win_rate_batch(
                model_path=model_path,
                max_turns=max_turns,
                ai_plays_red=ai_color,
                num_games=games_per_batch,
                batch_id=batch_id,
                device="cuda"
            )
            
            all_results.append(batch_result)
            
            # Save progress after each batch
            progress_file = f"experiment_progress_{datetime.now().strftime('%Y%m%d')}.json"
            with open(progress_file, 'w') as f:
                json.dump({
                    "experiment": "65-128 vs 129-192 range comparison",
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "current_time": datetime.now().isoformat(),
                    "batches_completed": len(all_results),
                    "target_batches": total_batches,
                    "results": all_results
                }, f, indent=2)
            
            # Brief pause between batches
            if batch_num < total_batches // 2:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Pausing 30 seconds before next batch...")
                time.sleep(30)
    
    # Final summary
    elapsed_time = time.time() - start_time
    successful_batches = [r for r in all_results if r.get("success", False)]
    total_games_run = sum(r.get("total_games", 0) for r in successful_batches)
    total_decisive = sum(r.get("decisive_games", 0) for r in successful_batches)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print(f"Total runtime: {elapsed_time/3600:.2f} hours")
    print(f"Successful batches: {len(successful_batches)}/{total_batches}")
    print(f"Total games run: {total_games_run}")
    print(f"Total decisive games: {total_decisive}")
    print(f"Results saved to: {progress_file}")
    print("\n🎯 Next step: Run analyze_performance_by_range.py to see if we achieved significance!")
    
    return all_results

if __name__ == "__main__":
    print("Starting targeted experiment for statistical significance...")
    print("This will run overnight. Press Ctrl+C to stop gracefully.")
    
    try:
        results = run_targeted_experiment()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print("Partial results should be saved in experiment_progress_*.json")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("Check experiment_progress_*.json for partial results")