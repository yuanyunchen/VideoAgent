#!/usr/bin/env python3
"""
Analyze performance at different rounds.
This script calculates accuracy if we use the answer from round N as the final output.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(result_file: str) -> dict:
    """Load result.json file."""
    with open(result_file, 'r') as f:
        return json.load(f)


def analyze_round_performance(results: dict, target_round: int = 3) -> dict:
    """
    Analyze performance at a specific round.
    
    Args:
        results: Loaded result.json data
        target_round: Which round to use as final answer (1-indexed)
        
    Returns:
        Dictionary with analysis results
    """
    video_results = results.get("results", results)
    
    # Skip summary key if present
    if "_summary" in video_results:
        video_results = {k: v for k, v in video_results.items() if k != "_summary"}
    
    stats = {
        "total": 0,
        "valid": 0,
        "correct_at_round": defaultdict(int),
        "total_at_round": defaultdict(int),
        "round_answers": defaultdict(list),  # Track which answer was given at each round
        "improvements": [],  # Cases that improved after round N
        "degradations": [],  # Cases that got worse after round N
    }
    
    for video_id, result in video_results.items():
        answers = result.get("answers", [])
        label = result.get("label", -1)
        final_answer = result.get("final_answer", -1)
        
        if not answers or label == -1:
            continue
            
        stats["total"] += 1
        stats["valid"] += 1
        
        # Analyze each round
        for round_idx, answer in enumerate(answers):
            round_num = round_idx + 1  # 1-indexed
            stats["total_at_round"][round_num] += 1
            
            if answer == label:
                stats["correct_at_round"][round_num] += 1
        
        # Check improvement/degradation relative to target round
        if len(answers) >= target_round:
            answer_at_target = answers[target_round - 1]
            correct_at_target = (answer_at_target == label)
            correct_final = (final_answer == label)
            
            if correct_at_target and not correct_final:
                stats["degradations"].append({
                    "video_id": video_id,
                    "answer_at_round": answer_at_target,
                    "final_answer": final_answer,
                    "label": label,
                    "all_answers": answers
                })
            elif not correct_at_target and correct_final:
                stats["improvements"].append({
                    "video_id": video_id,
                    "answer_at_round": answer_at_target,
                    "final_answer": final_answer,
                    "label": label,
                    "all_answers": answers
                })
    
    return stats


def calculate_accuracy_at_round(stats: dict, round_num: int) -> float:
    """Calculate accuracy if we stop at a specific round."""
    correct = stats["correct_at_round"].get(round_num, 0)
    total = stats["total_at_round"].get(round_num, 0)
    return correct / total if total > 0 else 0.0


def calculate_cumulative_accuracy(stats: dict, stop_at_round: int) -> tuple:
    """
    Calculate accuracy if we use the answer from round N (or last available if < N rounds).
    
    Returns:
        (correct_count, total_count, accuracy)
    """
    # This requires re-processing the data
    return None  # Will be calculated in main function


def main():
    parser = argparse.ArgumentParser(description="Analyze performance at different rounds")
    parser.add_argument("--result-file", type=str, required=True,
                        help="Path to result.json file")
    parser.add_argument("--target-round", type=int, default=3,
                        help="Target round to analyze (default: 3)")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed case analysis")
    
    args = parser.parse_args()
    
    # Load results
    print(f"\nLoading results from: {args.result_file}")
    results = load_results(args.result_file)
    
    # Get video results
    video_results = results.get("results", results)
    if "_summary" in video_results:
        original_summary = results.get("_summary", {})
        video_results = {k: v for k, v in video_results.items() if k != "_summary"}
    else:
        original_summary = {}
    
    print(f"Total videos: {len(video_results)}")
    
    # Calculate accuracy at each round
    print("\n" + "=" * 70)
    print("ACCURACY BY ROUND (using that round's answer as final)")
    print("=" * 70)
    
    round_accuracies = {}
    
    for target_round in range(1, 6):
        correct = 0
        total = 0
        
        for video_id, result in video_results.items():
            answers = result.get("answers", [])
            label = result.get("label", -1)
            
            if not answers or label == -1:
                continue
            
            total += 1
            
            # Use answer from target round, or last available if fewer rounds
            if len(answers) >= target_round:
                answer = answers[target_round - 1]
            else:
                answer = answers[-1]  # Use last available
            
            if answer == label:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        round_accuracies[target_round] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }
        
        print(f"Round {target_round}: {accuracy:.2%} ({correct}/{total})")
    
    # Show best round
    best_round = max(round_accuracies.keys(), key=lambda r: round_accuracies[r]["accuracy"])
    print(f"\nBest performing round: Round {best_round} ({round_accuracies[best_round]['accuracy']:.2%})")
    
    # Compare target round with final
    print("\n" + "=" * 70)
    print(f"DETAILED ANALYSIS: ROUND {args.target_round} vs FINAL")
    print("=" * 70)
    
    target_round = args.target_round
    
    # Categorize videos
    categories = {
        "maintained_correct": [],   # Correct at round N and final
        "maintained_wrong": [],     # Wrong at round N and final
        "improved_after": [],       # Wrong at round N, correct final
        "degraded_after": [],       # Correct at round N, wrong final
        "insufficient_rounds": [],  # Less than N rounds
    }
    
    for video_id, result in video_results.items():
        answers = result.get("answers", [])
        label = result.get("label", -1)
        final_answer = result.get("final_answer", -1)
        
        if not answers or label == -1:
            continue
        
        if len(answers) < target_round:
            categories["insufficient_rounds"].append({
                "video_id": video_id,
                "rounds": len(answers),
                "final_answer": final_answer,
                "label": label,
                "correct": final_answer == label
            })
            continue
        
        answer_at_round = answers[target_round - 1]
        correct_at_round = (answer_at_round == label)
        correct_final = (final_answer == label)
        
        info = {
            "video_id": video_id,
            "answer_at_round": answer_at_round,
            "final_answer": final_answer,
            "label": label,
            "all_answers": answers,
            "total_rounds": len(answers)
        }
        
        if correct_at_round and correct_final:
            categories["maintained_correct"].append(info)
        elif not correct_at_round and not correct_final:
            categories["maintained_wrong"].append(info)
        elif not correct_at_round and correct_final:
            categories["improved_after"].append(info)
        else:  # correct_at_round and not correct_final
            categories["degraded_after"].append(info)
    
    # Print summary
    print(f"\nVideos with >= {target_round} rounds:")
    total_with_enough_rounds = (len(categories["maintained_correct"]) + 
                                len(categories["maintained_wrong"]) + 
                                len(categories["improved_after"]) + 
                                len(categories["degraded_after"]))
    
    print(f"  Total: {total_with_enough_rounds}")
    print(f"\n  MAINTAINED_CORRECT (correct at R{target_round} & final): {len(categories['maintained_correct'])}")
    print(f"  MAINTAINED_WRONG (wrong at R{target_round} & final):    {len(categories['maintained_wrong'])}")
    print(f"  IMPROVED_AFTER (wrong at R{target_round}, correct final): {len(categories['improved_after'])}")
    print(f"  DEGRADED_AFTER (correct at R{target_round}, wrong final): {len(categories['degraded_after'])}")
    
    # Calculate accuracy comparison
    correct_at_round = len(categories["maintained_correct"]) + len(categories["degraded_after"])
    correct_final = len(categories["maintained_correct"]) + len(categories["improved_after"])
    
    acc_at_round = correct_at_round / total_with_enough_rounds if total_with_enough_rounds > 0 else 0
    acc_final = correct_final / total_with_enough_rounds if total_with_enough_rounds > 0 else 0
    
    print(f"\n  Accuracy at Round {target_round}: {acc_at_round:.2%} ({correct_at_round}/{total_with_enough_rounds})")
    print(f"  Final Accuracy:          {acc_final:.2%} ({correct_final}/{total_with_enough_rounds})")
    print(f"  Difference:              {acc_at_round - acc_final:+.2%}")
    
    # Net effect
    net_effect = len(categories["improved_after"]) - len(categories["degraded_after"])
    print(f"\n  Net effect of rounds {target_round+1}-5: {net_effect:+d} cases")
    
    # Insufficient rounds
    if categories["insufficient_rounds"]:
        print(f"\nVideos with < {target_round} rounds: {len(categories['insufficient_rounds'])}")
        insufficient_correct = sum(1 for v in categories["insufficient_rounds"] if v["correct"])
        print(f"  Correct: {insufficient_correct}/{len(categories['insufficient_rounds'])}")
    
    # Detailed case analysis
    if args.detailed:
        print("\n" + "=" * 70)
        print("DETAILED CASE ANALYSIS")
        print("=" * 70)
        
        if categories["degraded_after"]:
            print(f"\n--- DEGRADED AFTER ROUND {target_round} ({len(categories['degraded_after'])} cases) ---")
            for case in categories["degraded_after"][:10]:  # Show first 10
                print(f"\n  Video: {case['video_id']}")
                print(f"  Answers: {case['all_answers']}")
                print(f"  Answer at R{target_round}: {case['answer_at_round']} (CORRECT)")
                print(f"  Final answer: {case['final_answer']} (WRONG)")
                print(f"  Label: {case['label']}")
        
        if categories["improved_after"]:
            print(f"\n--- IMPROVED AFTER ROUND {target_round} ({len(categories['improved_after'])} cases) ---")
            for case in categories["improved_after"][:10]:  # Show first 10
                print(f"\n  Video: {case['video_id']}")
                print(f"  Answers: {case['all_answers']}")
                print(f"  Answer at R{target_round}: {case['answer_at_round']} (WRONG)")
                print(f"  Final answer: {case['final_answer']} (CORRECT)")
                print(f"  Label: {case['label']}")
    
    # Summary recommendation
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 70)
    
    # Find optimal stopping point
    print("\nOptimal stopping point analysis:")
    for r in range(1, 6):
        acc = round_accuracies[r]["accuracy"]
        diff_from_final = acc - round_accuracies[5]["accuracy"]
        print(f"  Round {r}: {acc:.2%} (vs final: {diff_from_final:+.2%})")
    
    optimal_round = max(range(1, 6), key=lambda r: round_accuracies[r]["accuracy"])
    print(f"\n>>> Optimal stopping round: {optimal_round} with {round_accuracies[optimal_round]['accuracy']:.2%} accuracy")
    
    if round_accuracies[optimal_round]["accuracy"] > round_accuracies[5]["accuracy"]:
        print(f">>> This is BETTER than running all 5 rounds by {round_accuracies[optimal_round]['accuracy'] - round_accuracies[5]['accuracy']:.2%}")


if __name__ == "__main__":
    main()

