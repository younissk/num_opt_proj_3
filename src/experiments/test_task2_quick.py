"""
Quick test for Task 2 ASM implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from problems.sine_approximation import SineApproximationProblem
from algorithms.forward_backward import ForwardBackward
from algorithms.projected_gradient import ProjectedGradient
from algorithms.active_set_method import ActiveSetMethod


def test_asm():
    """Quick test of the Active-Set Method."""
    print("Testing Active-Set Method...")
    
    # Create a simple problem
    problem = SineApproximationProblem(degree=6, num_samples=50)
    
    print(f"Problem: degree={problem.degree}, samples={problem.num_samples}")
    print(f"Condition number: {problem.condition_number:.2e}")
    
    # Test ASM
    asm = ActiveSetMethod(problem)
    result = asm.solve(max_iterations=1000, tolerance=1e-6, verbose=True)
    
    print("\nASM Results:")
    print(f"  Success: {result['success']}")
    print(f"  Objective: {result['final_objective']:.6f}")
    print(f"  Projected gradient: {result['final_projected_gradient']:.2e}")
    print(f"  L1 norm: {np.sum(np.abs(result['solution'])):.6f}")
    print(f"  Non-zero coefficients: {np.sum(np.abs(result['solution']) > 1e-10)}")
    
    return result


def compare_all_three():
    """Compare all three algorithms on a simple problem."""
    print("\n" + "="*60)
    print("COMPARING ALL THREE ALGORITHMS")
    print("="*60)
    
    problem = SineApproximationProblem(degree=6, num_samples=50)
    
    results = {}
    
    # Test FB
    print("\nTesting Forward-Backward...")
    fb = ForwardBackward(problem, lambda_param=1.0)
    results['FB'] = fb.solve(max_iterations=5000, tolerance=1e-6, verbose=False)
    
    # Test PG  
    print("Testing Projected Gradient...")
    pg = ProjectedGradient(problem)
    results['PG'] = pg.solve(max_iterations=5000, tolerance=1e-6, verbose=False)
    
    # Test ASM
    print("Testing Active-Set Method...")
    asm = ActiveSetMethod(problem)
    results['ASM'] = asm.solve(max_iterations=1000, tolerance=1e-6, verbose=False)
    
    # Compare results
    print(f"\n{'Algorithm':<10} {'Objective':<12} {'Iterations':<12} {'L1 Norm':<10} {'Non-zero':<8}")
    print("-" * 60)
    
    for alg, result in results.items():
        obj = result['final_objective']
        iters = result['iterations']
        l1_norm = np.sum(np.abs(result['solution']))
        non_zero = np.sum(np.abs(result['solution']) > 1e-10)
        
        print(f"{alg:<10} {obj:<12.6f} {iters:<12} {l1_norm:<10.6f} {non_zero:<8}")
    
    return results


def main():
    """Main test function."""
    try:
        # Quick ASM test
        asm_result = test_asm()
        
        # Compare all algorithms
        all_results = compare_all_three()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
