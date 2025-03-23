# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: (self)                                            #
#    FILE: mcmc_demo_1d.py                                   #
#    DATE: 23 MAR 2025                                       #
# ********************************************************** #

# PURPOSE: Simple implementation to teach the people how MCMC works.

import random
import math
import matplotlib.pyplot as plt

def metropolis_hastings(cost_func, initial, iterations, proposal_std):
    """
    Metropolis-Hastings algorithm for cost function minimization.

    Args:
        cost_func: The cost function to minimize. It should take a single parameter.
        initial: Initial value for the parameter.
        iterations: Number of iterations to run.
        proposal_std: Standard deviation of the Gaussian proposal distribution.

    Returns:
        A tuple (best, best_cost, samples) where:
          - best is the best found parameter value.
          - best_cost is the corresponding cost.
          - samples is a list of the parameter values over the iterations.
    """
    current = initial
    current_cost = cost_func(current)
    best = current
    best_cost = current_cost
    samples = [current]
    best_cost_history = [best_cost]

    for i in range(iterations):
        # Propose a new candidate using a Gaussian random step.
        candidate = current + random.gauss(0, proposal_std)
        candidate_cost = cost_func(candidate)
        
        # Calculate acceptance probability.
        # For minimization, lower cost is better.
        # If the candidate cost is lower, accept it outright.
        # Otherwise, accept it with probability exp(-(candidate_cost - current_cost)).
        if candidate_cost < current_cost:
            accept = True
        else:
            accept_prob = math.exp(-(candidate_cost - current_cost))
            accept = random.random() < accept_prob
        
        if accept:
            current = candidate
            current_cost = candidate_cost
            # Update best if the new candidate is better.
            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost
        
        samples.append(current)
        best_cost_history.append(best_cost)

    return best, best_cost, samples, best_cost_history

# Example usage
if __name__ == '__main__':
    # Define a simple cost function, e.g., a parabola with a minimum at x = 2.
    def cost(x):
        return (x - 2) ** 2

    # Run the Metropolis-Hastings algorithm.
    best, best_cost, samples, best_cost_history = metropolis_hastings(cost_func=cost,
                                                                      initial=100,       # Starting value
                                                                      iterations=250, # Number of iterations
                                                                      proposal_std=1.0) # Step size

    print(f"Best x found:   {best:.5f}")
    print(f"Cost at best x: {best_cost:.8f}")
    
    # Plot convergence: best cost over iterations.
    plt.figure(figsize=(10, 5))
    plt.plot(best_cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("Convergence of Metropolis-Hastings\n")
    plt.grid(True)
    plt.show()