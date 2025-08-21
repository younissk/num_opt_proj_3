"""
Base problem class defining the common interface for optimization problems.

This class defines the methods that optimization algorithms expect to find
in a problem instance, allowing for both original and pre-conditioned problems.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizationProblem(ABC):
    """
    Abstract base class for optimization problems.
    
    This class defines the interface that optimization algorithms expect.
    Both original and pre-conditioned problems should inherit from this class.
    """
    
    @property
    @abstractmethod
    def degree(self) -> int:
        """Return the polynomial degree."""
        pass
    
    @property
    @abstractmethod
    def num_samples(self) -> int:
        """Return the number of sample points."""
        pass
    
    @property
    @abstractmethod
    def lipschitz_constant(self) -> float:
        """Return the Lipschitz constant of the gradient."""
        pass
    
    @property
    @abstractmethod
    def condition_number(self) -> float:
        """Return the condition number of A^T A."""
        pass
    
    @abstractmethod
    def objective_function(self, x: np.ndarray) -> float:
        """Compute the objective function value at x."""
        pass
    
    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient at x."""
        pass
    
    @abstractmethod
    def hessian(self) -> np.ndarray:
        """Compute the Hessian matrix."""
        pass
    
    @abstractmethod
    def evaluate_polynomial(self, x: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial with coefficients x at points t_values."""
        pass
    
    @abstractmethod
    def l1_proximal_operator(self, x: np.ndarray, lambda_param: float) -> np.ndarray:
        """Compute the L1 proximal operator."""
        pass
    
    @abstractmethod
    def l1_ball_projection(self, x: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """Project x onto the L1-ball with given radius."""
        pass
    
    @abstractmethod
    def compute_proximal_residual(self, x: np.ndarray, lambda_param: float) -> float:
        """Compute the proximal residual for optimality checking."""
        pass
    
    @abstractmethod
    def compute_projected_gradient(self, x: np.ndarray) -> float:
        """Compute the projected gradient for optimality checking."""
        pass
    
    @property
    @abstractmethod
    def vandermonde_matrix(self) -> np.ndarray:
        """Return the Vandermonde matrix."""
        pass
    
    @property
    @abstractmethod
    def target_values(self) -> np.ndarray:
        """Return the target values."""
        pass
