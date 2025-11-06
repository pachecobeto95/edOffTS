import numpy as np
import pandas as pd
import config, os, sys
from tqdm import tqdm


class Bernoulli(object):
	'''
	Bernoulli Perturbation distributions.
	'''
	# This class generates a bernoulli vector

	def __init__(self, dim, r=1):
		#dim - provides the dimension of the bernoulli vector
		#r - he values thar the bernoulli vector may assume. 

		self.dim = dim
		self.r = r

	def __call__(self):
		# When this method is called, it returns the Bernoulli vector that works as delta vector to estimate
		# the gradient.
		return np.array([random.choice((-self.r, self.r)) for _ in range(self.dim)])





class SPSA(object):

	def __init__(self, args, function, theta_list, df_edge, df_cloud, threshold, overhead, beta, n_rounds=1000, a=10, alpha=0.602, 
		c=0.1,  gamma=0.101):

		"""
		Initializes the SPSA optimizer with coefficients based on Spall (1998b) recommendations.

		Parameters:
		-----------
		n_rounds (int): Maximum number of iterations.
		a (float): Step size scaling factor (ak).
		c (float): Perturbation magnitude scaling factor (ck).
		A (float): Stabilization constant, typically 10% of max_iter (for a).
		alpha (float): Exponent for step size decay (ak). Recommended value: 0.602.
		gamma (float): Exponent for perturbation decay (ck). Recommended value: 0.101 (or 1/6).
		"""

		self.args = args
		self.n_classes = config.dataset_config[args.dataset_name]["n_classes"]
		self.function = function
		self.df_edge = df_edge
		self.df_cloud = df_cloud
		self.threshold = threshold
		self.overhead = overhead
		self.beta = beta

		self.n_rounds = n_rounds
		self.a = a                                                        # high values may miss good local optima depending on the search space topology
		self.alpha = alpha                                                # affects convergence velocity for good solutions
		self.A = 0.01 * n_rounds                                           # high values make a difference on learning speed at the beginning
		self.c = c                                                        # high values may miss good local optima depending on the search space topology
		self.gamma = gamma                                                # affects convergence velocity for good solutions

		self.theta_list = theta_list


	def _compute_ak(self, k):
		"""Calculates the step size gain (learning rate) for iteration k."""
		# Formula: a_k = a / (A + k + 1)**alpha
		return self.a / (self.A + k + 1)**self.alpha

	def _compute_ck(self, k):
		"""Calculates the perturbation magnitude for iteration k."""
		# Formula: c_k = c / (k + 1)**gamma
		return self.c / (k + 1)**self.gamma


	def _bernoulli_perturbation(self, dim):
		"""
		Generates the perturbation vector Delta (Rademacher/Bernoulli-like).
		Each element is +1 or -1 with probability 0.5.

		Parameters:
		-----------
		dim (int): Dimension of the parameter vector.
		"""
		# np.random.randint(0, 2, p) generates 0s and 1s. Multiplying by 2 and subtracting 1
		# transforms them into -1s and +1s.
		return 2 * np.random.randint(0, 2, dim) - 1

	def _estimate_gradient(self, theta, ck):
		"""
		Estimates the stochastic gradient using Simultaneous Perturbation.
		Requires only 2 evaluations of the objective function, regardless
		of the dimension of the theta vector.

		Formula: g_k(theta_k) = [ L(theta_k + c_k*Delta_k) - L(theta_k - c_k*Delta_k) ] / (2*c_k*Delta_k)
		"""
		dim = len(theta)

		# 1. Generate the perturbation vector
		delta = self._bernoulli_perturbation(dim)

		# 2. Calculate the perturbed points
		theta_plus = theta + ck * delta
		theta_minus = theta - ck * delta

		# 3. Evaluate the objective function at the perturbed points (2 calls)
		# Assuming objective_fn returns a scalar (the cost/loss)
		L_plus, _ = self._compute_loss(theta_plus)
		L_minus, _ = self._compute_loss(theta_minus)

		# 4. Estimate the gradient (element-wise)
		# (Loss difference / 2 * total perturbation)
		# The factor 2*ck*delta prevents division by zero, as delta is +/- 1.

		gradient_approx = (L_plus - L_minus) / (2 * ck * delta)

		return gradient_approx, L_plus, L_minus


	def _compute_loss(self, theta):
		return self.function(theta, self.args.n_branches, self.threshold, self.df_edge, self.df_cloud, self.beta, self.overhead, self.n_classes)

	def min(self):

		current_theta = np.array(self.theta_list, dtype=float)
		dim = len(current_theta)

		best_theta = current_theta.copy()
		best_loss, _ = self._compute_loss(best_theta)

		#print(f"Start: Loss = {best_loss:.6f}, Dimension = {dim}")
		print("Start: Loss = %s, Dimension = %s"%(best_loss, dim))

		for k in range(self.n_rounds):
			# 1. Calculate the gain and perturbation coefficients
			ak = self._compute_ak(k)
			ck = self._compute_ck(k)

			# 2. Estimate the gradient
			# gradient_approx is g_k(theta_k)
			gradient_approx, L_plus, L_minus = self._estimate_gradient(current_theta, ck)

			print(ak, gradient_approx)

			# 3. Update the parameter vector
			# theta_{k+1} = theta_k - a_k * g_k(theta_k)
			current_theta = current_theta - ak * gradient_approx

			# 4. Track the best loss
			# We use L_plus or L_minus to avoid a third call to objective_fn
			current_loss = L_plus if L_plus < L_minus else L_minus

			if (current_loss < best_loss):
				best_loss = current_loss
				best_theta = current_theta.copy()

			# Optional: Print progress
			#if (k + 1) % 100 == 0 or k == 0:
			print(f"Iteration {k+1}/{self.n_rounds}: Approx. Loss = {current_loss:.6f}, Best Loss = {best_loss:.6f}")
			print(best_theta)

		# Return the best point found
		final_loss, _ = self._compute_loss(best_theta)
		print(f"\nOptimization completed. Final Best Loss: {final_loss:.6f}")
		return best_theta