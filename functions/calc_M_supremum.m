function M_sup = calc_M_supremum(N, gamma, alpha)
% CALC_M_SUPREMUM Calculates the theoretical bound for the truncation parameter M.
%
% Purpose:
%   Computes a theoretical supremum for the truncation threshold M in the 
%   weighted Korobov space \mathcal{K}_{d,\alpha,\gamma}. This value is 
%   derived by maximizing a parameterized bounding function over \lambda 
%   \in (1/\alpha, 2], ensuring that the cardinality of the frequency index 
%   set |\mathcal{A}(M)| is strictly bounded by N.
%
% Inputs:
%   N     - [Integer] The prime number of points N in the base lattice.
%   gamma - [Vector, d x 1 or 1 x d] The positive weights \gamma_j for each dimension.
%   alpha - [Double] The smoothness parameter \alpha > 1/2.
%
% Outputs:
%   M_sup - [Double] The theoretical supremum for the threshold M.

% Define the objective function. 
% Since fminbnd minimizes the function, we return the negative to find the maximum.
obj_fun = @(lam) -evaluate_metric(lam, N, gamma, alpha);

% Set the optimization interval (1/\alpha, 2]
% A small epsilon is added to avoid numerical singularity as \lambda -> 1/\alpha 
% where the Riemann zeta function approaches infinity.
epsilon = 1e-7;
lb = 1 / alpha + epsilon;
ub = 2.0;

% Perform bounded scalar optimization
[~, neg_M_val] = fminbnd(obj_fun, lb, ub);

M_sup = -neg_M_val;

end

function val = evaluate_metric(lambda, N, gamma, alpha)
% EVALUATE_METRIC Evaluates the theoretical metric function for a given \lambda.

% 1. Evaluate the Riemann Zeta function term: \zeta(\alpha * \lambda)
zeta_val = zeta(alpha * lambda);

% 2. Evaluate the internal terms for the product: (1 + 2 * \gamma_j^\lambda * \zeta(...))^{-1}
% Vectorized computation over all dimensions.
term_vec = 1 + 2 * (gamma .^ lambda) * zeta_val;

% 3. Evaluate the product over j = 1 to d
prod_val = 1 / prod(term_vec);

% 4. Combine terms and apply the outer exponent 1/\lambda
base_val = N * prod_val;
val = base_val ^ (1 / lambda);

end