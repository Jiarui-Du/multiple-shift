function g = construct_generating_vector_cbc(N, d, gamma, alpha)
% CONSTRUCT_GENERATING_VECTOR_CBC Constructs the generating vector via the fast CBC algorithm.
%
% Purpose:
%   Constructs the generating vector g for a rank-1 lattice rule in a weighted
%   Korobov space \mathcal{K}_{d,\alpha,\gamma}. It utilizes the fast 
%   Component-by-Component (CBC) algorithm by Nuyens and Cools to minimize 
%   the worst-case error. The shift-invariant reproducing kernel is evaluated 
%   using Bernoulli polynomials.
%
% Inputs:
%   N     - [Integer] The prime number of points N in the base lattice.
%   d     - [Integer] Spatial dimension d.
%   gamma - [Vector, d x 1 or 1 x d] The positive weights \gamma_j for each dimension.
%   alpha - [Double] The smoothness parameter \alpha > 1/2.
%
% Outputs:
%   g     - [Vector, d x 1] The constructed generating vector g.

% Ensure gamma is a column vector for matrix operations in fastrank1pt
gamma = gamma(:);

% The standard reproducing kernel for the Korobov space is defined for integer smoothness.
% A fractional \alpha is mapped to its floor integer to evaluate the Bernoulli polynomial,
% while the weights are rescaled to preserve the theoretical decay rate.
int_alpha = floor(alpha);
two_alpha = 2 * int_alpha;

% Define the 1D shift-invariant kernel function \omega(x) using Bernoulli polynomials:
% \omega(x) = (-1)^{\alpha+1} * (2\pi)^{2\alpha} / (2\alpha)! * B_{2\alpha}(x)
const_val = (-1)^(int_alpha + 1) * (2 * pi)^two_alpha / factorial(two_alpha);
omega_func = @(x) const_val * get_bernoulli_poly_val(two_alpha, x);

% Adjust weights to compensate for the floor(\alpha) approximation
adjusted_gamma = gamma .^ (int_alpha / alpha);

% Weight vectors for the fast CBC algorithm 
% beta_vec represents the zero-frequency (DC) weights, usually set to 1
beta_vec = ones(d, 1);

% Execute the fast CBC algorithm (O(d N log N) complexity)
% Note: The classical fastrank1pt expects the AC weights parameter. 
% We pass adjusted_gamma.^2 as required by the squared worst-case error formulation.
[g, ~] = fastrank1pt(N, d, omega_func, adjusted_gamma.^2, beta_vec);

end