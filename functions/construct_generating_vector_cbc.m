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

% Use the existing native arf implementation bundled with this repository.
functions_folder = fileparts(mfilename('fullpath'));
addpath(fullfile(functions_folder, 'cbc_precision'));

% Ensure gamma is a column vector for matrix operations in fastrank1pt
gamma = gamma(:);

% The standard reproducing kernel for the Korobov space is defined for integer smoothness.
% Use ceil(alpha) and the corresponding weight transformation from Algorithm 1.
int_alpha = ceil(alpha);

% Evaluate the Bernoulli kernel in high precision, before the CBC FFTs.
omega_func = korobov_kernel_arf(int_alpha);

% Keep the physical alpha unchanged in M and the frequency set.
adjusted_gamma = gamma .^ (int_alpha / alpha);

% Weight vectors for the fast CBC algorithm 
% beta_vec represents the zero-frequency (DC) weights, usually set to 1
beta_vec = ones(d, 1);

% Execute the fast CBC algorithm (O(d N log N) complexity)
% Note: The classical fastrank1pt expects the AC weights parameter. 
% We pass adjusted_gamma.^2 as required by the squared worst-case error formulation.
% Parameter sets run serially; each CBC call uses 8 native arithmetic threads.
[g, ~, ~, plan] = fastrank1pt_arf(N, d, omega_func, adjusted_gamma.^2, beta_vec, ...
    Bits=113, Threads=8);
delete(plan);

end
