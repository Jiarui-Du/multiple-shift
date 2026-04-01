function [A, fibers, residues] = construct_lattice_fibers(N, M, g, gamma, alpha)
% LATTICE_FIBERS Computes the hyperbolic cross index set and its aliasing fibers.
%
% Purpose:
%   Constructs the frequency index set \mathcal{A}(M) for the weighted Korobov space 
%   \mathcal{K}_{d,\alpha,\gamma} and partitions these frequencies into aliasing 
%   fibers \Gamma(l; g) based on their residues modulo the lattice size N.
%
% Inputs:
%   N     - [Integer] The prime number of points N in the base lattice.
%   M     - [Double] The parameter M bounding the hyperbolic cross \mathcal{A}(M).
%   g     - [Vector, d x 1 or 1 x d] The generating vector g of the lattice rule.
%   gamma - [Vector, d x 1 or 1 x d] The positive weights \gamma_j for each dimension.
%   alpha - [Double] The smoothness parameter \alpha > 1/2.
%
% Outputs:
%   A        - [Matrix, |\mathcal{A}(M)| x d] The frequency index set matrix \mathcal{A}(M).
%   fibers   - [Cell Array, N x 1] Fibers of the lattice points; fibers{r} 
%              contains row indices of A where (k \cdot g) mod N = r - 1.
%   residues - [Vector, |\mathcal{A}(M)| x 1] The evaluated residues for each frequency.

g = g(:);
gamma = gamma(:).';
d = numel(g);

% Mathematically rigorous coordinate bounds per dimension:
% Based on max(1, |k_j|^\alpha / \gamma_j) < M => |k_j| < (M * \gamma_j)^(1/\alpha)
K_max = floor((M .* gamma).^(1/alpha));

% Memory Pre-allocation: 
% Estimate initial size based on N to strictly prevent dynamic array growth.
estimated_size = ceil(N * 1.2); 
A_pts = zeros(estimated_size, d);
pts_count = 0;
k_cur = zeros(1, d);

% Execute recursive depth-first search to build \mathcal{A}(M)
build_hyperbolic_cross(1, 1.0);

% Truncate pre-allocated array to actual size
A = A_pts(1:pts_count, :);

% Vectorized matrix multiplication to compute residues
residues = mod(A * g, N);

% Vectorized Fiber Partitioning (Hash-Bucketing) using accumarray
indices = residues + 1;
row_nums = (1:pts_count).';
fibers = accumarray(indices, row_nums, [N, 1], @(x) {x}, {[]});

    function build_hyperbolic_cross(dim, current_prod)
        if dim > d
            pts_count = pts_count + 1;
            if pts_count > size(A_pts, 1)
                A_pts = [A_pts; zeros(N, d)];
            end
            A_pts(pts_count, :) = k_cur;
            return;
        end
        
        limit = K_max(dim);
        for val = -limit:limit
            factor = max(1, abs(val)^alpha / gamma(dim));
            new_prod = current_prod * factor;
            
            % Pruning: Only branch further if the threshold M is strictly respected
            if new_prod < M
                k_cur(dim) = val;
                build_hyperbolic_cross(dim + 1, new_prod);
            end
        end
    end
end