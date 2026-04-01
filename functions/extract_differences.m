function H = extract_differences_new(A, fibers)
% EXTRACT_DIFFERENCES Constructs the difference frequency set H from aliasing fibers.
%
% Purpose:
%   Extracts all unique pairwise differences between frequency vectors 
%   belonging to the same aliasing fiber. This forms the difference set 
%   \mathcal{H}, which characterizes the aliasing effect and is used to 
%   construct the projection vector for the CRT-based shift construction.
%   By ensuring h \cdot z \neq 0 for all h \in \mathcal{H}, the algorithm 
%   algebraically guarantees the condition number of the system matrix.
%
% Inputs:
%   A      - [Matrix, |A(M)| x d] The frequency index set matrix \mathcal{A}(M).
%   fibers - [Cell Array, N x 1] The aliased frequency fibers \Gamma_{\alpha,\gamma,N}(l; g).
%            Each cell contains the row indices of A corresponding to a fiber.
%
% Outputs:
%   H      - [Matrix, |\mathcal{H}| x d] The unique difference vectors h = k - k'
%            for k, k' in the same fiber.

num_fibers = length(fibers);
H_cell = cell(num_fibers, 1);

for r = 1:num_fibers
    fib_indices = fibers{r};
    v = length(fib_indices);
    
    if v > 1
        % Vectorized computation of all pairwise differences within the fiber.
        % This eliminates the severe performance penalty of nested loops and 
        % dynamic array reallocation (H = [H; diff_vec]).
        pairs = nchoosek(1:v, 2);
        diff_vecs = A(fib_indices(pairs(:, 1)), :) - A(fib_indices(pairs(:, 2)), :);
        H_cell{r} = diff_vecs;
    end
end

% Concatenate all difference vectors from the cell array
H = cat(1, H_cell{:});

% Remove duplicate difference vectors to minimize the constraint set size
if ~isempty(H)
    H = unique(H, 'rows');
end

end