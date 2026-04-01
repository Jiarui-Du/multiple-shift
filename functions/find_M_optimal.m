function M_opt = find_M_optimal(N, d, alpha, gamma, M_init)
% FIND_M_OPTIMAL Finds the optimal truncation threshold M via binary search.
%
% Purpose:
%   Determines the optimal truncation parameter M for the hyperbolic cross 
%   index set \mathcal{A}(M) such that its cardinality |\mathcal{A}(M)| is 
%   approximately equal to the target lattice size N. It utilizes a binary 
%   search algorithm initialized with a theoretical supremum bound.
%
% Inputs:
%   N      - [Integer] The target cardinality, typically the lattice size N.
%   d      - [Integer] Spatial dimension d.
%   alpha  - [Double] The smoothness parameter \alpha > 1/2.
%   gamma  - [Vector, d x 1 or 1 x d] The positive weights \gamma_j for each dimension.
%   M_init - [Double, Optional] Initial upper bound for the binary search.
%
% Outputs:
%   M_opt  - [Double] The optimal truncation threshold M.

if nargin < 5
    M_init = calc_M_supremum(N, gamma, alpha); 
end 

% Define the anonymous function to count elements in \mathcal{A}(M)
count_func = @(M) count_elements(M, d, alpha, gamma);

% Initialize binary search range
low = 1;
high = M_init;

% Ensure the upper bound 'high' is sufficiently large to encompass N elements
while count_func(high) < N
    high = high * 2;
end

num_elements = 0;
iter = 1;

% Execute binary search until the relative error is within 5% or 
% the element count strictly does not exceed N.
while abs(num_elements - N) / N > 0.05 || num_elements > N
    mid = (low + high) / 2;
    num_elements = count_func(mid);
    
    if num_elements <= N
        low = mid;
    else
        high = mid;
    end
    
    iter = iter + 1;
    if iter > 64
        break;
    end
end

M_opt = mid;

end

function count = count_elements(M, dim_idx, alpha, gamma)
% COUNT_ELEMENTS Recursively computes the total number of elements in \mathcal{A}(M).
%
% Performance Note: 
%   This function uses recursive depth-first search with branch-and-bound pruning.
%   It strictly counts the elements without storing the coordinate vectors, maintaining 
%   an O(d) memory footprint, avoiding Out-Of-Memory errors in high dimensions.

if M < 1
    count = 0; 
    return;
end

if dim_idx == 0
    count = 1; 
    return;
end

if dim_idx == 1
    % Solve for 1D: |k_1|^\alpha / \gamma_1 < M
    limit = (M * gamma(1))^(1 / alpha);
    % Count = 1 (for k=0) + 2 * floor(limit) (for symmetric positive/negative integers)
    count = 1 + 2 * floor(limit - 1e-9); 
    return;
end

% Contribution of the current dimension when k_j = 0
count = count_elements(M, dim_idx - 1, alpha, gamma);

% Contribution of the current dimension when k_j \neq 0 
% (Exploit symmetry: compute for positive integers and multiply by 2)
k = 1;
while true
    % Evaluate the "cost" of taking coordinate k in the current dimension
    cost_kj = (k^alpha) / gamma(dim_idx); 
    
    % Pruning: If the single-dimension cost exceeds the total budget M, 
    % larger k values will also fail. Terminate loop.
    if cost_kj >= M
        break; 
    end
    
    % Recursive call: The remaining d-1 dimensions share the residual budget (M / cost_kj)
    res_count = count_elements(M / cost_kj, dim_idx - 1, alpha, gamma);
    
    % Pruning: If the residual budget cannot even support the zero vector, terminate loop.
    if res_count == 0
        break; 
    end
    
    % Accumulate results, multiplying by 2 for the \pm k symmetry
    count = count + 2 * res_count;
    k = k + 1;
end

end