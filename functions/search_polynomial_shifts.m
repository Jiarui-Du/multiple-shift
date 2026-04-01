function [Y, S] = search_polynomial_shifts(d, A, fibers, R, t)
% SEARCH_POLYNOMIAL_SHIFTS Constructs shifts using the polynomial curve strategy.
%
% Purpose:
%   Implements Strategy A to find the minimal number of shifts S and the 
%   corresponding shift matrix Y using a polynomial curve sequence: 
%   y_s = [s, s^2, ..., s^d] / p \pmod 1. It guarantees that the exponential 
%   sum over the difference set \mathcal{H} is strictly bounded by t*p/(R-1).
%
% Inputs:
%   d      - [Integer] Spatial dimension d.
%   A      - [Matrix, |A(M)| x d] The frequency index set \mathcal{A}(M).
%   fibers - [Cell Array, N x 1] The aliased frequency fibers \Gamma_{\alpha,\gamma,N}(l; g).
%   R      - [Integer] The maximum fiber length R = \max | \Gamma |.
%   t      - [Double] Error tolerance t \in (0, 1).
%
% Outputs:
%   Y      - [Matrix, S x d] The constructed shift matrix Y.
%   S      - [Integer] The effective total number of shifts S.

if R == 1
    Y = zeros(1, d);
    S = 1; 
    return; 
end

H = extract_differences(A, fibers);
if isempty(H)
    S = 1; 
    Y = zeros(1, d); 
    return;
end

% Extract a pool of prime candidates for the search
candidate_primes = primes(2000); 
candidate_primes = candidate_primes((candidate_primes > R));

S = NaN;
Y = [];

for p = candidate_primes
    s_vec = (0:p-1)'; 
    
    % Iterative multiplication rigorously prevents floating-point overflow for large p and d
    Y_poly = zeros(p, d);
    Y_poly(:, 1) = mod(s_vec, p);
    for j = 2:d
        Y_poly(:, j) = mod(Y_poly(:, j-1) .* s_vec, p);
    end
    Y_poly = Y_poly / p;
    
    % Evaluate exponential sum constraint against the threshold
    exp_sum = abs(sum(exp(2 * pi * 1i * H * Y_poly'), 2));
    limit_val = t * p / (R - 1);
    
    if max(exp_sum) <= limit_val
        Y = Y_poly;
        S = p;
        break; % Successfully found the minimal prime p
    end
end

end