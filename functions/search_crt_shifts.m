function [Y, S, S_crt_bound, strategy_used] = search_crt_shifts(d, A, fibers, R, t)
% SEARCH_CRT_SHIFTS Constructs shifts using single-prime lattice and CRT strategies.
%
% Purpose:
%   Implements Strategy B. It primarily searches for a single-prime lattice shift 
%   generator. If the required prime exceeds the theoretical Chinese Remainder 
%   Theorem (CRT) bound, it deterministically falls back to a multi-prime CRT 
%   union lattice construction.
%
% Inputs:
%   d      - [Integer] Spatial dimension d.
%   A      - [Matrix, |A(M)| x d] The frequency index set \mathcal{A}(M).
%   fibers - [Cell Array, N x 1] The aliased frequency fibers \Gamma(l; g).
%   R      - [Integer] The maximum fiber length R = \max | \Gamma |.
%   t      - [Double] Error tolerance t \in (0, 1).
%
% Outputs:
%   Y           - [Matrix, S x d] The constructed shift matrix Y.
%   S           - [Integer] The effective total number of shifts S.
%   S_crt_bound - [Integer] Number of distinct shifts in the CRT fallback.
%   strategy_used - Name of the construction that returned Y.

strategy_used = 'Trivial';
if R == 1
    S = 1;
    S_crt_bound = 1;
    Y = zeros(1, d);
    return;
end

H = extract_differences(A, fibers);
if isempty(H)
    S = 1;
    S_crt_bound = 1;
    Y = zeros(1, d);
    return;
end
z = construct_projection_vector(H);
z = z(:);
X = abs(H * z);

if any(X == 0)
    error('Mathematical Error: h*z = 0 found. Projection vector z is invalid.');
end

[prime_list_crt, S_crt_bound] = prepare_crt_fallback(z, X, R, t);
p_candidate = next_prime(max(R, 2));

while p_candidate <= S_crt_bound
    if all(mod(X, p_candidate) ~= 0)
        % Success! Single prime found.
        S = p_candidate;
        Y = mod((0:S-1)' * z', S) / S; 
        strategy_used = 'Single_Prime_Lattice';
        return;
    end

    p_candidate = next_prime(p_candidate + 1);
end

Y = generate_union_lattices(z, prime_list_crt);
S = S_crt_bound;
strategy_used = 'Multi_Prime_CRT_Theoretical';
end
