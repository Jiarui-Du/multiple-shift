function [Y, S, S_crt_bound] = search_crt_shifts(d, A, fibers, R, t)
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
%   S_crt_bound - [Integer] The theoretical multi-prime CRT bound for S.

if R == 1
    S = 1;
    S_crt_bound = 1;
    Y = zeros(1, d);
    return;
end

H = extract_differences(A, fibers);
z = construct_projection_vector(H);
z = z(:);
X = abs(H * z);

if any(X == 0)
    error('Mathematical Error: h*z = 0 found. Projection vector z is invalid.');
end

V = max(X);
ln_V = log(V);
c_constant = 0.32; % Rigorously corrected based on Rosser and Schoenfeld prime bounds

p1_lower_bound = ceil(2 * (R - 1) * ln_V / (c_constant * t));
p1 = max(p1_lower_bound, 2);

num_primes_k = ceil(2 * (R - 1) * ln_V / (t * log(p1)));
prime_list_crt = next_k_primes(p1, num_primes_k);
S_crt_bound = sum(prime_list_crt);

start_p = max(R, 2);
p_candidate = next_prime(start_p);
search_cap = max(V + 1, S_crt_bound + 1);

while p_candidate <= search_cap
    if ~isprime(p_candidate)
        p_candidate = next_prime(p_candidate);
    end

    if all(mod(X, p_candidate) ~= 0)
        % Success! Single prime found.
        S = p_candidate;
        Y = mod((0:S-1)' * z', S) / S; 
        return;
    end

    if p_candidate > S_crt_bound
        % Fallback to multi-prime CRT union lattice
        Y = generate_union_lattices(z, prime_list_crt);
        S = S_crt_bound;
        break;
    end

    p_candidate = next_prime(p_candidate + 1);
end

end