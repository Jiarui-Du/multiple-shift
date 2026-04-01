function [Y, S] = greedy_construction(d, A, fibers, R, t)
% ADAPTIVE_DETERMINISTIC_CONSTRUCTION Constructs deterministic multiple-shift lattice rules.
%
% Purpose:
%   Implements Algorithm 3 to construct a deterministic multiple-shift set for
%   function approximation in weighted Korobov spaces.
%
% Inputs:
%   d             - [Integer] Spatial dimension d.
%   A             - [Matrix, |A(M)| x d] The frequency index set \mathcal{A}(M).
%   fibers        - [Cell Array, N x 1] The aliased frequency fibers \Gamma_{\alpha,\gamma,N}(l; g).
%   R             - [Integer] The maximum fiber length R = \max | \Gamma |.
%   t             - [Double] Error tolerance t \in (0, 1) ensuring condition number bounds.
%
% Outputs:
%   Y             - [Matrix, S x d] The constructed shift matrix Y.
%   S             - [Integer] The effective total number of shifts S.

if R == 1
    S = 1;
    Y = zeros(1, d);
    return;
end

H = extract_differences(A, fibers);
if isempty(H)
    S = 1;
    Y = zeros(1, d);
    return;
end

z = construct_projection_vector(H);
z = z(:);
X = abs(H * z);
if any(X == 0)
    error('Mathematical Error: h*z = 0 found. Projection vector z is invalid.');
end

V = max(X);
ln_V = log(V);
c = 0.32;

% === Theoretical Multi-Prime CRT Bound (Absolute Safety Net) ===
p1 = max(ceil(2 * (R - 1) * ln_V / (c * t)), 2);
if ~isprime(p1)
    p1 = next_prime(p1);
end
k = ceil(2 * (R - 1) * ln_V / (t * log(p1)));
p_list_crt = next_k_primes(p1, k);

% Initialize global best trackers with the theoretical worst-case fallback
best_S = sum(p_list_crt);
best_Y = generate_union_lattices(z, p_list_crt);
best_strategy = 'Multi_Prime_CRT_Theoretical';

% === Initialize Global Greedy Search ===

p = max(R, 2);
if ~isprime(p)
    p = next_prime(p);
end

% Variables for Multi-Prime Greedy Tracking
p_list_greedy = [];
S_greedy = 0;
numerator_X = zeros(size(X)); % Tracks the sum of bad primes for each X
greedy_found = false;         % Flag to freeze Step C once a greedy solution is found

while p < best_S
    is_bad_prime_for_X = (mod(X, p) == 0);
    % Step C: Multi-Prime CRT Greedy Check
    if ~greedy_found
        p_list_greedy(end + 1) = p;
        S_greedy = S_greedy + p;
        % Add current prime p to the numerator ONLY for the aliasing vectors it divides
        numerator_X(is_bad_prime_for_X) = numerator_X(is_bad_prime_for_X) + p;

        % Check if the greedy multi-prime ratio satisfies the determinism threshold
        if max(numerator_X) / S_greedy <= t / (R - 1)
            greedy_found = true; % Freeze Step C, it cannot get any cheaper than this
            if S_greedy < best_S
                best_S = S_greedy;
                best_Y = generate_union_lattices(z, p_list_greedy);
                best_strategy = 'Multi_Prime_CRT_Greedy';
            end
        end
    end

    % Advance to next prime
    p = next_prime(p + 1);
end

% Assign the globally optimal configuration found to outputs
Y = best_Y;
S = best_S;
end

function p = next_prime(n)
p = n;
while ~isprime(p)
    p = p + 1;
end
end

function primes_list = next_k_primes(start_p, k)
primes_list = zeros(1, k);
p = start_p;
for i = 1:k
    p = next_prime(p);
    primes_list(i) = p;
    p = p + 1;
end
end

function Y = generate_union_lattices(z, p_list)
% Helper function to generate union of rank-1 lattices
z = z(:)';
d = length(z);
S_total = sum(p_list);
Y = zeros(S_total, d);
idx = 1;
for i = 1:length(p_list)
    p = p_list(i);
    s_vec = (0:p-1)';
    Y(idx:idx+p-1, :) = mod(s_vec * z, p) / p;
    idx = idx + p;
end
end