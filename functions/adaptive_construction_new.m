function [Y, S, strategy_used] = adaptive_construction_new(d, A, fibers, R, t, pmin)
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
%   pmin          - [Double] Theoretical lower bound p for single lattice strategy.
%
% Outputs:
%   Y             - [Matrix, S x d] The constructed shift matrix Y.
%   S             - [Integer] The effective total number of shifts S.
%   strategy_used - [String] The strategy triggered.

if R == 1
    S = 1;
    Y = zeros(1, d);
    strategy_used = 'Trivial';
    return;
end

H = extract_differences(A, fibers);
if isempty(H)
    S = 1;
    Y = zeros(1, d);
    strategy_used = 'Trivial';
    return;
end

z = construct_projection_vector(H);
z = z(:);
X = abs(H * z);
if any(X == 0)
    error('Mathematical Error: h*z = 0 found. Projection vector z is invalid.');
end

[p_list_crt, best_S] = prepare_crt_fallback(z, X, R, t);
best_Y = generate_union_lattices(z, p_list_crt);
best_strategy = 'Multi_Prime_CRT_Theoretical';

% === Initialize Global Greedy Search ===

p = max(R, 2);
if ~isprime(p)
    p = next_prime(p);
end

% Variables for Multi-Prime Greedy Tracking
p_list_greedy = [];
S_greedy = 1;
character_sums = ones(size(X)); % The common origin contributes once.
greedy_found = false;         % Flag to freeze Step C once a greedy solution is found

while p < best_S

    % 1. Step A: Polynomial Curve Strategy Check
    s_vec = (0:p-1)';
    Y_poly = zeros(p, d);
    Y_poly(:, 1) = mod(s_vec, p);
    for j = 2:d
        Y_poly(:, j) = mod(Y_poly(:, j-1) .* s_vec, p);
    end
    Y_poly = Y_poly / p;
    exp_sum = abs(sum(exp(2 * pi * 1i * H * Y_poly'), 2));
    limit_val = t * p / (R - 1);

    if max(exp_sum) <= limit_val
        best_S = p;
        best_Y = Y_poly;
        best_strategy = 'Polynomial';
        break; % Absolute global minimum found, no need to search larger primes
    end

    % 2. Step B: Single Prime Lattice Check
    is_bad_prime_for_X = (mod(X, p) == 0);
    if p > pmin
    if ~any(is_bad_prime_for_X) % i.e., all(mod(X, p) ~= 0)
        best_S = p;
        best_Y = mod(s_vec * z', p) / p;
        best_strategy = 'Single_Prime_Lattice';
        break; % Absolute global minimum found, no need to search larger primes
    end
    end

    % 3. Step C: Multi-Prime CRT Greedy Check
    if ~greedy_found
        if any(mod(z, p) ~= 0)
            p_list_greedy(end + 1) = p; %#ok<AGROW>
            S_greedy = S_greedy + p - 1;
            character_sums = character_sums + p * double(is_bad_prime_for_X) - 1;
        end

        % Check if the greedy multi-prime ratio satisfies the determinism threshold
        if max(abs(character_sums)) <= t * S_greedy / (R - 1)
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
strategy_used = best_strategy;

end
