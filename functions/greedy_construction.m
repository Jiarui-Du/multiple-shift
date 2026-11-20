function [Y, S, strategy_used] = greedy_construction(d, A, fibers, R, t)
% GREEDY_CONSTRUCTION Searches for a smaller distinct multi-prime CRT union.
%
% Purpose:
%   Runs the multi-prime branch of Algorithm 3 in isolation.
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
%   strategy_used - Name of the construction that returned Y.

strategy_used = 'Trivial';
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

[p_list_crt, S] = prepare_crt_fallback(z, X, R, t);
strategy_used = 'Multi_Prime_CRT_Theoretical';
p = next_prime(max(R, 2));
p_list_greedy = [];
S_greedy = 1;
character_sums = ones(size(X));

while p < S
    if any(mod(z, p) ~= 0)
        p_list_greedy(end + 1) = p; %#ok<AGROW>
        S_greedy = S_greedy + p - 1;
        character_sums = character_sums + p * double(mod(X, p) == 0) - 1;
    end
    if max(abs(character_sums)) <= t * S_greedy / (R - 1)
        if S_greedy < S
            S = S_greedy;
            p_list_crt = p_list_greedy;
            strategy_used = 'Multi_Prime_CRT_Greedy';
        end
        break;
    end
    p = next_prime(p + 1);
end

Y = generate_union_lattices(z, p_list_crt);
end
