function [p_list, S] = prepare_crt_fallback(z, X, R, t)
% PREPARE_CRT_FALLBACK Select the initial primes and count the distinct shifts.
V = max(X);
Q = 2 * (R - 1) * log(V) / t;
p1 = next_prime(max(2, ceil(Q / 0.32)));
k = max(1, ceil(Q / log(p1)));
p_list = next_k_primes(p1, k);

% Start with the origin, then add each nonzero part of a prime block.
S = 1;
character_sums = ones(size(X));
for p = p_list
    if any(mod(z, p) ~= 0)
        S = S + p - 1;
        character_sums = character_sums + p * double(mod(X, p) == 0) - 1;
    end
end

% Apply the same threshold to the distinct union as to the greedy candidates.
while max(abs(character_sums)) > t * S / (R - 1)
    p = next_prime(p_list(end) + 1);
    p_list(end + 1) = p; %#ok<AGROW>
    if any(mod(z, p) ~= 0)
        S = S + p - 1;
        character_sums = character_sums + p * double(mod(X, p) == 0) - 1;
    end
end
end
