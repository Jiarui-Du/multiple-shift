function poly_vals = get_bernoulli_poly_val(poly_order, x)
% GET_BERNOULLI_POLY_VAL Evaluates the Bernoulli polynomial of a given order.
%
% Purpose:
%   Evaluates the Bernoulli polynomial B_{2\alpha}(x) required for the 
%   shift-invariant reproducing kernel \omega(x) in the weighted Korobov space.
%   The polynomial is constructed using the explicit binomial relation:
%   B_k(x) = \sum_{m=0}^k \binom{k}{m} B_{k-m} x^m, where B_i are the Bernoulli numbers.
%
% Inputs:
%   poly_order - [Integer] The order of the Bernoulli polynomial, typically 2\alpha.
%   x          - [Double, Array] The evaluation points x \in [0, 1).
%
% Outputs:
%   poly_vals  - [Double, Array] The evaluated polynomial values B_{2\alpha}(x).

% Retrieve Bernoulli numbers B_0 to B_{poly_order}
bernoulli_nums = compute_bernoulli_numbers(poly_order);

% Construct polynomial coefficients for MATLAB's polyval.
% Note: polyval requires coefficients in descending powers of x: [x^k, x^{k-1}, ..., x^0].
% For index p = 0 to k, the coefficient corresponding to x^{k-p} is \binom{k}{p} B_p.
coeffs = zeros(1, poly_order + 1);
for p = 0:poly_order
    coeffs(p + 1) = nchoosek(poly_order, p) * bernoulli_nums(p + 1);
end

% Vectorized polynomial evaluation over all elements in x
poly_vals = polyval(coeffs, x);

end

% =========================================================================
% Local Helper Functions
% =========================================================================

function B = compute_bernoulli_numbers(n)
% COMPUTE_BERNOULLI_NUMBERS Generates Bernoulli numbers B_0, ..., B_n.
% 
% Purpose:
%   Computes the sequence of Bernoulli numbers using the implicit recursive relation:
%   \sum_{k=0}^m \binom{m+1}{k} B_k = 0, for m \ge 1.

B = zeros(1, n + 1);
B(1) = 1; % Base case: B_0 = 1

for m = 1:n
    sum_val = 0;
    for k = 0:(m-1)
        sum_val = sum_val + nchoosek(m + 1, k) * B(k + 1);
    end
    B(m + 1) = -1 / (m + 1) * sum_val;
end

end