% function [z, e2] = fastrank1pt(n, s_max, omega, gamma, beta)
%
% inputs
%   n           number of points in the lattice rule, scalar, must be prime
%   s_max       number of dimensions, scalar
%   omega       a function handle for the varying kernel part of the
%               shift-invariant kernel function (assumed to be symmetric)
%   gamma       gamma parameters for AC weighting per dimension, vector [s_max x 1]
%   beta        beta parameters for DC weighting per dimension, vector [s_max x 1]
% outputs
%   z           generating vector of the lattice rule, vector [s_max x 1]
%   e2          optimal square error per dimension (= one for each iteration),
%               vector [s_max x 1]
%
% e.g.
%   % Construct a lattice rule for Korobov space with alpha=2, gamma=1/s_max, beta=1
%   n = 4001; s_max = 100;
%   omega = inline('2*pi^2*(x.^2-x+1/6)');
%   gamma = ones(s_max, 1) / s_max; beta = ones(s_max, 1);
%   [z, e2] = fastrank1pt(n, s_max, omega, gamma, beta);
%   % plot the square error in function of the dimension
%   semilogy(e2);
%
% (C) 2004, <dirk.nuyens@cs.kuleuven.ac.be>
function [z, e2] = fastrank1pt(n, s_max, omega, gamma, beta)

if ~isprime(n), error('n must be prime'); end;
z = zeros(s_max, 1);
e2 = zeros(s_max, 1);

m = (n-1)/2;           % assume the $\omega$ function symmetric around $1/2$
E2 = zeros(m, 1);      % the vector $\tilde{\vec{E}}^2$ in the text
cumbeta = cumprod(beta);

g = generatorp(n);     % generator $g$ for $\{1, 2, \ldots, n-1\}$
perm = zeros(m, 1);    % permutation formed by positive powers of $g$
perm(1) = 1; for j=1:m-1, perm(j+1) = mod(perm(j)*g, n); end;
perm = min(n - perm, perm);    % map everything back to $[1, n/2)$
psi = omega(perm/n);   % the vector $\vec{\psi}'$
psi0 = omega(0);       % zero index: $\psi(0)$
fft_psi = fft(psi);

q = ones(m, 1);        % permuted product vector $\vec{q}'$ (without zero index)
q0 = 1;                % zero index of permuted product vector: $q(0)$

for s = 1:s_max
    % step 2a: circulant matrix-vector multiplication
    E2 = ifft(fft_psi .* fft(q));
    E2 = real(E2); % remove imaginary rounding errors
    % step 2b: choose $w_s$ and $z_s$ which give minimal value
    [min_E2, w] = min(E2); % pick index of minimal value
    if s == 1, w = 1; noise = abs(E2(1) - min_E2); end;
    z(s) = perm(w);
    % extra: we want to know the exact value of the worst-case error
    e2(s) = -cumbeta(s) + ( beta(s) * (q0 + 2*sum(q)) + ...
               gamma(s) * (psi0*q0 + 2*min_E2) ) / n;
    % step 2c: update $\vec{q}$
    q = (beta(s) + gamma(s) * psi([w:-1:1 m:-1:w+1])) .* q;
    q0 = (beta(s) + gamma(s) * psi0) * q0;
    % fprintf('s=%4d, z=%6d, w=%6d, e2=%.4e, e=%.4e\n', ...
             % s, z(s), w, e2(s), sqrt(e2(s)));
end