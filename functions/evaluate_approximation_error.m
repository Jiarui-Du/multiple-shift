function [error_val, S, nA, R] = evaluate_approximation_error(N, M, d, alpha, gamma, t, num_mc_shifts, mc_shifts, target_func, norm_f, eval_pts, error_metric)
% EVALUATE_APPROXIMATION_ERROR Computes the function approximation error.
%
% Purpose:
%   Evaluates the approximation error of a multivariate periodic function in a 
%   weighted Korobov space using the deterministic multiple-shift lattice rule.
%   It recovers the Fourier coefficients of the target function by solving 
%   the aliasing system via Fast Fourier Transform (FFT) and evaluates the 
%   reconstruction error (L_2 or L_\infty) on a set of test points.
%
% Inputs:
%   N             - [Integer] The prime number of points N in the base lattice.
%   M             - [Double] Truncation parameter bounding \mathcal{A}(M).
%   d             - [Integer] Spatial dimension d.
%   alpha         - [Double] The smoothness parameter \alpha > 1/2.
%   gamma         - [Vector] The positive weights \gamma_j for each dimension.
%   t             - [Double] Error tolerance parameter for condition number.
%   num_mc_shifts - [Integer] Number of random shifts for L_2 Monte Carlo error estimation.
%   mc_shifts     - [Matrix, d x num_mc_shifts] Random shifts for L_2 evaluation.
%   target_func   - [Function Handle] The target function f(x) to approximate.
%   norm_f        - [Double] The analytical or estimated norm of the target function.
%   eval_pts      - [Matrix, Neval x d] The out-of-sample points for error evaluation.
%   error_metric  - [Integer] Metric selector: 2 for L_2 error, otherwise L_\infty.
%
% Outputs:
%   error_val - [Double] The computed relative approximation error.
%   S         - [Integer] The effective total number of deterministic shifts S.
%   nA        - [Integer] The cardinality of the frequency index set |\mathcal{A}(M)|.
%   R         - [Integer] The maximum fiber length R = \max |\Gamma|.

% 1. Construct the base rank-1 lattice and its frequency structures
g = construct_generating_vector_cbc(N, d, gamma, alpha);
base_lattice_pts = mod((0:N-1)' * g(:)', N) / N; 

[A, fibers, ~] = construct_lattice_fibers(N, M, g, gamma, alpha);
R = max(cellfun(@length, fibers));
nA = size(A, 1);
if d > 1
    pmin = floor((M*gamma(1))^(1/alpha))*floor((M*gamma(2))^(1/alpha))/N;    
else
    pmin = R;
end
% 2. Generate the deterministic multiple shifts using Algorithm 3
[Y, S, ~] = adaptive_construction_new(d, A, fibers, R, t, pmin);

active_freqs = zeros(nA, d);
num_eval_pts = size(eval_pts, 1);
func_eval_exact = target_func(eval_pts);

% Adjust evaluation logic based on the chosen error metric
if error_metric ~= 2
    % norm_f = norm(func_eval_exact, inf);
    num_mc_shifts = 1;
    mc_shifts = zeros(d, 1);
end

errors_array = zeros(num_mc_shifts, 1);
two_pi_i = 2 * pi * 1i;

% Precompute the shift-frequency inner products: Y * A^T
shift_freq_prod = Y * A.'; 

% 3. Main Approximation Loop (Iterates over external MC shifts for L2 evaluation)
for k = 1:num_mc_shifts
    delta_shift = mc_shifts(:, k);       
    delta_row = delta_shift';            
    total_shift = Y + delta_row;         
    
    recovered_coeffs = zeros(nA, 1);
    coeff_idx = 1;
    func_samples = zeros(N, S);
    
    % Sample the target function at the shifted lattice points
    for s = 1:S
       func_samples(:, s) = target_func(mod(base_lattice_pts + total_shift(s, :), 1));
    end
    
    % Vectorized 1D FFT along the lattice dimension N
    F_hat = fft(func_samples); 

    % Solve the overdetermined aliasing system for each fiber
    for r = 0:N-1
        idx = fibers{r+1};
        v = length(idx);
        
        if v > 0
            Gamma = A(idx, :); 
            
            % Construct the reconstruction system matrix B (size: S x v)
            B = exp(two_pi_i * shift_freq_prod(:, idx));
            
            % Extract the corresponding FFT coefficients (size: S x 1)
            FPs = F_hat(r+1, :).' / N; 
            
            % Least-squares solve to algebraically decouple the aliased frequencies
            Coeffs = B \ FPs;

            % Apply the phase correction for the external MC shift
            phase_correction = exp(-two_pi_i * Gamma * delta_shift);
            recovered_coeffs(coeff_idx:coeff_idx+v-1) = phase_correction .* Coeffs;

            active_freqs(coeff_idx:coeff_idx+v-1, :) = Gamma;
            coeff_idx = coeff_idx + v;
        end
    end

    % 4. Reconstruct the approximated function on the evaluation points
    func_eval_approx = evaluate_fourier_series_gpu(eval_pts, active_freqs, recovered_coeffs, 1024);
    
    if error_metric ~= 2
        errors_array(k) = norm(func_eval_exact - func_eval_approx, inf);
    else
        errors_array(k) = norm(func_eval_exact - real(func_eval_approx), 2)^2 / num_eval_pts;
    end
end

% 5. Finalize the relative error calculation
if error_metric ~= 2
    error_val = errors_array / norm_f;
else
    error_val = sqrt(mean(errors_array)) / norm_f;
end

end

% =========================================================================
% Local Helper Functions
% =========================================================================

function approx_vals = evaluate_fourier_series_gpu(eval_pts, freqs, coeffs, blk_size)
% EVALUATE_FOURIER_SERIES_GPU Evaluates the recovered Fourier series using GPU acceleration.
%
% Performance Note: 
%   To prevent out-of-memory errors on massive index sets \mathcal{A}(M), the 
%   matrix multiplications are executed in blocks and offloaded to the GPU.

    % Transfer data to GPU memory
    d_eval_pts = gpuArray(eval_pts);
    d_freqs = gpuArray(freqs);
    d_coeffs = gpuArray(coeffs);
    
    num_eval_pts = size(eval_pts, 1);
    d_approx_vals = gpuArray.zeros(num_eval_pts, 1, 'double');
    imag_unit = double(2 * pi * 1i);

    % Block-wise parallel evaluation of the exponential sum
    for i = 1:blk_size:num_eval_pts
        idx = i:min(i+blk_size-1, num_eval_pts);
        d_approx_vals(idx) = exp(imag_unit * (d_eval_pts(idx, :) * d_freqs')) * d_coeffs;
    end
    
    % Transfer results back to host memory
    approx_vals = double(gather(d_approx_vals)); 
end