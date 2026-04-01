function [err_u, err_f, S, nA, R] = evaluate_pde_approximation_error(N, M, d, alpha, gamma, t, num_mc_shifts, mc_shifts, target_u, target_f, tent_map, u_0_mean, norm_u, norm_f, eval_pts, error_metric)
% EVALUATE_PDE_APPROXIMATION_ERROR Solves a Neumann PDE via multiple-shift lattice rules.
%
% Purpose:
%   Evaluates the approximation error of a Poisson problem with homogeneous 
%   Neumann boundary conditions \nabla^2 u = f. The function f is evaluated 
%   using the deterministic multiple-shift lattice algorithm combined with a 
%   tent transformation to map it into the half-period cosine space C_{d,\alpha,\gamma}.
%   The PDE is then solved analytically in the frequency domain.
%
% Inputs:
%   N             - [Integer] Base lattice size N.
%   M             - [Double] Truncation parameter bounding \mathcal{A}(M).
%   d             - [Integer] Spatial dimension d.
%   alpha         - [Double] Smoothness parameter.
%   gamma         - [Vector] Positive weights \gamma_j for each dimension.
%   t             - [Double] Tolerance parameter ensuring condition number bounds.
%   num_mc_shifts - [Integer] Number of random shifts for out-of-sample error estimation.
%   mc_shifts     - [Matrix] Random shifts (d x num_mc_shifts).
%   target_u      - [Function Handle] Exact solution u(x).
%   target_f      - [Function Handle] Exact Laplacian source f(x).
%   tent_map      - [Function Handle] Tent transformation mapping [0,1]^d to [0,1]^d.
%   u_0_mean      - [Double] The known mean value \hat{u}(0) of the solution.
%   norm_u        - [Double] Analytical L2 norm of u.
%   norm_f        - [Double] Analytical L2 norm of f.
%   eval_pts      - [Matrix] Out-of-sample evaluation points.
%   error_metric  - [Integer] Metric selector: 2 for L_2, otherwise L_\infty.
%
% Outputs:
%   err_u - [Double] The computed relative approximation error of the PDE solution u.
%   err_f - [Double] The computed relative approximation error of the source function f.
%   S     - [Integer] Effective total number of deterministic shifts.
%   nA    - [Integer] Cardinality of the frequency index set |\mathcal{A}(M)|.
%   R     - [Integer] Maximum fiber length.

% 1. Construct base rank-1 lattice and its frequency structures
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

% 2. Generate deterministic multiple shifts
[Y, S, ~] = adaptive_construction_new(d, A, fibers, R, t, pmin);

active_freqs = zeros(nA, d);
num_eval_pts = size(eval_pts, 1);

% Map evaluation points to the non-periodic spatial domain
eval_pts_mapped = tent_map(eval_pts);
exact_u_eval = target_u(eval_pts_mapped);
exact_f_eval = target_f(eval_pts_mapped);

if error_metric ~= 2
    % norm_u = norm(exact_u_eval, inf);
    % norm_f = norm(exact_f_eval, inf);
    num_mc_shifts = 1;
    mc_shifts = zeros(d, 1);
end

errors_f = zeros(num_mc_shifts, 1);
errors_u = zeros(num_mc_shifts, 1);
two_pi_i = 2 * pi * 1i;
shift_freq_prod = Y * A.'; 

% 3. Main Approximation Loop
for k = 1:num_mc_shifts
    delta_shift = mc_shifts(:, k);       
    total_shift = Y + delta_shift';         
    
    recovered_coeffs = zeros(nA, 1);
    coeff_idx = 1;
    f_samples = zeros(N, S);
    
    % Sample the mapped target function f(x)
    for s = 1:S
       shifted_pts = mod(base_lattice_pts + total_shift(s, :), 1);
       mapped_pts = tent_map(shifted_pts);
       f_samples(:, s) = target_f(mapped_pts);
    end
    
    F_hat = fft(f_samples); 

    % Solve the overdetermined aliasing system for each fiber
    for r = 0:N-1
        idx = fibers{r+1};
        v = length(idx);
        
        if v > 0
            Gamma = A(idx, :); 
            B = exp(two_pi_i * shift_freq_prod(:, idx));
            FPs = F_hat(r+1, :).' / N; 
            
            % Least-squares solve
            B_conj_trans = B';
            G = B_conj_trans * B;
            Coeffs = G \ (B_conj_trans * FPs);

            phase_correction = exp(-two_pi_i * Gamma * delta_shift);
            recovered_coeffs(coeff_idx:coeff_idx+v-1) = phase_correction .* Coeffs;

            active_freqs(coeff_idx:coeff_idx+v-1, :) = Gamma;
            coeff_idx = coeff_idx + v;
        end
    end

    % 4. Coefficient Conversion via the Averaging Operator M
    % Map indices to absolute values to combine symmetric frequencies
    K_abs = abs(active_freqs);
    [unique_K, ~, idx_map] = unique(K_abs, 'rows');
    
    % Accumulate the real part of the coefficients (conjugate symmetry \hat{c}_{-h} = \overline{\hat{c}_h})
    sum_coeffs = accumarray(idx_map, real(recovered_coeffs));

    % Normalize to orthonormal cosine basis coefficients
    k_norm0 = sum(unique_K > 0, 2); 
    basis_factor = 1.0 ./ (sqrt(2) .^ k_norm0);
    f_hat_coeffs = sum_coeffs .* basis_factor;
    
    % 5. Spectral Solution of the Poisson Equation (\nabla^2 u = f)
    sum_k_sq = sum(unique_K.^2, 2);
    u_hat_coeffs = zeros(size(f_hat_coeffs));
    
    % Isolate the non-zero frequencies
    nz_idx = sum_k_sq > 0;
    
    % Assign the known physical mean value \hat{u}(0)
    u_hat_coeffs(~nz_idx) = u_0_mean;
    
    % Apply the inverse Laplacian operator in the frequency domain
    if any(nz_idx)
        u_hat_coeffs(nz_idx) = -f_hat_coeffs(nz_idx) ./ (pi^2 * sum_k_sq(nz_idx));
    end
    
    % 6. GPU Accelerated Function Reconstruction
    approx_f_eval = evaluate_cosine_series_gpu(d, num_eval_pts, eval_pts_mapped, unique_K, f_hat_coeffs);
    approx_u_eval = evaluate_cosine_series_gpu(d, num_eval_pts, eval_pts_mapped, unique_K, u_hat_coeffs);
    
    if error_metric ~= 2
        errors_f(k) = norm(exact_f_eval - approx_f_eval, inf);
        errors_u(k) = norm(exact_u_eval - approx_u_eval, inf); 
    else
        errors_f(k) = norm(exact_f_eval - approx_f_eval, 2)^2 / num_eval_pts; 
        errors_u(k) = norm(exact_u_eval - approx_u_eval, 2)^2 / num_eval_pts; 
    end
end

% Finalize relative errors
if error_metric ~= 2
    err_f = errors_f / norm_f;
    err_u = errors_u / norm_u;
else
    err_f = sqrt(mean(errors_f)) / norm_f;
    err_u = sqrt(mean(errors_u)) / norm_u;
end

end

% =========================================================================
% Local Helper Functions
% =========================================================================

function approx_vals = evaluate_cosine_series_gpu(dim_d, num_pts, eval_pts_mapped, freqs, coeffs)
% EVALUATE_COSINE_SERIES_GPU Evaluates the reconstructed cosine series.
%
% Performance Note: 
%   Memory usage is strictly bounded by processing the coefficient evaluations 
%   in chunks, effectively preventing out-of-memory errors on the GPU.

    g_pts = gpuArray(eval_pts_mapped); 
    g_freqs = gpuArray(freqs);
    g_coeffs = gpuArray(coeffs);
    
    num_coeffs = size(freqs, 1);
    g_approx_vals = gpuArray.zeros(num_pts, 1, 'double');
    
    blk_size = 1024; 
    sqrt2 = sqrt(2);
    pi_val = pi;
    
    for i = 1:blk_size:num_coeffs
        idx_end = min(i + blk_size - 1, num_coeffs);
        curr_idx = i:idx_end;
        blk_len = length(curr_idx);
        
        K_blk = g_freqs(curr_idx, :);      
        coeffs_blk = g_coeffs(curr_idx); 
        
        g_Mcos_blk = gpuArray.ones(num_pts, blk_len, 'double');
        
        for j = 1:dim_d
            kj = K_blk(:, j)'; 
            nz_mask = kj > 0;
            
            if any(nz_mask)
                sub_kj = kj(nz_mask); 
                args = (pi_val * g_pts(:, j)) * sub_kj;
                g_Mcos_blk(:, nz_mask) = g_Mcos_blk(:, nz_mask) .* (sqrt2 * cos(args));
            end
        end
        
        g_approx_vals = g_approx_vals + g_Mcos_blk * coeffs_blk;
    end
    
    approx_vals = double(gather(g_approx_vals));
end