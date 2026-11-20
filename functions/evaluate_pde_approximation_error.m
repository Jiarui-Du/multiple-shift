function [err_f, err_u, recovery_time, spectral_time, evaluation_time] = evaluate_pde_approximation_error( ...
    g, Y, prepared, num_mc_shifts, mc_shifts, target_u, target_f, tent_map, ...
    u_0_mean, norm_u, norm_f, eval_pts, error_metric)
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
%   g             - [Vector, d x 1] The base lattice generating vector.
%   Y             - [Matrix, S x d] Deterministic auxiliary shifts.
%   prepared      - [Struct] Fiber layout, factors, and cosine spectral data.
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
%   err_f - [Double] The computed relative approximation error of the source function f.
%   err_u - [Double] The computed relative approximation error of the PDE solution u.
%   recovery_time   - [Double] Mean sampling/FFT/solve time per global shift.
%   spectral_time   - [Double] Mean cosine conversion and spectral solve time per shift.
%   evaluation_time - [Double] Shared GPU evaluation time divided by the shift count.

d = size(Y, 2);
num_eval_pts = size(eval_pts, 1);
eval_pts_mapped = tent_map(eval_pts);
exact_u_eval = target_u(eval_pts_mapped);
exact_f_eval = target_f(eval_pts_mapped);

if error_metric ~= 2
    num_mc_shifts = 1;
    mc_shifts = zeros(d, 1);
end

errors_f = zeros(num_mc_shifts, 1);
errors_u = zeros(num_mc_shifts, 1);
recovery_times = zeros(num_mc_shifts, 1);
spectral_times = zeros(num_mc_shifts, 1);
nonzero_modes = prepared.nonzero_modes;
f_coeffs = zeros(size(prepared.cosine_freqs, 1), num_mc_shifts);
u_coeffs = zeros(size(f_coeffs));
u_coeffs(~nonzero_modes, :) = u_0_mean;
mapped_source = @(pts) target_f(tent_map(pts));

for k = 1:num_mc_shifts
    recovery_clock = tic;
    recovered_coeffs = recover_lattice_coefficients( ...
        g, Y, prepared, mc_shifts(:, k), mapped_source);
    recovery_times(k) = toc(recovery_clock);

    % Convert to the orthonormal cosine basis and solve the nonzero modes.
    spectral_clock = tic;
    f_coeffs(:, k) = accumarray(prepared.cosine_groups, real(recovered_coeffs)) .* prepared.basis_factor;
    u_coeffs(nonzero_modes, k) = -f_coeffs(nonzero_modes, k) ./ prepared.laplacian_eigenvalues;
    spectral_times(k) = toc(spectral_clock);
end
recovery_time = mean(recovery_times);
spectral_time = mean(spectral_times);

% Evaluate both fields and all shifts using the same cosine basis blocks.
evaluation_clock = tic;
approx_vals = evaluate_cosine_series_gpu(eval_pts_mapped, prepared.cosine_freqs, [f_coeffs, u_coeffs]);
evaluation_time = toc(evaluation_clock) / num_mc_shifts;

for k = 1:num_mc_shifts
    approx_f_eval = approx_vals(:, k);
    approx_u_eval = approx_vals(:, num_mc_shifts + k);
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

function approx_vals = evaluate_cosine_series_gpu(eval_pts_mapped, freqs, coeffs)
% Tile the points as well as the frequencies to bound GPU basis storage.
[num_pts, dim_d] = size(eval_pts_mapped);
num_freqs = size(freqs, 1);
g_pts = gpuArray(eval_pts_mapped);
g_freqs = gpuArray(freqs);
g_coeffs = gpuArray(coeffs);
g_approx_vals = gpuArray.zeros(num_pts, size(coeffs, 2), 'double');

frequency_block = min(1024, num_freqs);
device = gpuDevice;
% Allow space for phases, basis entries and temporary arrays.
max_entries = min(2^25, floor(device.AvailableMemory / 64));
point_block = max(1, floor(max_entries / frequency_block));
sqrt2 = sqrt(2);

for i = 1:frequency_block:num_freqs
    freq_rows = i:min(i + frequency_block - 1, num_freqs);
    block_coeffs = g_coeffs(freq_rows, :);
    nonzero_mask = freqs(freq_rows, :) > 0;

    for p = 1:point_block:num_pts
        point_rows = p:min(p + point_block - 1, num_pts);
        basis = gpuArray.ones(numel(point_rows), numel(freq_rows), 'double');
        for j = 1:dim_d
            columns = nonzero_mask(:, j);
            if any(columns)
                phases = (pi * g_pts(point_rows, j)) * g_freqs(freq_rows(columns), j).';
                basis(:, columns) = basis(:, columns) .* (sqrt2 * cos(phases));
            end
        end
        g_approx_vals(point_rows, :) = g_approx_vals(point_rows, :) + basis * block_coeffs;
        clear basis phases
    end
end

approx_vals = gather(g_approx_vals);
end
