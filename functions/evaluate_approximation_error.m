function [error_val, recovery_time, evaluation_time] = evaluate_approximation_error(g, Y, prepared, num_mc_shifts, mc_shifts, target_func, norm_f, eval_pts, error_metric)
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
%   g             - [Vector, d x 1] The base lattice generating vector.
%   Y             - [Matrix, S x d] Deterministic auxiliary shifts.
%   prepared      - [Struct] Fiber layout and factors from prepare_lattice_recovery.
%   num_mc_shifts - [Integer] Number of random shifts for L_2 Monte Carlo error estimation.
%   mc_shifts     - [Matrix, d x num_mc_shifts] Random shifts for L_2 evaluation.
%   target_func   - [Function Handle] The target function f(x) to approximate.
%   norm_f        - [Double] The analytical or estimated norm of the target function.
%   eval_pts      - [Matrix, Neval x d] The out-of-sample points for error evaluation.
%   error_metric  - [Integer] Metric selector: 2 for L_2 error, otherwise L_\infty.
%
% Outputs:
%   error_val - [Double] The computed relative approximation error.
%   recovery_time   - [Double] Mean sampling/FFT/solve time per global shift.
%   evaluation_time - [Double] Mean Fourier evaluation time per global shift.

% 1. Use the structures formed in the main loop.
d = size(Y, 2);
num_eval_pts = size(eval_pts, 1);
func_eval_exact = target_func(eval_pts);

% Adjust evaluation logic based on the chosen error metric
if error_metric ~= 2
    % norm_f = norm(func_eval_exact, inf);
    num_mc_shifts = 1;
    mc_shifts = zeros(d, 1);
end

errors_array = zeros(num_mc_shifts, 1);
recovery_times = zeros(num_mc_shifts, 1);
evaluation_times = zeros(num_mc_shifts, 1);

% 2. Main Approximation Loop (Iterates over external MC shifts for L2 evaluation)
for k = 1:num_mc_shifts
    recovery_clock = tic;
    [recovered_coeffs, active_freqs] = recover_lattice_coefficients( ...
        g, Y, prepared, mc_shifts(:, k), target_func);
    recovery_times(k) = toc(recovery_clock);

    % 3. Reconstruct on the test points; gather waits for GPU completion.
    evaluation_clock = tic;
    func_eval_approx = evaluate_fourier_series_gpu(eval_pts, active_freqs, recovered_coeffs, 1024);
    evaluation_times(k) = toc(evaluation_clock);
    
    if error_metric ~= 2
        errors_array(k) = norm(func_eval_exact - func_eval_approx, inf);
    else
        errors_array(k) = norm(func_eval_exact - real(func_eval_approx), 2)^2 / num_eval_pts;
    end
end

% 4. Finalize the relative error calculation and per-shift timings.
if error_metric ~= 2
    error_val = errors_array / norm_f;
else
    error_val = sqrt(mean(errors_array)) / norm_f;
end
recovery_time = mean(recovery_times);
evaluation_time = mean(evaluation_times);

end
