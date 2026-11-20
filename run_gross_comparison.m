% RUN_GROSS_COMPARISON Compares multiple shifts with Gross et al. (2021).
% Both methods use the same frequency set and 2^18 test points.
% Setup starts with A given. Recovery includes forming nodes and sampling.
% Errors are absolute L_infinity estimates with zero global shift.
% Gross source: https://users.math.msu.edu/users/iwenmark/Code/MultipleRank1Lattice_index.html

clear all;
close all;
format compact;
format short;
clc;
addpath('./functions');
gross_source = fullfile('functions', 'MultipleRank1Lattice', 'SupplementaryMaterial');
assert(isfile(fullfile(gross_source, 'deterministic_mr1l_approx_get_lattice_sizes.m')), ...
    'Run setup_dependencies.ps1 from the repository root to obtain the Gross et al. routines.');
addpath(gross_source);
rng(2026);

%% 1. Experimental Parameters Setup
N_list_pool = 2.^(8:20);
num_N = length(N_list_pool);
for i = 1:num_N
    N0 = N_list_pool(i);
    N_list(i) = nearprime(N0);
end
dim_list = [2,4];
alpha = 2.5;
threshold_t = 0.95;
num_eval_pts = 2^18;
num_timing_runs = 1;
target_func = @(x) prod((x - 1/2).^2 .* sin(2 * pi * x - pi), 2);
norm_f = 1;

num_dims = length(dim_list);
method_names = {'Multiple shift', 'Gross (explicit)'};
error_matrix = zeros(num_dims, num_N, 2);
sample_matrix = zeros(num_dims, num_N, 2);
frequency_time_matrix = zeros(num_dims, num_N);
setup_time_matrix = zeros(num_dims, num_N, 2);
recovery_time_matrix = zeros(num_dims, num_N, 2);
evaluation_time_matrix = zeros(num_dims, num_N, 2);
setup_times = zeros(num_dims, num_N, 2, num_timing_runs);
recovery_times = zeros(num_dims, num_N, 2, num_timing_runs);
nA_matrix = zeros(num_dims, num_N);
s_matrix = zeros(num_dims, num_N);
max_len_matrix = zeros(num_dims, num_N);
gross_sizes = cell(num_dims, num_N);
test_point_states = cell(num_dims, num_N);

%% 2. Main Serial Computation Loop
for n_idx = 1:num_N
    for d_idx = 1:num_dims
        N = N_list(n_idx);
        d = dim_list(d_idx);
        gamma = 2.^((1 - (1:d)) / 10);

        % Common preparation, excluded from both Setup times.
        frequency_clock = tic;
        M = find_M_optimal(N, d, alpha, gamma);
        A = construct_frequency_set(N, M, d, gamma, alpha);
        frequency_time_matrix(d_idx, n_idx) = toc(frequency_clock);
        nA = size(A, 1);
        test_point_states{d_idx, n_idx} = rng;
        sobol_seq = scramble(sobolset(d), 'MatousekAffineOwen');
        eval_pts = net(sobol_seq, num_eval_pts);
        exact_values = target_func(eval_pts);

        % Multiple-shift Setup: CBC, fibers, shifts, Gram matrices and factors.
        for repeat = 1:num_timing_runs
            clear prepared
            setup_clock = tic;
            g = construct_generating_vector_cbc(N, d, gamma, alpha);
            [fibers, ~] = construct_lattice_fibers(A, g, N);
            max_len = max(cellfun(@length, fibers));
            pmin = floor((M*gamma(1))^(1/alpha))*floor((M*gamma(2))^(1/alpha))/N;
            [Y, S_val, ~] = adaptive_construction_new(d, A, fibers, max_len, threshold_t, pmin);

            prepared = prepare_lattice_recovery(A, fibers, Y);
            setup_times(d_idx, n_idx, 1, repeat) = toc(setup_clock);
            clear fibers
        end

        % Gross Setup: explicit full-grid generator, then the approximation
        % construction (not the recursive polynomial-only construction).
        for repeat = 1:num_timing_runs
            setup_clock = tic;
            N_I = max(max(A, [], 1) - min(A, [], 1) + 1);
            initial_size = N_I^d;
            % Follow the author's explicit generator, using factors only
            % when the integer powers or projections become too large.
            if N_I^(d-1) > flintmax || max(sum(abs(A), 2))*N_I^(d-1) > 2^62
                z = cell(1, d);
                z{1} = int64(1);
                for j = 2:d
                    z{j} = repmat(int64(N_I), 1, j-1);
                end
            else
                z = N_I.^(0:d-1);
            end
            Ms = deterministic_mr1l_approx_get_lattice_sizes(A, z, initial_size);

            zs = zeros(length(Ms), d);
            reco_indices = cell(length(Ms), 1);
            fft_indices = cell(length(Ms), 1);
            counter = zeros(nA, 1);
            for ell = 1:length(Ms)
                z_mod = ones(1, d, 'int64');
                for j = 2:d
                    z_mod(j) = mod(z_mod(j-1) * int64(N_I), int64(Ms(ell)));
                end
                zs(ell, :) = double(z_mod);
                bins = mod(A * zs(ell, :)', Ms(ell)) + 1;
                counts = accumarray(bins, 1, [Ms(ell), 1]);
                idx = find(counts(bins) == 1);
                reco_indices{ell} = idx;
                fft_indices{ell} = bins(idx);
                counter(idx) = counter(idx) + 1;
            end
            setup_times(d_idx, n_idx, 2, repeat) = toc(setup_clock);
        end

        % Recovery: form sampling nodes, sample and recover Fourier coefficients.
        % Repeat only Setup/Recovery; evaluate each final approximation once.
        for repeat = 1:num_timing_runs
            recovery_clock = tic;
            [shift_coeffs, shift_freqs] = recover_lattice_coefficients( ...
                g, Y, prepared, zeros(d, 1), target_func);
            recovery_times(d_idx, n_idx, 1, repeat) = toc(recovery_clock);

            recovery_clock = tic;
            gross_coeffs = recover_gross_coefficients( ...
                Ms, zs, reco_indices, fft_indices, counter, target_func);
            recovery_times(d_idx, n_idx, 2, repeat) = toc(recovery_clock);
        end

        % Use the existing GPU Fourier evaluation and complex absolute error.
        evaluation_clock = tic;
        shift_values = evaluate_fourier_series_gpu(eval_pts, shift_freqs, shift_coeffs, 1024);
        evaluation_time_matrix(d_idx, n_idx, 1) = toc(evaluation_clock);
        error_matrix(d_idx, n_idx, 1) = norm(exact_values - shift_values, inf) / norm_f;

        evaluation_clock = tic;
        gross_values = evaluate_fourier_series_gpu(eval_pts, A, gross_coeffs, 1024);
        evaluation_time_matrix(d_idx, n_idx, 2) = toc(evaluation_clock);
        error_matrix(d_idx, n_idx, 2) = norm(exact_values - gross_values, inf) / norm_f;

        nA_matrix(d_idx, n_idx) = nA;
        s_matrix(d_idx, n_idx) = S_val;
        max_len_matrix(d_idx, n_idx) = max_len;
        gross_sizes{d_idx, n_idx} = Ms;
        sample_matrix(d_idx, n_idx, 1) = N * S_val;
        sample_matrix(d_idx, n_idx, 2) = 1 + sum(Ms - 1);
        setup_time_matrix(d_idx, n_idx, :) = median(setup_times(d_idx, n_idx, :, :), 4);
        recovery_time_matrix(d_idx, n_idx, :) = median(recovery_times(d_idx, n_idx, :, :), 4);

        fprintf('\nN = %6d, d = %d, |A| = %6d\n', N, d, nA);
        for method = 1:2
            fprintf('  %-16s | samples = %8d | Error = %.3e | Setup = %.3f s | Recovery = %.3f s | Evaluation = %.3f s\n', ...
                method_names{method}, sample_matrix(d_idx, n_idx, method), ...
                error_matrix(d_idx, n_idx, method), setup_time_matrix(d_idx, n_idx, method), ...
                recovery_time_matrix(d_idx, n_idx, method), evaluation_time_matrix(d_idx, n_idx, method));
        end
        clear prepared
    end
end

%% 3. Save Results
result_fields = {'N_list', 'dim_list', 'alpha', 'num_eval_pts', 'method_names', ...
    'nA_matrix', 'error_matrix', 'sample_matrix', 'setup_time_matrix', 'recovery_time_matrix'};
available_fields = who;
missing_fields = setdiff(result_fields, available_fields, 'stable');
if ~isempty(missing_fields)
    error('GrossComparison:IncompleteResults', 'Missing result fields: %s.', strjoin(missing_fields, ', '));
end

optional_fields = {'evaluation_time_matrix', 'threshold_t', 'num_timing_runs', ...
    'norm_f', 'test_point_states', 's_matrix', 'max_len_matrix', 'gross_sizes', ...
    'setup_times', 'recovery_times', 'frequency_time_matrix'};
saved_optional_fields = intersect(optional_fields, available_fields, 'stable');
result_fields = [result_fields, saved_optional_fields(:).'];
if ~exist('results', 'dir'), mkdir('results'); end
result_file = fullfile('results', sprintf('gross_comparison_Linf_eval%d.mat', num_eval_pts));
save(result_file, result_fields{:});
fprintf('Saved results to %s\n', result_file);

%% 4. Error Comparisons (both dimensions on the same axes)
% Load a saved result and run Sections 4--7 to redraw without computation.
if ~exist('figs', 'dir'), mkdir('figs'); end
% Color and marker identify the method. Line style identifies the dimension.
curve_colors = lines(2);
curve_markers = {'o', 's'};
line_styles = {'-', '--', '-.', ':'};
curve_styles = line_styles(mod(0:length(dim_list)-1, length(line_styles)) + 1);
marker_faces = {curve_colors(1, :), 'none'};
curve_names = cell(length(dim_list), 2);
for d_idx = 1:length(dim_list)
    for method = 1:2
        curve_names{d_idx, method} = sprintf('%s, $d = %d$', method_names{method}, dim_list(d_idx));
    end
end
reference_colors = [0.5 0 0.5; 0 0.5 0.5];
error_title = sprintf('Convergence for $f_1$ ($\\alpha = %.1f$)', alpha);
x_labels = {'Total Function Evaluations $N_{\mathrm{tot}}$', 'Base Lattice Size $N$'};
file_labels = {'error', 'error_N'};
alpha_eff = alpha - 0.5;
n_ref = min(9, length(N_list));
for horizontal = 1:2
    figure('Position', [100, 100, 600, 500], 'Color', 'w');
    for d_idx = 1:length(dim_list)
        for method = 1:2
            x = N_list;
            if horizontal == 1
                x = sample_matrix(d_idx, :, method);
            end
            loglog(x, error_matrix(d_idx, :, method), curve_styles{d_idx}, ...
                'Color', curve_colors(method, :), 'Marker', curve_markers{method}, ...
                'MarkerFaceColor', marker_faces{method}, ...
                'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', curve_names{d_idx, method});
            hold on;
        end
    end
    axis padded; box on;
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12, ...
        'XGrid', 'off', 'YGrid', 'off', 'XMinorGrid', 'off', 'YMinorGrid', 'off');
    xlabel(x_labels{horizontal}, 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('$L_\infty$ Error', 'Interpreter', 'latex', 'FontSize', 14);
    if horizontal == 1
        % Match the Korobov total-evaluation guides, spanning both methods' costs.
        costs = sample_matrix(isfinite(sample_matrix) & sample_matrix > 0);
        ref_x = [min(costs), max(costs)];
        ref_slopes = -alpha_eff * [1, 0.75, 0.5];
        ref_labels = {'$\mathcal{O}(N_{\mathrm{tot}}^{-(\alpha-1/2)})$', ...
            '$\mathcal{O}(N_{\mathrm{tot}}^{-0.75(\alpha-1/2)})$', ...
            '$\mathcal{O}(N_{\mathrm{tot}}^{-0.5(\alpha-1/2)})$'};
        ref_colors = [reference_colors; 0.5 0.5 0];
        base_row = ceil(length(dim_list)/2);
        ref_centers = [sample_matrix(base_row, 1, 1), sample_matrix(base_row, 1, 1), sample_matrix(end, n_ref, 1)];
        base_error = 2*error_matrix(base_row, 1, 1);
        ref_heights = [base_error, base_error, 1.2*error_matrix(end, n_ref, 1)];
    else
        % Match the base-lattice Korobov L_inf references, using multiple-shift errors for placement.
        ref_slopes = [-alpha_eff, -0.5*alpha_eff];
        ref_labels = {'$\mathcal{O}(N^{-(\alpha-1/2)})$', '$\mathcal{O}(N^{-0.5(\alpha-1/2)})$'};
        ref_colors = reference_colors([2, 1], :);
        ref_x = N_list([1, end]);
        ref_centers = N_list([1, n_ref]);
        ref_heights = [2*error_matrix(1, 1, 1), 1.2*error_matrix(end, n_ref, 1)];
    end
    for reference = 1:numel(ref_slopes)
        loglog(ref_x, ref_heights(reference) * (ref_x/ref_centers(reference)).^ref_slopes(reference), ...
            '--', 'Color', ref_colors(reference, :), 'LineWidth', 1.5, ...
            'DisplayName', ref_labels{reference});
    end
    axis padded;
    plot_x_limits = xlim; plot_y_limits = ylim;
    x_decades = ceil(log10(plot_x_limits(1))):floor(log10(plot_x_limits(2)));
    y_decades = ceil(log10(plot_y_limits(1))):floor(log10(plot_y_limits(2)));
    if numel(x_decades) > 1, xticks(10.^x_decades); end
    if numel(y_decades) > 1, yticks(10.^y_decades); end
    legend('Interpreter', 'latex', 'Location', 'southwest', 'FontSize', 10);
    title(error_title, 'Interpreter', 'latex', 'FontSize', 16);
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, sprintf('figs/gross_comparison_Linf_eval%d_%s.eps', num_eval_pts, file_labels{horizontal}), '-depsc', '-vector');
    print(gcf, sprintf('figs/gross_comparison_Linf_eval%d_%s.png', num_eval_pts, file_labels{horizontal}), '-dpng', '-r150');
end

% Compare errors and oversampling at the same frequency set (measured only).
% Similar errors do not imply identical errors; larger oversampling is not
% universal at small N. Report the largest completed case for each dimension.
for d_idx = 1:length(dim_list)
    last = length(N_list);
    fprintf('d=%d, |A|=%d: Gross/MS error ratio = %.3f; samples/|A|: MS = %.2f, Gross = %.2f\n', ...
        dim_list(d_idx), nA_matrix(d_idx, last), ...
        error_matrix(d_idx, last, 2)/error_matrix(d_idx, last, 1), ...
        sample_matrix(d_idx, last, 1)/nA_matrix(d_idx, last), ...
        sample_matrix(d_idx, last, 2)/nA_matrix(d_idx, last));
end

%% 5. Setup, Recovery and Their Sum versus Base Lattice Size N
time_matrices = {setup_time_matrix, recovery_time_matrix, setup_time_matrix + recovery_time_matrix};
time_labels = {'Pre-computation', 'Recovery', 'Pre-computation + recovery'};
figure('Position', [150, 150, 1500, 500], 'Color', 'w');
for stage = 1:3
    subplot(1, 3, stage);
    times = time_matrices{stage};
    % Incomplete runs leave zeros; logarithmic plots require finite positive times.
    times(~isfinite(times) | times <= 0) = NaN;
    for d_idx = 1:length(dim_list)
        for method = 1:2
            loglog(N_list, times(d_idx, :, method), curve_styles{d_idx}, ...
                'Color', curve_colors(method, :), 'Marker', curve_markers{method}, ...
                'MarkerFaceColor', marker_faces{method}, ...
                'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', curve_names{d_idx, method});
            hold on;
        end
    end
    axis padded; grid on; box on;
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12, ...
        'XMinorGrid', 'off', 'YMinorGrid', 'off');
    xlabel('Base Lattice Size $N$', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 14);
    title(time_labels{stage}, 'Interpreter', 'latex', 'FontSize', 16);
    % Normalize the power guides at the largest N with a plotted time.
    positive_times = times(isfinite(times));
    if ~isempty(positive_times)
        last = find(any(isfinite(times), [1 3]), 1, 'last');
        end_times = times(:, last, :);
        ref_x = N_list([1, end]);
        ref_ratio = ref_x / N_list(last);
        linear_guide = 0.3 * min(end_times, [], 'all', 'omitnan') * ref_ratio;
        loglog(ref_x, linear_guide, '--', 'Color', reference_colors(2, :), ...
            'LineWidth', 1.5, 'DisplayName', '$N$ reference');
        y_values = [positive_times(:); linear_guide(:)];
        if stage ~= 2   % Recovery has a linear reference only.
            quadratic_guide = 2.5 * max(end_times, [], 'all', 'omitnan') * ref_ratio.^2;
            loglog(ref_x, quadratic_guide, '--', 'Color', reference_colors(1, :), ...
                'LineWidth', 1.5, 'DisplayName', '$N^2$ reference');
            y_values = [y_values; quadratic_guide(:)];
        end
        first_exponent = floor(log10(min(y_values)));
        last_exponent = max(first_exponent + 1, ceil(log10(max(y_values))));
        ylim(10.^[first_exponent, last_exponent]);
        yticks(10.^(first_exponent:last_exponent));
    end
    plot_x_limits = xlim;
    x_decades = ceil(log10(plot_x_limits(1))):floor(log10(plot_x_limits(2)));
    if numel(x_decades) > 1, xticks(10.^x_decades); end
    legend('Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 10);
end
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, sprintf('figs/gross_comparison_Linf_eval%d_runtime.eps', num_eval_pts), '-depsc', '-vector');
print(gcf, sprintf('figs/gross_comparison_Linf_eval%d_runtime.png', num_eval_pts), '-dpng', '-r150');

%% 6. Oversampling Factor versus |A|
figure('Position', [100, 100, 600, 500], 'Color', 'w');
for d_idx = 1:length(dim_list)
    for method = 1:2
        oversampling = sample_matrix(d_idx, :, method) ./ nA_matrix(d_idx, :);
        oversampling(~isfinite(oversampling) | oversampling <= 0) = NaN;
        semilogx(nA_matrix(d_idx, :), oversampling, curve_styles{d_idx}, ...
            'Color', curve_colors(method, :), 'Marker', curve_markers{method}, ...
            'MarkerFaceColor', marker_faces{method}, ...
            'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', curve_names{d_idx, method});
        hold on;
    end
end
axis padded; grid on; box on;
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12, ...
    'XMinorGrid', 'off', 'YMinorGrid', 'off');
xlabel('Number of Frequencies $|A|$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Oversampling factor (function evaluations / $|A|$)', ...
    'Interpreter', 'latex', 'FontSize', 14);
legend('Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 10);
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, sprintf('figs/gross_comparison_Linf_eval%d_oversampling.eps', num_eval_pts), '-depsc', '-vector');
print(gcf, sprintf('figs/gross_comparison_Linf_eval%d_oversampling.png', num_eval_pts), '-dpng', '-r150');

%% 7. Auxiliary Shift Counts for the Multiple-Shift Method
plot_shift_growth(N_list, dim_list, s_matrix, [100, 100, 600, 500], '');
set(gcf, 'Color', 'w', 'PaperPositionMode', 'auto');
print(gcf, sprintf('figs/gross_comparison_Linf_eval%d_shifts.eps', num_eval_pts), '-depsc', '-vector');
