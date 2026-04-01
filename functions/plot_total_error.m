function plot_total_error(alpha, num_dims, dim_list, s_matrix, N_list, error_matrix, slopes, L_metric, ref_idx_k)
% PLOT_TOTAL_ERROR Visualizes the approximation error against the total computational cost.
%
% Purpose:
%   Plots the evaluated approximation error (L_2 or L_\infty) with respect to 
%   the total number of function evaluations N_{tot} = N \times S. It compares 
%   the empirical convergence rates across different dimensions against the 
%   theoretical optimal bounds derived for the weighted Korobov space.
%
% Inputs:
%   alpha        - [Double] The smoothness parameter \alpha.
%   num_dims     - [Integer] The number of different dimensions evaluated.
%   dim_list     - [Vector] The list of spatial dimensions d.
%   s_matrix     - [Matrix, num_dims x num_N] The total shifts S for each (d, N) pair.
%   N_list       - [Vector, 1 x num_N] The list of base lattice sizes N.
%   error_matrix - [Matrix, num_dims x num_N] The evaluated approximation errors.
%   slopes       - [Vector] Pre-computed empirical convergence slopes (legacy parameter).
%   L_metric     - [Integer] Metric selector: 2 for L_2 error, otherwise L_\infty.
%   ref_idx_k    - [Integer] The column index in N_list used to anchor the reference line.

hold on;

colors = lines(num_dims);
markers = {'o', 's', 'd', '^', 'v', '>'};
legend_str = cell(num_dims + 3, 1); 

min_cost = inf;
max_cost = 0;

% Plot empirical error curves for each dimension
for idx = 1:num_dims
    d = dim_list(idx);

    % Total function evaluations N_{tot} = N * S for the current dimension
    x_axis_data = N_list .* s_matrix(idx, :);

    min_cost = min(min_cost, min(x_axis_data));
    max_cost = max(max_cost, max(x_axis_data));

    loglog(x_axis_data, error_matrix(idx, :), ...
        'Color', colors(idx, :), ...
        'Marker', markers{mod(idx-1, length(markers)) + 1}, ...
        'LineWidth', 2, ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', colors(idx, :));

    legend_str{idx} = sprintf('$d = %d$', d);
end

% Construct theoretical reference lines
base_x = N_list(1) * s_matrix(ceil(num_dims/2), 1);
base_err = error_matrix(ceil(num_dims/2), 1); 
ref_start = base_err * 2; 

ref_x = linspace(min_cost, max_cost, 100);

if L_metric ~= 2
    effective_alpha = alpha - 0.5;
    metric_str = '$L_{\infty}$';
else
    effective_alpha = alpha;
    metric_str = '$L_2$';
end

% Reference 1: Optimal rate O(N_{tot}^{-\alpha'})
ref_y1 = ref_start * (ref_x / base_x).^(-effective_alpha);
loglog(ref_x, ref_y1, '--', 'LineWidth', 1.5, 'Color', [0.5, 0, 0.5]);

% Reference 2: Intermediate rate O(N_{tot}^{-0.75\alpha'})
ref_y2 = ref_start * (ref_x / base_x).^(-effective_alpha * 0.75);
loglog(ref_x, ref_y2, '--', 'LineWidth', 1.5, 'Color', [0, 0.5, 0.5]);

% Reference 3: Probabilistic baseline rate O(N_{tot}^{-0.5\alpha'})
anchor_x = N_list(ref_idx_k) * s_matrix(num_dims, ref_idx_k);
ref_y3 = error_matrix(num_dims, ref_idx_k) * 1.2 * (ref_x / anchor_x).^(-effective_alpha * 0.5);
loglog(ref_x, ref_y3, '--', 'LineWidth', 1.5, 'Color', [0.5, 0.5, 0]);

% Populate reference line legends using LaTeX formatting
if L_metric ~= 2
    legend_str{num_dims+1} = '$\mathcal{O}(N_{\mathrm{tot}}^{-(\alpha-1/2)})$';
    legend_str{num_dims+2} = '$\mathcal{O}(N_{\mathrm{tot}}^{-0.75(\alpha-1/2)})$';
    legend_str{num_dims+3} = '$\mathcal{O}(N_{\mathrm{tot}}^{-0.5(\alpha-1/2)})$';
else
    legend_str{num_dims+1} = '$\mathcal{O}(N_{\mathrm{tot}}^{-\alpha})$';
    legend_str{num_dims+2} = '$\mathcal{O}(N_{\mathrm{tot}}^{-0.75\alpha})$';
    legend_str{num_dims+3} = '$\mathcal{O}(N_{\mathrm{tot}}^{-0.5\alpha})$';
end

box on;
xlabel('Total Function Evaluations ($N \times S$)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel([metric_str, ' Error'], 'Interpreter', 'latex', 'FontSize', 14);

lgd = legend(legend_str, 'Location', 'southwest', 'FontSize', 12);
set(lgd, 'Interpreter', 'latex'); 

set(gca, 'XScale', 'log', 'YScale', 'log');
xlim([min_cost * 0.8, max_cost * 1.2]);

hold off;
end