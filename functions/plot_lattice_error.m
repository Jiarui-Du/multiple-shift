function plot_lattice_error(alpha, num_dims, dim_list, N_list, error_matrix, slopes, L_metric, ref_idx_k)
% PLOT_LATTICE_ERROR Visualizes the approximation error against the base lattice size.
%
% Purpose:
%   Plots the evaluated approximation error (L_2 or L_\infty) with respect to 
%   the base lattice size N. It compares the empirical convergence rates 
%   across different dimensions against the theoretical optimal bounds.
%
% Inputs:
%   alpha        - [Double] The smoothness parameter \alpha.
%   num_dims     - [Integer] The number of different dimensions evaluated.
%   dim_list     - [Vector] The list of spatial dimensions d.
%   N_list       - [Vector, 1 x num_N] The list of base lattice sizes N.
%   error_matrix - [Matrix, num_dims x num_N] The evaluated approximation errors.
%   slopes       - [Vector] Pre-computed empirical convergence slopes (legacy parameter).
%   L_metric     - [Integer] Metric selector: 2 for L_2 error, otherwise L_\infty.
%   ref_idx_k    - [Integer] The column index in N_list used to anchor the reference line.

hold on;

colors = lines(num_dims); 
markers = {'o', 's', 'd', '^', 'v', '>'};
legend_str = cell(num_dims + 2, 1); 

% Plot empirical error curves for each dimension
for idx = 1:num_dims
    d = dim_list(idx);
    
    loglog(N_list, error_matrix(idx, :), ...
        'Color', colors(idx, :), ...
        'Marker', markers{mod(idx-1, length(markers)) + 1}, ...
        'LineWidth', 2, ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', colors(idx, :));
    
    legend_str{idx} = sprintf('$d = %d$', d);
end

% Construct theoretical reference lines
base_N = N_list(1);
base_err = error_matrix(1, 1); 
ref_start = base_err * 2; 

ref_x = [N_list(1), N_list(end)];

if L_metric ~= 2
    effective_alpha = alpha - 0.5;
    metric_str = '$L_{\infty}$';
else
    effective_alpha = alpha;
    metric_str = '$L_2$';
end

% Reference 1: Optimal rate O(N^{-\alpha'})
ref_y1 = ref_start * (ref_x / base_N).^(-effective_alpha);
loglog(ref_x, ref_y1, '--', 'LineWidth', 1.5, 'Color', [0, 0.5, 0.5]);

% Reference 2: Probabilistic baseline rate O(N^{-0.5\alpha'})
ref_y2 = error_matrix(num_dims, ref_idx_k) * 1.2 * (ref_x / N_list(ref_idx_k)).^(-effective_alpha * 0.5);
loglog(ref_x, ref_y2, '--', 'LineWidth', 1.5, 'Color', [0.5, 0, 0.5]);

% Populate reference line legends using LaTeX formatting
if L_metric ~= 2
    legend_str{num_dims+1} = '$\mathcal{O}(N^{-(\alpha-1/2)})$';
    legend_str{num_dims+2} = '$\mathcal{O}(N^{-0.5(\alpha-1/2)})$';
else
    legend_str{num_dims+1} = '$\mathcal{O}(N^{-\alpha})$';
    legend_str{num_dims+2} = '$\mathcal{O}(N^{-0.5\alpha})$';
end

box on;
xlabel('Number of Lattice Points $N$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel([metric_str, ' Error'], 'Interpreter', 'latex', 'FontSize', 14);

lgd = legend(legend_str, 'Location', 'southwest', 'FontSize', 12);
set(lgd, 'Interpreter', 'latex'); 

set(gca, 'XScale', 'log', 'YScale', 'log');
xlim([N_list(1) * 0.9, N_list(end) * 1.1]);

hold off;
end