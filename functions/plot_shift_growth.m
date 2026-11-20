function plot_shift_growth(N_list, dim_list, s_matrix, position, title_text)
% PLOT_SHIFT_GROWTH Plot the original d=2/4 data and quadratic log2(N) fits.
% The fit is descriptive and does not modify the measured shift counts.
log2_N = log2(N_list);
S_d2 = s_matrix(find(dim_list == 2), :);
S_d4 = s_matrix(find(dim_list == 4), :);

% Perform quadratic polynomial fitting
p_d2 = polyfit(log2_N, S_d2, 2);
p_d4 = polyfit(log2_N, S_d4, 2);
x_fit = linspace(min(log2_N), max(log2_N), 100);
S_fit_d2 = polyval(p_d2, x_fit);
S_fit_d4 = polyval(p_d4, x_fit);

figure('Position', position);
hold on; grid on; box on;
h1 = plot(log2_N, S_d2, 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
h2 = plot(log2_N, S_d4, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(x_fit, S_fit_d2, 'b--', 'LineWidth', 2);
plot(x_fit, S_fit_d4, 'r--', 'LineWidth', 2);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
xlabel('$\log_2(N)$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Number of shifts $S$', 'Interpreter', 'latex', 'FontSize', 14);

% Construct formatted legend strings for the quadratic models
leg_str_d2 = sprintf('$d=2$ Fit: $S \\approx %.2f (\\log_2(N))^2 %+.2f (\\log_2(N)) %+.2f$', p_d2(1), p_d2(2), p_d2(3));
leg_str_d4 = sprintf('$d=4$ Fit: $S \\approx %.2f (\\log_2(N))^2 %+.2f (\\log_2(N)) %+.2f$', p_d4(1), p_d4(2), p_d4(3));
legend([h1, h2], {leg_str_d2, leg_str_d4}, 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 11);

% Title adjusted to match the Korobov style
title(title_text, 'Interpreter', 'latex', 'FontSize', 16);

fprintf('\n--- Quadratic Fit Coefficients (S = a*x^2 + b*x + c) ---\n');
fprintf('d=2: a = %7.4f, b = %7.4f, c = %7.4f\n', p_d2(1), p_d2(2), p_d2(3));
fprintf('d=4: a = %7.4f, b = %7.4f, c = %7.4f\n', p_d4(1), p_d4(2), p_d4(3));

end
