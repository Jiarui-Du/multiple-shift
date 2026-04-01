% RUN_S_TEST Evaluates the sampling complexity of different shift strategies.
%
% Purpose:
%   Reproduces Figure 1 from the paper. It evaluates and compares the total 
%   number of shifts S required by the polynomial curve strategy, 
%   single-prime lattice strategy, and the adaptive deterministic 
%   construction framework across varying lattice sizes N and dimensions d.

clear all;
close all;
format short;
clc;
addpath("./functions")
rng(26); % Fix random seed for reproducibility

%% 1. Parameters Setup
N_list_pool = 2.^(10:20); 
num_N = length(N_list_pool);
for i = 1:num_N
    N0 = N_list_pool(i);
k = 0;
while true
    if isprime(N0 + k)
        N = N0 + k;
        break;
    elseif N0 - k > 1 && isprime(N0 - k)
        N = N0 - k;
        break;
    end
    k = k + 1;
end
N_list(i) = N;
end

dim_list = [2:5, 10, 15, 20, 25, 30, 50];
num_dims = length(dim_list);

alpha = 1;
t = 0.95;
num_tasks = num_N * num_dims;

% Pre-generate flat task grids for parfor efficiency
[task_n_idx_grid, task_d_idx_grid] = meshgrid(1:num_N, 1:num_dims);
task_n_indices = task_n_idx_grid(:);
task_d_indices = task_d_idx_grid(:);

%% 2. Pre-allocate Output Arrays (1D arrays to support parfor)
res_S_poly = zeros(num_tasks, 1);
res_S_adap  = zeros(num_tasks, 1);
res_S_multi = zeros(num_tasks, 1);
res_S_single  = zeros(num_tasks, 1);
res_S_crt  = zeros(num_tasks, 1);
res_R      = zeros(num_tasks, 1);
res_N_real = zeros(num_tasks, 1);
res_nA     = zeros(num_tasks, 1);

res_time = zeros(num_tasks, 1);
res_time_poly = zeros(num_tasks, 1);
res_time_adap = zeros(num_tasks, 1);
res_time_multi = zeros(num_tasks, 1);
res_time_single  = zeros(num_tasks, 1);

%% 3. Parallel Execution
tic;
parfor k = 1:num_tasks
    g = [];
    gamma = [];
    M = [];
    A = [];
    fibers = [];
    d_idx = task_d_indices(k);
    n_idx = task_n_indices(k);
    
    d = dim_list(d_idx);
    N = N_list(n_idx);
    gamma = 2.^((1 - (1:d)) / 10);
    % gamma =  0.9.^((1:d)-1);
    
    % Construct generating vector and lattice fibers
    t0 = tic;
    g = construct_generating_vector_cbc(N, d, gamma, alpha);
    M = find_M_optimal(N, d, alpha, gamma);
    [A, fibers, ~] = construct_lattice_fibers(N, M, g, gamma, alpha);
    res_time(k) = toc(t0);

    pmin = floor((M*gamma(1))^(1/alpha))*floor((M*gamma(2))^(1/alpha))/N;
    
    fiber_lengths = cellfun(@length, fibers);
    R = max(fiber_lengths);
    nA = length(A);
    
    res_nA(k) = nA;
    res_R(k) = R;
    res_N_real(k) = N;
    
    % --- Timing: Strategy A (Polynomial Search) ---
    t1 = tic;
    [~, S_poly] = search_polynomial_shifts(d, A, fibers, R, t);
    res_time_poly(k) = toc(t1);
    
    % --- Timing: Strategy B (Single-Prime / CRT Search) ---
    t2 = tic;
    [~, S_single, S_crt] = search_crt_shifts(d, A, fibers, R, t);
    res_time_single(k) = toc(t2);
    
    % --- Timing: Strategy C (Multi-Prime Greedy Search) ---
    t3 = tic;
    [~, S_multi] = greedy_construction(d, A, fibers, R, t);
    res_time_multi(k) = toc(t3);
    
    % --- Timing: Algorithm 3 (Adaptive Deterministic Search) ---
    t4 = tic;
    [~, S_adap, ~] = adaptive_construction_new(d, A, fibers, R, t, pmin);
    res_time_adap(k) = toc(t4);   

    res_S_poly(k) = S_poly;
    res_S_adap(k) = S_adap;
    res_S_single(k) = S_single;
    res_S_crt(k) = S_crt;
    res_S_multi(k) = S_multi;
    
    fprintf('N = %d, d = %d\n', N, d);

    if mod(k, 10) == 0; java.lang.System.gc(); end
end
total_time = toc;
fprintf('Total parfor loop time: %.2f seconds\n', total_time);

%% 4. Data Restructuring & Saving
Data.S_poly = reshape(res_S_poly, num_dims, num_N);
Data.S_adap  = reshape(res_S_adap, num_dims, num_N);
Data.S_multi  = reshape(res_S_multi, num_dims, num_N);
Data.S_single  = reshape(res_S_single, num_dims, num_N);
Data.S_crt  = reshape(res_S_crt, num_dims, num_N);
Data.R      = reshape(res_R, num_dims, num_N);
Data.N_real = reshape(res_N_real, num_dims, num_N);
Data.nA     = reshape(res_nA, num_dims, num_N);

Data.time = reshape(res_time, num_dims, num_N);
Data.time_poly = reshape(res_time_poly, num_dims, num_N);
Data.time_single  = reshape(res_time_single, num_dims, num_N);
Data.time_multi  = reshape(res_time_multi, num_dims, num_N);
Data.time_adap  = reshape(res_time_adap, num_dims, num_N);

%% Save result
if ~exist('results', 'dir'), mkdir('results'); end
save('results/S_optimal.mat', 'Data', 'dim_list', 'N_list', 'num_N');

%% Load result
% load('results/S_optimal.mat')

%% 5. Visualization (Phase Transition of Shifts)
d_vals = dim_list;
N_exponents = log2(N_list);
IN_exponents = round(N_exponents);

idx_d_low  = find(d_vals == 2);   
idx_d_high = find(d_vals == 50);  
idx_N_max  = num_N; % N = 2^16

colors = lines(7);
c_poly = colors(1,:); 
c_single = colors(2,:); 
c_adap = colors(3,:);
c_multi = colors(4,:);

lw = 1.5; 
ms = 6;   

figure('Position', [100, 100, 1200, 350]);

% --- Panel 1: Low Dimension (d=2) vs N ---
subplot(1, 3, 1);
hold on; grid on; box on;
semilogy(N_exponents, Data.S_single(idx_d_low,:), '-s', 'Color', c_single, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(N_exponents, Data.S_poly(idx_d_low,:), '-d', 'Color', c_poly, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(N_exponents, Data.S_multi(idx_d_low,:), '-o',  'Color', c_multi, 'LineWidth', 2, 'MarkerFaceColor', c_multi);
semilogy(N_exponents, Data.S_adap(idx_d_low,:), '-o',  'Color', c_adap, 'LineWidth', 2, 'MarkerFaceColor', c_adap);
semilogy(N_exponents, sqrt(N_list)/2.5, '--',  'Color', colors(6,:), 'LineWidth', lw);
xlim([N_exponents(1), N_exponents(end)*1.05]); 
ax = gca;
set(gca, 'XTick', IN_exponents);
xtick_labels = arrayfun(@(x) sprintf('$2^{%d}$', x), IN_exponents, 'UniformOutput', false);
set(gca, 'TickLabelInterpreter', 'latex', 'XTickLabel', xtick_labels);
xlabel('Base Lattice Size $N$', 'Interpreter', 'latex');
ylabel('Number of Shifts $S$', 'Interpreter', 'latex');
title('Low Dimension ($d=2$)', 'Interpreter', 'latex');

leg = legend('Single ($S_{sing}$)', 'Polynomial ($S_{poly}$)', 'Multiple ($S_{multi}$)', 'Adaptive ($S_{adap}$)', '$\sqrt{N}/2.5$', ...
             'Location', 'northwest', 'Interpreter', 'latex');
set(leg, 'FontSize', 10);

yyaxis right
ax.YColor = [0, 0, 0]; 
plot(N_exponents, Data.R(idx_d_low, :), '--s', 'Color', colors(5,:), ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w','DisplayName', 'Max Fiber Length $R$');
% ylabel('Max Fiber Length $R$','Interpreter', 'latex');
ylim([0, max(Data.R(idx_d_low, :))+1]); 

% --- Panel 2: High Dimension (d=30) vs N ---
subplot(1, 3, 2);
hold on; grid on; box on;
x_fit = linspace(min(log2(N_list)), max(log2(N_list)), 100);
p_adap = polyfit(log2(N_list), Data.S_adap(idx_d_high,:), 2);
S_fit = polyval(p_adap, x_fit);
semilogy(x_fit, S_fit, '--',  'Color', colors(6,:), 'LineWidth', lw);
semilogy(N_exponents, Data.S_single(idx_d_high,:), '-s', 'Color', c_single, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(N_exponents, Data.S_poly(idx_d_high,:), '-d', 'Color', c_poly, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(N_exponents, Data.S_multi(idx_d_high,:), '-o',  'Color', c_multi, 'LineWidth', 2, 'MarkerFaceColor', c_multi);
semilogy(N_exponents, Data.S_adap(idx_d_high,:), '-o',  'Color', c_adap, 'LineWidth', 2, 'MarkerFaceColor', c_adap);

xlim([N_exponents(1), N_exponents(end)*1.05]); 
ax = gca;
set(gca, 'XTick', IN_exponents);
set(gca, 'TickLabelInterpreter', 'latex', 'XTickLabel', xtick_labels);
xlabel('Base Lattice Size $N$', 'Interpreter', 'latex');
title('High Dimension ($d=50$)', 'Interpreter', 'latex');
% ylabel('Number of Shifts $S$', 'Interpreter', 'latex');
leg = legend(sprintf('$%.2f (\\log_2(N))^2 %+.0f (\\log_2(N)) %+.0f$', p_adap(1), p_adap(2), p_adap(3)),...
             'Location', 'northwest', 'Interpreter', 'latex');
set(leg, 'FontSize', 10);

yyaxis right
ax.YColor = [0, 0, 0]; 
plot(N_exponents, Data.R(idx_d_high, :), '--s', 'Color', colors(5,:), ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w',...
    'DisplayName', 'Max Fiber Length $R$','HandleVisibility', 'off');
% ylabel('Max Fiber Length $R$','Interpreter', 'latex');
ylim([0, max(Data.R(idx_d_high, :))+1]); 

% --- Panel 3: Max Sample Size (N=2^16) vs d ---
subplot(1, 3, 3);
hold on; grid on; box on;
semilogy(d_vals, Data.S_single(:,idx_N_max), '-s', 'Color', c_single, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(d_vals, Data.S_poly(:,idx_N_max), '-d', 'Color', c_poly, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(d_vals, Data.S_multi(:,idx_N_max), '-o',  'Color', c_multi, 'LineWidth', 2, 'MarkerFaceColor', c_multi);
semilogy(d_vals, Data.S_adap(:,idx_N_max), '-o',  'Color', c_adap, 'LineWidth', 2, 'MarkerFaceColor', c_adap);

xlim([d_vals(1), d_vals(end)*1.05]); 
ax = gca;
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('Dimension $d$', 'Interpreter', 'latex');
% ylabel('Number of Shifts $S$', 'Interpreter', 'latex');
title(sprintf('$N = %d$', N_list(idx_N_max)), 'Interpreter', 'latex');

yyaxis right
ax.YColor = [0, 0, 0]; 
plot(d_vals, Data.R(:,idx_N_max), '--s', 'Color', colors(5,:), ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w');
ylabel('Max Fiber Length $R$','Interpreter', 'latex');
ylim([0, max(Data.R(:,idx_N_max))+1]); 

set(findall(gcf,'-property','FontSize'),'FontSize', 15);