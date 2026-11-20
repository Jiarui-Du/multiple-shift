function [recovered_coeffs, active_freqs] = recover_lattice_coefficients(g, Y, prepared, delta_shift, target_func)
% RECOVER_LATTICE_COEFFICIENTS Sample, apply FFTs and reuse the fiber factors.

N = prepared.N;
S = size(Y, 1);
active_freqs = prepared.frequencies;
base_lattice_pts = mod((0:N-1)' * g(:).', N) / N;
total_shift = Y + delta_shift(:).';
func_samples = zeros(N, S);

for s = 1:S
    func_samples(:, s) = target_func(mod(base_lattice_pts + total_shift(s, :), 1));
end
F_hat = fft(func_samples, [], 1);

recovered_coeffs = zeros(size(active_freqs, 1), 1);
for j = 1:numel(prepared.fft_rows)
    b = F_hat(prepared.fft_rows(j), :).'/N;
    rhs = prepared.BH{j} * b;
    L = prepared.L{j};
    rows = prepared.offsets(j)+1:prepared.offsets(j+1);
    recovered_coeffs(rows) = L' \ (L \ rhs);
end

if any(delta_shift)
    phase = exp(-2 * pi * 1i * (active_freqs * delta_shift(:)));
    recovered_coeffs = phase .* recovered_coeffs;
end
end
