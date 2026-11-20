function coeffs = recover_gross_coefficients(Ms, zs, reco_indices, fft_indices, counter, target_func)
% RECOVER_GROSS_COEFFICIENTS Samples f and averages the alias-free FFT coefficients.
% This is the direct recovery in Gross et al.'s multi_lattice_ifft_noniter_avg,
% with the frequency maps prepared in Setup and the common origin sampled once.

coeffs = zeros(length(counter), 1);
origin_value = target_func(zeros(1, size(zs, 2)));
for ell = 1:numel(Ms)
    lattice_pts = mod((1:Ms(ell)-1)' * zs(ell, :), Ms(ell)) / Ms(ell);
    samples = [origin_value; target_func(lattice_pts)];
    F_hat = fft(samples);
    idx = reco_indices{ell};
    coeffs(idx) = coeffs(idx) + F_hat(fft_indices{ell}) / length(samples);
end
coeffs = coeffs ./ counter;
end
