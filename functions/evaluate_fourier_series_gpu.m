function approx_vals = evaluate_fourier_series_gpu(eval_pts, freqs, coeffs, blk_size)
% EVALUATE_FOURIER_SERIES_GPU Evaluates the recovered Fourier series using GPU acceleration.
%
% Performance Note:
%   To prevent out-of-memory errors on massive index sets \mathcal{A}(M), the
%   matrix multiplications are executed in blocks and offloaded to the GPU.

    % Transfer data to GPU memory
    d_eval_pts = gpuArray(eval_pts);
    d_freqs = gpuArray(freqs);
    d_coeffs = gpuArray(coeffs);

    num_eval_pts = size(eval_pts, 1);
    d_approx_vals = gpuArray.zeros(num_eval_pts, 1, 'double');

    % Allow up to twice the old block size. Real phases and complex
    % basis values use 24 bytes per entry; leave half the free memory spare.
    device = gpuDevice;
    max_entries = min(2^25, floor(device.AvailableMemory / 48));
    blk_size = min(blk_size, max(1, floor(max_entries / size(freqs, 1))));
    % Round down, retaining the tested 512/1024-row choices when they fit.
    blk_size = 2^floor(log2(blk_size));

    % Block-wise parallel evaluation of the exponential sum
    for i = 1:blk_size:num_eval_pts
        idx = i:min(i+blk_size-1, num_eval_pts);
        phases = d_eval_pts(idx, :) * d_freqs';
        basis = arrayfun(@phase_exponential, phases);
        d_approx_vals(idx) = basis * d_coeffs;
        clear phases basis
    end

    % Transfer results back to host memory
    approx_vals = double(gather(d_approx_vals));
end

function value = phase_exponential(phase)
    % Fuse phase scaling and the complex exponential in one GPU operation.
    value = exp((2 * pi * 1i) * phase);
end
