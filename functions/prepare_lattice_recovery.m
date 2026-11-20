function prepared = prepare_lattice_recovery(A, fibers, Y)
% PREPARE_LATTICE_RECOVERY Factor each nonempty fiber before sampling.
% Use the original coefficient order: ascending residue, then rows in fiber order.

prepared.N = numel(fibers);
prepared.fft_rows = find(~cellfun(@isempty, fibers));
active_fibers = fibers(prepared.fft_rows);
prepared.offsets = [0; cumsum(cellfun(@numel, active_fibers(:)))];
prepared.frequencies = zeros(size(A));
num_fibers = numel(active_fibers);
prepared.BH = cell(num_fibers, 1);
prepared.L = cell(num_fibers, 1);
two_pi_i = 2 * pi * 1i;

for j = 1:num_fibers
    frequencies = A(active_fibers{j}, :);
    B = exp(two_pi_i * (Y * frequencies.'));
    BH = B';
    G = BH * B;
    [L, failure] = chol(G, 'lower');
    if failure ~= 0
        error('MultipleShift:NonpositiveGram', ...
            'Gram matrix for residue %d is not positive definite.', prepared.fft_rows(j) - 1);
    end
    rows = prepared.offsets(j)+1:prepared.offsets(j+1);
    prepared.frequencies(rows, :) = frequencies;
    prepared.BH{j} = BH;
    prepared.L{j} = L;
end
end
