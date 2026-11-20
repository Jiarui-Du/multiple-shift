function A = construct_frequency_set(N, M, d, gamma, alpha)
% CONSTRUCT_FREQUENCY_SET Enumerates the weighted hyperbolic cross A(M).
% N controls allocation only. The strict product test and frequency order
% are the same as in the original construct_lattice_fibers.

gamma = gamma(:).';
K_max = floor((M .* gamma).^(1/alpha));

% Pre-allocate the frequency array and start the recursive enumeration.
A_pts = zeros(ceil(N * 1.2), d);
pts_count = 0;
k_cur = zeros(1, d);
build_hyperbolic_cross(1, 1.0);
A = A_pts(1:pts_count, :);

    function build_hyperbolic_cross(dim, current_prod)
        if ~(current_prod < M)
            return;
        end

        % Find the last admissible coordinate using the original strict test.
        % This avoids traversing coordinates that cannot contribute to A(M).
        low = 0;
        high = K_max(dim);
        while low < high
            mid = ceil((low + high) / 2);
            if current_prod * max(1, mid^alpha / gamma(dim)) < M
                low = mid;
            else
                high = mid - 1;
            end
        end
        limit = low;

        % Append the last coordinate in one block, in the original order.
        if dim == d
            values = (-limit:limit).';
            count = numel(values);
            required_size = pts_count + count;
            if required_size > size(A_pts, 1)
                extra = ceil((required_size - size(A_pts, 1)) / N) * N;
                A_pts = [A_pts; zeros(extra, d)];
            end
            rows = pts_count + (1:count);
            A_pts(rows, :) = repmat(k_cur, count, 1);
            A_pts(rows, dim) = values;
            pts_count = required_size;
            return;
        end

        for val = -limit:limit
            factor = max(1, abs(val)^alpha / gamma(dim));
            new_prod = current_prod * factor;
            if new_prod < M
                k_cur(dim) = val;
                build_hyperbolic_cross(dim + 1, new_prod);
            end
        end
    end
end
