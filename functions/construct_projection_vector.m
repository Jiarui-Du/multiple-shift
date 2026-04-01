function z = construct_projection_vector(H)
% FIND_GENERATING_VECTOR Deterministically constructs a projection vector z.
%
% Purpose:
%   Implements Algorithm 2 (Symmetric Component-by-Component Construction) to 
%   find an integer projection vector z such that h \cdot z \neq 0 for all 
%   h \in \mathcal{H} \setminus \{0\}.
%
% Inputs:
%   H - [Matrix, |\mathcal{H}| x d] The difference frequency matrix \mathcal{H}.
%
% Outputs:
%   z - [Vector, d x 1] The constructed integer vector z.

[num_rows, d] = size(H);

if d == 1
    z = 1;
    return;
end

if any(all(H == 0, 2))
    error('Mathematical Error: H contains a zero row. No valid projection vector exists.');
end

% Step 1: Identify the active column index I(i) for each row
active_col_indices = zeros(num_rows, 1);
for i = 1:num_rows
    active_col_indices(i) = find(H(i, :), 1, 'last');
end

z = zeros(d, 1);

% Step 3: Iterate through each dimension j to construct z_j
for j = 1:d
    rows_in_Hj = (active_col_indices == j);
    
    if ~any(rows_in_Hj)
        z(j) = 0;
        continue;
    end
    
    % Vectorized computation of prefix sums C_i and pivots h_{i,j}
    sub_matrix = H(rows_in_Hj, :);
    pivots = sub_matrix(:, j);
    
    if j == 1
        prefix_sums = zeros(sum(rows_in_Hj), 1);
    else
        prefix_sums = sub_matrix(:, 1:j-1) * z(1:j-1);
    end
    
    % Identify forbidden values for z_j
    is_integer_div = (mod(-prefix_sums, pivots) == 0);
    forbidden_values = -prefix_sums(is_integer_div) ./ pivots(is_integer_div);
    forbidden_values = unique(forbidden_values);
    
    % Step 4: Find the smallest integer not in forbidden_values
    num_forbidden = length(forbidden_values);
    max_val = num_forbidden + 1;
    
    pos_vals = 1:max_val;
    neg_vals = -pos_vals;
    search_seq = [0, reshape([pos_vals; neg_vals], 1, [])];
    
    for candidate = search_seq
        if ~ismember(candidate, forbidden_values)
            z(j) = candidate;
            break;
        end
    end
end
end