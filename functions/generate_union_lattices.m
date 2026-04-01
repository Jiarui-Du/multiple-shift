function Y = generate_union_lattices(z, p_list)
z = z(:).';
d = length(z);
k = length(p_list);
S = sum(p_list);
Y = zeros(S, d);

idx = 1;
for j = 1:k
    p = p_list(j);
    i = (0:p-1).';
    Y(idx:idx+p-1, :) = mod(i * z / p, 1);
    idx = idx + p;
end
end