function Y = generate_union_lattices(z, p_list)
% Distinct prime blocks share one origin. A zero generator modulo p adds no nodes.
z = z(:).';
p_list = p_list(:).';
active = any(mod(z.', p_list) ~= 0, 1);
active_primes = p_list(active);
Y = zeros(1 + sum(active_primes - 1), numel(z));

first = 2;
for p = active_primes
    s = (1:p-1).';
    Y(first:first+p-2, :) = mod(s * z, p) / p;
    first = first + p - 1;
end
end
