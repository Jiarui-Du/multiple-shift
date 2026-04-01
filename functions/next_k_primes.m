function primes_out = next_k_primes(x_start, k)
primes_out = zeros(1, k);
cnt = 0;
p = max(x_start, 2);

if p > 2 && mod(p, 2) == 0
    p = p + 1;
end

while cnt < k
    if isprime(p)
        cnt = cnt + 1;
        primes_out(cnt) = p;
    end
    p = p + 1 + (p > 2);
end
end