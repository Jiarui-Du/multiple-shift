function p_next = next_prime(n)
if n <= 2
    p_next = 2;
    return;
elseif isprime(n)
    p_next = n;
    return;
end

p_next = n;
if mod(p_next,2) == 0
    p_next = p_next + 1;
end

while ~isprime(p_next)
    p_next = p_next + 2; 
end
end