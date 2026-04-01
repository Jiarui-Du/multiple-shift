function N = nearprime(N0)
k = 0;
while true
    if isprime(N0 + k)
        N = N0 + k;
        break;
    elseif N0 - k > 1 && isprime(N0 - k)
        N = N0 - k;
        break;
    end
    k = k + 1;
end
end