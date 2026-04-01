% Return a generator for the cyclic group multiplication modulo p.
% Also called a primitive root of p.
%
% input:
%   p       a prime number
% output:
%   g       the generator
function g = generatorp(p)

if ~isprime(p), error('p is not a prime'); end;

primef = unique(factor(p-1));
g = 2; i = 1;
while i <= length(primef)
    if powmod(g, (p-1)/primef(i), p) == 1
        g = g + 1; i = 0;
    end;
    i = i + 1;
end;