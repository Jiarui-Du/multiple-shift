% function acc = powmod(x, a, n)
%
% Calculate
%   x^a (mod n)
% using the Russian Peasant method.
%
% (C) 2003 <dirk.nuyens@cs.kuleuven.ac.be>
function y = powmod(x, a, n)

if ~usejava('jvm')
    y = 1; u = x;
    while a > 0
        if mod(a, 2) == 1
            y = mod(y * u, n); % this could overflow
        end;
        u = mod(u * u, n); % this could overflow
        a = floor(a / 2);
    end;
else
    x_ = java.math.BigInteger(num2str(x));
    a_ = java.math.BigInteger(num2str(a));
    n_ = java.math.BigInteger(num2str(n));
    y = x_.modPow(a_, n_).doubleValue();
end
end

function isok = usejava(feature)
% Octave doesn't support java so this function always returns false.
isok = false;
end