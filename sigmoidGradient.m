function g = sigmoidGradient(z)


g = zeros(size(z));

sig = (e .^ (-z) + 1) .^ (-1);

g = sig .* (1 - sig);





end
