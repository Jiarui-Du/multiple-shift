function [fibers, residues] = construct_lattice_fibers(A, g, N)
% CONSTRUCT_LATTICE_FIBERS Partitions an existing frequency set into fibers.
%
% Inputs:
%   A        - [Matrix, nA x d] The frequency index set.
%   g        - [Vector, d x 1] The lattice generating vector.
%   N        - [Integer] The prime number of lattice points.
%
% Outputs:
%   fibers   - [Cell Array, N x 1] Row indices of A in each residue class.
%              fibers{r+1} contains frequencies with (k*g) mod N = r.
%   residues - [Vector, nA x 1] The residue of each frequency.

residues = mod(A * g(:), N);
indices = residues + 1;
row_nums = (1:size(A, 1)).';
fibers = accumarray(indices, row_nums, [N, 1], @(x) {x}, {[]});
end
