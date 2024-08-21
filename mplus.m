function C = mplus(A, B)
%MPLUS Perform the 'multiplication' operation of the 'plus-plus' semiring
%algebra.
%
%   C = MTIMES(A,B) is the 'plus-plus' product of A and B. If A or B is a scalar
%   (a 1-by-1 matrix) this just reduces to element-wise addition.
%   Otherwise, the number of columns of A must equal the number of rows of
%   B. The output matrix is equal to C(i,j) = sum(A(:,i).' + B(:,j).
%
%   by Dillon Cislo 2024/08/20

% Validate the number of inputs
if nargin ~= 2
    error('mplus:InvalidInput', ...
        'Function requires exactly two input arguments.');
end

%--------------------------------------------------------------------------
% GPU COMPUTATION
%--------------------------------------------------------------------------

% Determine if inputs are on the GPU or CPU
isAonGPU = isa(A, 'gpuArray');
isBonGPU = isa(B, 'gpuArray');

if (isAonGPU && isBonGPU)

    % If A and B are empty, return an empty matrix
    if (isempty(A) || isempty(B))
        C = gpuArray([]);
        return;
    end

    % This function does not have support for sparse gpuArrays
    if (issparse(A) || issparse(B))
        error('mplus:MixedInput', ...
            'This function has no support for sparse gpuArrays');
    end

    % Handle type conversions. I will follow the convention that the output
    % should be cast to the lowest precision of any input variable
    if isa(A, 'single')
        if (isa(B, 'double') || islogical(B))
            B = single(B);
        end
    end

    if isa(B, 'single')
        if (isa(A, 'double') || islogical(A))
            A = single(A);
        end
    end

    % Convert remaining logical arrays to double
    if islogical(A), A = double(A); end
    if islogical(B), B = double(B); end

    % For now this function only supports floats on the GPU
    if ~float(A) || ~float(B)
        error('mplus:InvalidType', ...
            'Both inputs must be numeric or logical arrays.');
    end

    % If either A or B is scalar, broadcast as needed
    if isscalar(A), A = repmat(A, [1, size(B,1)]); end
    if isscalar(B), B = repmat(B, [size(A,2), 1]); end

    % Validate the input dimensions
    if (size(A,2) ~= size(B,1))
        error('mplus:DimensionMismatch', ...
            'Inner dimensions of A and B must agree.');
    end

    % Perform the 'mplus' operation. The output of this function should
    % have the appropriate floating point type relative to the input
    C = mplus_gpu(A, B);

    return;

elseif xor(isAonGPU, isBonGPU)

    % Ensure both inputs are either on the CPU or GPU
    error('mplus:MixedInput', ...
        'Both inputs must be either GPU arrays or CPU arrays.');
end

%--------------------------------------------------------------------------
% CPU COMPUTATION
%--------------------------------------------------------------------------

% If A and B are empty, return an empty matrix
if (isempty(A) || isempty(B))
    C = gpuArray([]);
    return;
end

% If either A or B is scalar, broadcast as needed
if isscalar(A), A = repmat(A, [1, size(B,1)]); end
if isscalar(B), B = repmat(B, [size(A,2), 1]); end

% Validate the input dimensions
if (size(A,2) ~= size(B,1))
    error('mplus:DimensionMismatch', ...
        'Inner dimensions of A and B must agree.');
end

% Perform the 'mplus' operation. GraphBLAS can do the type handling for the
% inputs
C = GrB.mxm('+.+', A, B);

% If either input is a GraphBLAS array, just return the result as is
if (isa(A, 'GrB') || isa(B, 'GrB')), return; end

% MATLAB only has sparse array support for logicals and doubles. I will
% follow the MATLAB convention that addition/multiplication of one dense
% and one sparse matrix results in a dense matrix.
if ~(issparse(A) && issparse(B))

    % Internally, GraphBLAS seems to cast output up to the highest
    % precision of any input variable: logical >> integer type >> single >>
    % double. I want to mirror my choice earlier to cast down from double
    % to single, but otherwise I will follow this ordering
    if (isa(A, 'double') && isa(B, 'single')) || ...
            (isa(A, 'single') && isa(B, 'double'))
        C = cast(C, 'single');   
    else
        C = cast(C, GrB.type(C));
    end

else

    if isa(A, 'double')
        C = cast(C, 'like', A);
    else
        C = cast(C, 'like', B);
    end

end


end

