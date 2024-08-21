# mplus
A MATLAB/mex implementation of the "multiplication" operator in the 'plus.plus' semiring algebra.

From GraphBLAS: "MATLAB can only compute `C=A*B` using the standard `'+.*.double'` and `'+.*.complex'` semirings. A semiring is defined in terms of a string, `'add.mult.type'`, where `add` is a monoid that takes the place of the additive operator, `mult` is the multiplicative operator, and `type` is the data type for the two inputs to the `mult` operator.

In the standard semiring, `C=A*B` is defined as:

    C(i,j) = sum(A(i,:).' .* B(:,j))

using `plus` as the monoid and `times` as the multiplicative operator. But in a more general semiring, `sum` can be any monoid, which is an associative and commutative operator that has an identity value."

The semiring algebra of interest here is the 'plus.plus' tropical algebra, where `C(i,j)` for `C=A*B` is defined as:

    C(i,j) = sum(A(i,:).' + B(:,j))

Why do this? There are lots of reasons, but the primary goal for me is to speed up stable computation of matrix-matrix products. Consider the computation `C=exp(A)*B` -- if the elements of `A` are really small (large) than repeated application of this operation can quickly run into numerical underflow (overflow) issues. A common solution is to instead compute `log(exp(A)*B)`, which can be calculated using standard tricks you can find in any implementation of the `logsumexp` function in your favorite language. Repeated multiplications of small (large) numbers become repeated additions of moderately sized numbers and the problem becomes stable.

Unfortunately, MATLAB has no graceful way of handling the ensuing operation `sum(A(i,:).' + logB(:,j))` using built-ins. Either you run everything in a big double `for`-loop, which takes forever even with CPU parallelism, or you try to use vectorized broadcasting tricks with `permute`, which mismanages memory and crashes the computation.

This repository seeks to provide such an operation with fast speed and good memory management. The CPU implementation uses [GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS) and the GPU implementation uses [cuASR](https://github.com/hpcgarage/cuASR.git). Since this is often an operation that I would seek to apply during computation of optimization error metrics, I may try to expand this code to include a `JAX`-able Python implementation. No promises though.
