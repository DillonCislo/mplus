# mplus
A MATLAB/mex implementation of the multiplication operator in the 'sum.plus' tropical algebra.

From GraphBLAS: MATLAB can only compute `C=A*B` using the standard `'+.*.double'` and `'+.*.complex'` semirings. A semiring is defined in terms of a string, `'add.mult.type'`, where `'add'` is a monoid that takes the place of the additive operator, `'mult'` is the multiplicative operator, and `'type'` is the data type for the two inputs to the `mult` operator.

In the standard semiring, `C=A*B` is defined as:

    C(i,j) = sum (A(i,:).' .* B(:,j))


