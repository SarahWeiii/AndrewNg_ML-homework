function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% n = size(theta); % number of features
% pro = X*theta;
% tmp1 = 0;
% tmp2 = 0;
% for i=1:m
%     tmp1 = tmp1 + y(i)*log(sigmoid(pro(i))) + (1-y(i))*log(1-sigmoid(pro(i)));
% end
% for i=1:n
%     tmp2 = tmp2 + theta(i)^2;
% end
% J = J - tmp1/m + lambda*tmp2/(2*m);
% for j=1:size(theta)
%     tmp = 0;
%     for i=1:m
%         tmp = tmp + (sigmoid(pro(i)) - y(i))*X(i,j);
%     end
%     if j == 1
%         grad(j) = tmp / m;
%     else
%         grad(j) = tmp / m + lambda * theta(j) / m;
%     end
% end
h = sigmoid(X*theta);
theta1 = theta;
theta1(1) = 0;

J = (-y'*log(h)-(1-y)'*log(1-h)+theta1'*theta1*lambda/2)/m;

grad = X'*(h-y)/m+lambda/m*theta1;



% =============================================================

end
