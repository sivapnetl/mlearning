function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X, 2);
prod = zeros(m, 1);
thetaCur = zeros(size(theta));

fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [ %.0f %.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    H = X * theta;
    diff = H - y;

    for feature = 1:n
      prod = diff .* X( :,  feature);
      thetaCur(feature) = (alpha * (sum(prod, 1)/m));
    end

    theta -= thetaCur;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
fprintf('First 10 examples from the cost values: \n');
fprintf(' x = [ %.0f],  \n', [J_history(1:10,:) ]');

end
