function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

delta = 1e-5;
for i=1:size(theta)
    theta(i) = theta(i) + delta;
    J_plus = J(theta);
    theta(i) = theta(i) - 2*delta;
    J_minus = J(theta);
    numgrad(i) = (J_plus - J_minus) / (2 * delta);    
    theta(i) = theta(i) + delta;
    
    if mod(i, 100) == 0
        fprintf('Finished %d of %d\n', i, size(theta, 1));
        fflush(stdout);
    end
end

%% ---------------------------------------------------------------
end
