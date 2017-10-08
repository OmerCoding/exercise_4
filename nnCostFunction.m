function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

y_vectorized = y;

for i = 1:(num_labels - 1)
  y_vectorized = [y_vectorized y];
end

values = 1:num_labels;

y_vectorized = (y_vectorized == values);

X = [ones(m, 1) X];

a_2_mat = sigmoid(Theta1 * X');
a_2_mat = [ones(m, 1)'; a_2_mat];

all_h = sigmoid(Theta2 * a_2_mat);

values_mat = -y_vectorized .* log(all_h)' - (1 - y_vectorized) .* log(1 - all_h)';

theta1_reg_mat = Theta1(:, 2:(input_layer_size + 1)) .^ 2;
theta2_reg_mat = Theta2(:, 2:(hidden_layer_size + 1)) .^ 2;

regularize = (lambda / (2 * m)) * (sum(theta1_reg_mat(:)) + sum(theta2_reg_mat(:)));


J = (1 / m) * sum(values_mat(:)) + regularize;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
