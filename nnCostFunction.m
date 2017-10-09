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

% cost function %

y_vectorized = y;

for i = 1:(num_labels - 1)
  y_vectorized = [y_vectorized y];
end


values = 1:num_labels;

y_vectorized = (y_vectorized == values);

X = [ones(m, 1) X];
z_2 = (Theta1 * X');

a_2_mat = sigmoid(z_2);
a_2_mat = [ones(m, 1)'; a_2_mat];

all_h = sigmoid(Theta2 * a_2_mat);

values_mat = -y_vectorized .* log(all_h)' - (1 - y_vectorized) .* log(1 - all_h)';

theta1_reg_mat = Theta1(:, 2:(input_layer_size + 1)) .^ 2;
theta2_reg_mat = Theta2(:, 2:(hidden_layer_size + 1)) .^ 2;

regularize = (lambda / (2 * m)) * (sum(theta1_reg_mat(:)) + sum(theta2_reg_mat(:)));

J = (1 / m) * sum(values_mat(:)) + regularize;

% backpropagation algorithem %

delta_3 = all_h' - y_vectorized;
z_2 = [ones(m, 1)'; z_2];

delta_2 = (Theta2' * delta_3') .* sigmoidGradient(z_2);

Theta1_grad = (1 / m) * (delta_2(2:end, :) * X);
Theta2_grad = (1 / m) * (delta_3' * a_2_mat');

% regularization %

Theta1_grad = [Theta1_grad(:, 1) (Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end))];
Theta2_grad = [Theta2_grad(:, 1) (Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end))];


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
