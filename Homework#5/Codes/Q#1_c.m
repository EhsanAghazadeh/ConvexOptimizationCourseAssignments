run('team_data.m')

A1 = sparse(1:m, train(:, 1), train(:, 3), m, n);
A2 = sparse(1:m, train(:, 2), -train(:, 3), m, n);
A = A1 + A2;

cvx_begin
 variable a_hat(n)
 minimize(-sum(log_normcdf(A * a_hat/sigma)))
 subject to
  a_hat >= 0
  a_hat <= 1
cvx_end

A1_test = sparse(1:m_test, test(:,1), 1, m_test, n);
A2_test = sparse(1:m_test, test(:,2), -1, m_test, n);
A_test = A1_test + A2_test;

preds = sign(A_test * a_hat)
accuracy_1 = 1 - length(find(preds - test(:, 3))) / m_test;
accuracy_2 = 1 - length(find(train(:, 3) - test(:, 3))) / m_test;

disp('Estimating method accuracy is: ')
disp(accuracy_1 * 100)

disp('Simple method accuracy is: ')
disp(accuracy_2 * 100)
