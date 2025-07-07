function [accuracy, R2, rmse] = evaluation(y_true, y_pred)
    %  Accuracy
    accuracy = sum(y_true == y_pred) / length(y_true);

    % R^2
    SS_res = sum((y_true - y_pred).^2);
    SS_tot = sum((y_true - mean(y_true)).^2);
    R2 = 1 - (SS_res / SS_tot);

    %  RMSE
    rmse = sqrt(mean((y_true - y_pred).^2));

    % print
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
    fprintf('R^2 Score: %.4f\n', R2);
    fprintf('RMSE: %.4f\n', rmse);
end
