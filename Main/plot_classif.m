function plot_classif(y_true, y_pred)
    
    figure;
    cm = confusionmat(y_true, y_pred);
    confusionchart(cm);
    title('Confusion Matrix');
    disp('Confusion Matrix:');
    disp(cm);
end
