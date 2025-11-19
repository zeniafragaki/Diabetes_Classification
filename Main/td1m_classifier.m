%----Machine Learning (ML) model for deciding dibetes or not----------
%
%----Author : @ Zenia Fragaki-----------------------------------------

---------------------------
function td1m_classifier

    
    % 1. Load and Process Data
    fprintf('<strong>[Step 1] Loading and Processing Data...</strong>\n');
    data = readtable('diabetes_dataset.xlsx'); 
    
    % Encode Gender
    data.gender = string(data.gender);
    data.gender(data.gender=='Female')=1;
    data.gender(data.gender=='Male')=0;
    data.gender = double(data.gender);

    % Encode Smoking
    data.smoking_history=string(data.smoking_history);
    data.smoking_history(data.smoking_history=='never')=-1; 
    data.smoking_history(data.smoking_history=='No Info')=0;
    data.smoking_history(data.smoking_history=='former')=1;
    data.smoking_history(data.smoking_history=='not current')=1.5;
    data.smoking_history(data.smoking_history=='current')=2;
    data.smoking_history(data.smoking_history=='ever')=3;
    data.smoking_history=double(data.smoking_history);

    data_n = table2array(data);

    % Normalize (Min-Max scaling to -1 to 1)
    maxdata = max(data_n);
    mindata = min(data_n);
    norm_data = zeros(size(data_n));
    
    for k=1:size(data_n,2)
        if (maxdata(k) - mindata(k)) ~= 0
            norm_data(:,k) = 2 * (data_n(:,k) - mindata(k)) / (maxdata(k) - mindata(k)) - 1;
        else
            norm_data(:,k) = data_n(:,k);
        end
    end

    % 2. Split Data (70% Train, 15% Val, 15% Test)
    rng(1); % Reproducibility
    N = size(norm_data,1);
    rand_dist = randperm(N);
    
    n_train = round(0.70 * N);
    n_val   = round(0.15 * N);
    
    train_idx = rand_dist(1 : n_train);
    val_idx   = rand_dist(n_train + 1 : n_train + n_val);
    test_idx  = rand_dist(n_train + n_val + 1 : end);
    
    X_train = norm_data(train_idx, 1:end-1);
    y_train = norm_data(train_idx, end);
    X_val   = norm_data(val_idx, 1:end-1);
    y_val   = norm_data(val_idx, end);
    X_test  = norm_data(test_idx, 1:end-1);
    y_test  = norm_data(test_idx, end);

    % Calculate Imbalance Weights
    num_neg = sum(y_train == -1); 
    num_pos = sum(y_train == 1);
    base_ratio = num_neg / num_pos;
    balance_factor = 0.6; 
    class_weight = base_ratio * balance_factor;
    
    fprintf('Healthy: %d, Diabetic: %d. Imbalance Ratio: %.2f. Target Weight: %.2f\n', ...
        num_neg, num_pos, base_ratio, class_weight);

    %---------------------------------------------------------------------
    % MODEL 1: SVM (Support Vector Machine)
    %---------------------------------------------------------------------
    fprintf('\n<strong>[Model 1] Training SVM...</strong>\n');
    box_constraint = ones(size(y_train));
    box_constraint(y_train == 1) = class_weight;
    
    Mdl_SVM = fitcsvm(X_train, y_train, 'Weights', box_constraint, 'KernelFunction', 'rbf');
    
    [label_svm, ~] = predict(Mdl_SVM, X_test);
    stats_svm = evaluation(y_test, label_svm, 'SVM');

    %---------------------------------------------------------------------
    % MODEL 2: Random Forest (TreeBagger)
    %---------------------------------------------------------------------
    fprintf('\n<strong>[Model 2] Training Random Forest...</strong>\n');
    % Random Forest handles weights via Cost Matrix
    % Cost(i,j) is cost of classifying class i as class j
    % [0, 1; W, 0] means misclassifying a diabetic (row 2) costs W times more
    cost_matrix = [0, 1; class_weight, 0]; 
    
    % Train 50 Trees
    Mdl_RF = TreeBagger(50, X_train, y_train, ...
        'Method', 'classification', ...
        'OOBPrediction', 'on', ...
        'Cost', cost_matrix);
        
    label_rf_str = predict(Mdl_RF, X_test);
    label_rf = str2double(label_rf_str); % Convert string output back to numbers
    stats_rf = evaluation(y_test, label_rf, 'Random Forest');

    %---------------------------------------------------------------------
    % MODEL 3: MLP (Neural Network)
    %---------------------------------------------------------------------
    fprintf('\n<strong>[Model 3] Training MLP Neural Network...</strong>\n');
    % fitcnet is available in R2021a+. If error, use patternnet (older).
    
    Mdl_MLP = fitcnet(X_train, y_train, ...
        'LayerSizes', [20, 10], ... % Two hidden layers
        'Weights', box_constraint, ...
        'Standardize', true);
        
    label_mlp = predict(Mdl_MLP, X_test);
    stats_mlp = evaluation(y_test, label_mlp, 'MLP Neural Net');

    %---------------------------------------------------------------------
    % MODEL 4: RBF Network with K-Means (Hybrid)
    %---------------------------------------------------------------------
    fprintf('\n<strong>[Model 4] Training RBF Network (K-Means)...</strong>\n');
    % 1. Find Centers using K-Means
    K = 50; % Number of RBF centers
    [idx, C] = kmeans(X_train, K, 'MaxIter', 100);
    
    % 2. Calculate Sigma (Spread) based on average distance
    dists = pdist(C);
    sigma = mean(dists); 
    gamma = 1 / (2 * sigma^2);
    
    % 3. Transform Data to RBF Feature Space (Phi Matrix)
    Phi_train = rbf_transform(X_train, C, gamma);
    Phi_test  = rbf_transform(X_test, C, gamma);
    
    % 4. Train Linear Output Layer (Weighted Linear SVM on features)
    % We use the transformed features (Phi) to train a linear classifier
    Mdl_RBF_Out = fitcsvm(Phi_train, y_train, ...
        'Weights', box_constraint, ...
        'KernelFunction', 'linear'); 
        
    [label_rbf, ~] = predict(Mdl_RBF_Out, Phi_test);
    stats_rbf = evaluation(y_test, label_rbf, 'RBF w/ K-Means');

    %---------------------------------------------------------------------
    % FINAL COMPARISON
    %---------------------------------------------------------------------
    fprintf('\n<strong>================ FINAL COMPARISON ================</strong>\n');
    fprintf('%-15s | %-10s | %-10s | %-10s\n', 'Model', 'Accuracy', 'Recall', 'F1-Score');
    fprintf('----------------------------------------------------------\n');
    print_row('SVM', stats_svm);
    print_row('Random Forest', stats_rf);
    print_row('MLP (Neural)', stats_mlp);
    print_row('RBF (K-Means)', stats_rbf);
    fprintf('==========================================================\n');
    
    % Plot all confusion matrices in one figure
    figure('Name', 'Benchmark Results', 'NumberTitle', 'off');
    subplot(2,2,1); confusionchart(confusionmat(y_test, label_svm)); title('SVM');
    subplot(2,2,2); confusionchart(confusionmat(y_test, label_rf)); title('Random Forest');
    subplot(2,2,3); confusionchart(confusionmat(y_test, label_mlp)); title('MLP');
    subplot(2,2,4); confusionchart(confusionmat(y_test, label_rbf)); title('RBF K-Means');
end

%-----------------------HELPER FUNCTIONS-----------------------------

function Phi = rbf_transform(X, Centers, gamma)
    % Manual RBF Transform: Gaussian function of distance to centers
    N = size(X, 1);
    K = size(Centers, 1);
    Phi = zeros(N, K);
    
    for i = 1:K
        % Euclidean distance squared between all X and Center i
        diff = X - Centers(i,:);
        dist_sq = sum(diff.^2, 2);
        Phi(:, i) = exp(-gamma * dist_sq);
    end
end

function stats = evaluation(y_true, y_pred, name)
    accuracy = sum(y_true == y_pred) / length(y_true);
    
    TP = sum(y_true == 1 & y_pred == 1);
    FP = sum(y_true == -1 & y_pred == 1);
    FN = sum(y_true == 1 & y_pred == -1);
    
    sensitivity = TP / (TP + FN); 
    if (TP+FP) == 0; precision = 0; else; precision = TP / (TP + FP); end
    if (precision + sensitivity) == 0
        f1 = 0;
    else
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity);
    end
    
    stats.acc = accuracy;
    stats.rec = sensitivity;
    stats.f1 = f1;
    

    fprintf(' -> %s Done. F1: %.4f (Recall: %.2f%%)\n', name, f1, sensitivity*100);
end

function print_row(name, stats)
    fprintf('%-15s | %-9.2f%% | %-9.2f%% | %.4f\n', ...
        name, stats.acc*100, stats.rec*100, stats.f1);
end
