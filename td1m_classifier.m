%----Machine Learning (ML) model for deciding dibetes or not----------
%
%----Author : @ Zenia Fragaki-----------------------------------------

%Load data

data=readtable('diabetes_dataset1.xlsx');
% data=table2array(data);

%------------Data processing--------------------------

%------Encoding strings to numerical------------------
data.gender = string(data.gender);
data.gender(data.gender=='Female')=1; % Female=1
data.gender(data.gender=='Male')=0;   % Male=0
data.gender = double(data.gender);

data.smoking_history=string(data.smoking_history);
data.smoking_history(data.smoking_history=='never')=-1; 

data.smoking_history(data.smoking_history=='No Info')=0;

data.smoking_history(data.smoking_history=='former')=1;
data.smoking_history(data.smoking_history=='not current')=1.5;

data.smoking_history(data.smoking_history=='current')=2;
data.smoking_history(data.smoking_history=='ever')=3;
data.smoking_history=double(data.smoking_history);
data_n=table2array(data);

% -----Normalization------------------------------------------

maxdata=zeros(1,size(data_n,2));
for k=1:size(data_n,2)
    maxdata(k)=max(data_n(:,k));
end

mindata=zeros(1,size(data_n,2));
for k=1:size(data_n,2)
    mindata(k)=min(data_n(:,k));
end

norm_data=zeros(size(data_n));
for k=1:size(data,2)
    for i=1:size(data,1)
        norm_data(i,k)=2*(data_n(i,k)-mindata(k))/(maxdata(k)-mindata(k))-1;
        % norm_data(i,k)=(data_n(i,k)-mindata(k))/(maxdata(k)-mindata(k));
    end
end
% disp(norm_data)

%----SPLIT: TRAIN - TEST ---------------------------------------
%
%----70%-30%----------------------------------------------------
%
number_of_samples=size(norm_data,1);
rand_dist=randperm(number_of_samples);

train_size=round(0.7*number_of_samples);
train_idx = rand_dist(1:train_size);
test_idx = rand_dist(train_size+1:end);

train_data = norm_data(train_idx, :);
test_data = norm_data(test_idx, :);

X_train = train_data(:, 1:end-1);
y_train = train_data(:, end);

X_test = test_data(:, 1:end-1);
y_test = test_data(:, end);

%---------------------------------------------------------------
X = norm_data(:, 1:end-1); 
Y = norm_data(:, end);     
Mdl = fitcsvm(X, Y);
[label, score] = predict(Mdl, X_test);
