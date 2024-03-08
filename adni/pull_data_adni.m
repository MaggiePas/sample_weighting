clear;
rng(0);

noisestd = 5; % perturbation noise
Knn=30; % graph construction parameters

%% load relevant data
info = readtable('adni_processed_2_28.csv','delimiter',',');
filtered_data = info(ismember(info.DX, {'MCI', 'Dementia'}), :);
filtered_data.PTGENDER = double(strcmp(filtered_data.PTGENDER, 'Male')); % strcmp will return 1 for 'Male', 0 otherwise

% Rename 'RID' column to 'subjects'
filtered_data.Properties.VariableNames{'subject'} = 'subjects';

% Find the first occurrence of each subject
[~, uniqueIndex] = unique(filtered_data.subjects, 'stable');

% Create a new table with only the first visit for each subject
firstVisits = filtered_data(uniqueIndex, :);

info = firstVisits;
subjects = unique(info.subjects);
subjects = num2cell(subjects);

N = length(subjects);

%% extract features
sex = info.PTGENDER;
age = info.AGE_visit;
apoe4 = info.APOE4;

feature = [sex,age,apoe4];
feature_unperturbed = [sex,age,apoe4];

for i = 1:size(feature,2)
    feature(isnan(feature(:,i)),i) = nanmean(feature(:,i)); % fill in missing values
    if i == 1
        feature(:,i) = zscore(feature(:,i)) + randn(N,1)/noisestd; % normalize
    elseif i>1
        feature(:,i) = zscore(feature(:,i)) + randn(N,1)/10; % normalize
    end
end

%% construct graph
w = zeros(N,N);
for i = 1:N
    for j = 1:N
        w(i,j) = 1/(sum((feature(i,:) - feature(j,:)).^2)+1);
    end

    sw = sort(w(i,:),'descend');
    w(i, w(i,:) < sw(Knn)) = 0; 
    w(i,i)=0;
end
w = (w + w')/2;

%% extract/visualize eigenbasis
d = diag(sum(w));
l = d - w;
[V,D] = eig(l);

subplot(1,5,1);
scatter3(feature(:,1),feature(:,2),feature(:,3),50*ones(N,1),V(:,2),'filled');
xlabel('sex','FontSize',20)
ylabel('age','FontSize',20)
zlabel('APOE4','FontSize',20)
subplot(1,5,2);
scatter3(feature(:,1),feature(:,2),feature(:,3),50*ones(N,1),V(:,3),'filled');
xlabel('sex','FontSize',20)
ylabel('age','FontSize',20)
zlabel('APOE4','FontSize',20)
subplot(1,5,3);
scatter3(feature(:,1),feature(:,2),feature(:,3),50*ones(N,1),V(:,4),'filled');
xlabel('sex','FontSize',20)
ylabel('age','FontSize',20)
zlabel('APOE4','FontSize',20)
subplot(1,5,4);
scatter3(feature(:,1),feature(:,2),feature(:,3),50*ones(N,1),V(:,5),'filled');
xlabel('sex','FontSize',20)
ylabel('age','FontSize',20)
zlabel('APOE4','FontSize',20)
subplot(1,5,5);
scatter3(feature(:,1),feature(:,2),feature(:,3),50*ones(N,1),V(:,6),'filled');
xlabel('sex','FontSize',20)
ylabel('age','FontSize',20)
zlabel('APOE4','FontSize',20)

%% save eigenbasis/values
results = [cell2table(subjects),array2table(V(:,2:end))];
writetable(results,'./adni/graph_factors/eigenvectors_adni_lower_pert_knn_ad_mci_30.csv');
d = diag(D); d = d(2:end);
save('./adni/graph_factors/eigenvalue_adni_lower_pert_knn_ad_mci_30.txt','d','-ascii');
writematrix(feature,'./adni/graph_factors/graph_factors_adni_pert_knn_ad_mci_30.csv');