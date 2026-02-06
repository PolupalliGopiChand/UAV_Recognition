clc; clear; close all;

rng(1); % Reproducibility

%% ====================== GLOBAL PARAMETERS ===============================
fs = 4000;
SNR_dB = 10;
N_samples = 100;
SNR_sweep = -10:2:20;

%% ============================ DATASET ===================================
X = [];
y = [];

%% ============================ BLOCK 1–5 (DATASET CREATION) ===============
for n = 1:N_samples

    %% BLOCK 1: Signal Generation (randomized & cluttered)
    [s_uav, s_bird, t] = B1_generate_signals(fs, SNR_dB);

    %% BLOCK 2: Chirp Optimization
    [~, x_uav, ~, ~, ~]  = B2_chirp_optimizer(s_uav,  fs);
    [~, x_bird, ~, ~, ~] = B2_chirp_optimizer(s_bird, fs);

    %% BLOCK 3: Doppler Enhancement
    [~, S_uav, fd_uav, t_uav]    = B3_doppler_enhancer(x_uav,  fs);
    [~, S_bird, fd_bird, t_bird]= B3_doppler_enhancer(x_bird, fs);

    %% BLOCK 4: TF Signature
    TF_uav  = B4_TF_signature_generator(S_uav,  fd_uav,  t_uav);
    TF_bird = B4_TF_signature_generator(S_bird, fd_bird, t_bird);

    %% BLOCK 5: Feature Extraction
    X = [X;
         B5_feature_extractor(TF_uav);
         B5_feature_extractor(TF_bird)];
    y = [y; 1; 0]; % UAV=1, Bird=0
end

fprintf('Dataset created: %d samples × %d features\n', size(X,1), size(X,2));

%% ============================ FEATURE NORMALIZATION ======================
[X, mu, sigma] = zscore(X);   %#ok<ASGLU>

%% ============================ BLOCK 6 ===================================
% k-Fold Cross-Validation
K = 5;
cv = cvpartition(y, 'KFold', K);
acc = zeros(K,1);

for i = 1:K
    Xtrain = X(training(cv,i),:);
    ytrain = y(training(cv,i));
    Xtest  = X(test(cv,i),:);
    ytest  = y(test(cv,i));

    svmModel = fitcsvm(Xtrain, ytrain, ...
        'KernelFunction','rbf', ...
        'OptimizeHyperparameters','auto', ...
        'HyperparameterOptimizationOptions', ...
        struct('ShowPlots',false,'Verbose',0));

    ypred = predict(svmModel, Xtest);
    acc(i) = mean(ypred == ytest);
end

fprintf('5-Fold CV Accuracy = %.2f %% ± %.2f %%\n', ...
        mean(acc)*100, std(acc)*100);

%% ============================ FINAL MODEL ================================
svmFinal = fitcsvm(X, y, ...
    'KernelFunction','rbf', ...
    'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('ShowPlots',false,'Verbose',0));

%% ============================ BLOCK 7 ===================================
% Confusion Matrix on full data (illustrative)
y_pred = predict(svmFinal, X);
C = confusionmat(y, y_pred);

TP = C(2,2); TN = C(1,1);
FP = C(1,2); FN = C(2,1);

accuracy  = (TP+TN)/sum(C(:));
precision = TP/(TP+FP+eps);
recall    = TP/(TP+FN+eps);
F1        = 2*(precision*recall)/(precision+recall+eps);

fprintf('\n===== FINAL PERFORMANCE =====\n');
fprintf('Accuracy  : %.2f %%\n', accuracy*100);
fprintf('Precision : %.2f\n', precision);
fprintf('Recall    : %.2f\n', recall);
fprintf('F1 Score  : %.2f\n', F1);
disp('Confusion Matrix:'); disp(C);

%% ========================= Visualization ================================

% Feature Distribution Visualization
figure;
feature_names = {'BW','Entropy','RidgeMean','RidgeStd','Peaks','RidgeCurv','Asymmetry'};
for i = 1:size(X,2)
    subplot(3,3,i);
    histogram(X(y==1,i),20,'Normalization','pdf'); hold on;
    histogram(X(y==0,i),20,'Normalization','pdf');
    title(feature_names{i});
    legend('UAV','Bird');
    grid on;
end
sgtitle('Feature Distributions (UAV vs Bird)');

%PCA Feature Space Visualization
[coeff,score,~,~,explained] = pca(X);
figure;
gscatter(score(:,1), score(:,2), y, 'rb', 'ox');
xlabel(['PC1 (',num2str(explained(1),2),'%)']);
ylabel(['PC2 (',num2str(explained(2),2),'%)']);
title('PCA Projection of Feature Space');
grid on;

% Confusion Matrix Heatmap
figure;
confusionchart(C, {'Bird','UAV'});
title('Confusion Matrix');

% Doppler Ridge Comparison
figure;
subplot(2,1,1);
plot(TF_uav.t_stft, TF_uav.tf_ridge,'LineWidth',1.5);
title('UAV Micro-Doppler Ridge');
xlabel('Time (s)'); ylabel('Doppler (Hz)');
grid on;
subplot(2,1,2);
plot(TF_bird.t_stft, TF_bird.tf_ridge,'LineWidth',1.5);
title('Bird Micro-Doppler Ridge');
xlabel('Time (s)'); ylabel('Doppler (Hz)');
grid on;

% Feature Importance
baseAcc = mean(predict(svmFinal,X)==y);
importance = zeros(1,size(X,2));

for i = 1:size(X,2)
    Xp = X;
    Xp(:,i) = Xp(randperm(size(X,1)),i);
    importance(i) = baseAcc - mean(predict(svmFinal,Xp)==y);
end

figure;
bar(importance);
xticklabels(feature_names);
ylabel('Accuracy Drop');
title('Feature Importance (Permutation Method)');
grid on;


% Learning Curve
train_sizes = 20:20:200;
acc_lc = zeros(size(train_sizes));

for k = 1:length(train_sizes)
    idx = randperm(length(y), train_sizes(k));
    model = fitcsvm(X(idx,:), y(idx),'KernelFunction','rbf','Standardize',true);
    acc_lc(k) = mean(predict(model,Xtest)==ytest);
end

figure;
plot(train_sizes, acc_lc,'-o','LineWidth',2);
xlabel('Training Samples');
ylabel('Accuracy');
title('Learning Curve');
grid on;

% ROC CURVE
[~, score] = predict(svmFinal, X);
[Xroc, Yroc, ~, AUC] = perfcurve(y, score(:,2), 1);

figure;
plot(Xroc, Yroc, 'LineWidth', 2);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC,2), ')']);
grid on;

% SNR ROBUSTNESS
acc_snr = zeros(size(SNR_sweep));

for k = 1:length(SNR_sweep)
    Xt = []; yt = [];
    for n = 1:30
        [su, sb, ~] = B1_generate_signals(fs, SNR_sweep(k));
        [~, xu] = B2_chirp_optimizer(su, fs);
        [~, xb] = B2_chirp_optimizer(sb, fs);
        [~, Su, fdu, tu] = B3_doppler_enhancer(xu, fs);
        [~, Sb, fdb, tb] = B3_doppler_enhancer(xb, fs);
        Xt = [Xt;
              B5_feature_extractor(B4_TF_signature_generator(Su,fdu,tu));
              B5_feature_extractor(B4_TF_signature_generator(Sb,fdb,tb))];
        yt = [yt;1;0];
    end
    Xt = (Xt - mu) ./ sigma;
    acc_snr(k) = mean(predict(svmFinal,Xt)==yt);
end

figure;
plot(SNR_sweep, acc_snr,'-o','LineWidth',2);
xlabel('SNR (dB)'); ylabel('Accuracy');
title('Robustness vs SNR');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ============================ FUNCTIONS ================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% -------- B1 --------
function [s_uav, s_bird, t] = B1_generate_signals(fs, SNR_dB)

T = 1; t = 0:1/fs:T-1/fs;
amp = 1.0;

alpha_uav  = 2e4*(0.8+0.4*rand);
alpha_bird = 1000*(0.8+0.4*rand);

uav_freqs  = (200:20:400)*(0.9+0.2*rand);
bird_freqs = (2:2:26)*(0.9+0.2*rand);

uav_phase = 0;
for k = 1:length(uav_freqs)
    uav_phase = uav_phase + (amp/k)*sin(2*pi*(uav_freqs(k)*t + 0.5*alpha_uav*t.^2));
end
s_uav = exp(1j*uav_phase);

bird_phase = 0;
for k = 1:length(bird_freqs)
    bird_phase = bird_phase + (amp/k)*sin(2*pi*(bird_freqs(k)*t + 0.5*alpha_bird*t.^2));
end
s_bird = exp(1j*bird_phase);

% Low-frequency clutter
clutter = filter(1,[1 -0.95],randn(size(t)));
s_uav  = s_uav  + 0.1*clutter;
s_bird = s_bird + 0.1*clutter;

s_uav  = awgn(s_uav,  SNR_dB,'measured');
s_bird = awgn(s_bird, SNR_dB,'measured');
end

%% -------- B2 --------
function [t, x_alpha, alpha_opt, alpha_vec, H2] = B2_chirp_optimizer(x, fs)

t = (0:length(x)-1)/fs;
alpha_vec = linspace(-5e4,5e4,64);
win = hamming(512);
H2 = zeros(length(alpha_vec),1);

for i = 1:length(alpha_vec)
    xd = x.*exp(-1j*pi*alpha_vec(i)*t.^2);
    S = abs(spectrogram(xd,win,384,1024,fs)).^2;
    P = S/sum(S(:))+eps;
    H2(i) = -log(sum(P(:).^2));
end

[~,idx] = min(H2);
alpha_opt = alpha_vec(idx);
x_alpha = x.*exp(-1j*pi*alpha_opt*t.^2);
end

%% -------- B3 --------
function [S_orig,S_enh,fd,t] = B3_doppler_enhancer(x,fs)

[X,fd,t] = spectrogram(x,hamming(256),192,1024,fs);
S_orig = abs(X).^2;

X(abs(fd)<0.05*fs,:) = 0;
noise = median(abs(X).^2,2);
X = X./sqrt(noise+eps);

X = conv2(abs(X),fspecial('gaussian',[11 1],2),'same');
[U,S,V] = svd(X,'econ');
S_enh = max(X - U(:,1)*S(1,1)*V(:,1)',0);
end

%% -------- B4 --------
function TF = B4_TF_signature_generator(S,fd,t)

S = S./(sum(S,1)+eps);
dop = sum(S,2);

[~,idx] = max(S,[],1);
TF.tf_ridge = fd(idx);

p = dop/sum(dop);
TF.spectral_entropy = -sum(p.*log(p+eps));

cdf = cumsum(dop)/max(cumsum(dop));
TF.doppler_bw = fd(find(cdf>0.95,1)) - fd(find(cdf>0.05,1));

TF.doppler_marginal = dop;
TF.fd = fd; TF.t_stft = t; TF.S_normalized = S;
end

%% -------- B5 --------
function feat = B5_feature_extractor(TF)

f1 = TF.doppler_bw;
f2 = TF.spectral_entropy;
f3 = mean(TF.tf_ridge);
f4 = std(TF.tf_ridge);

[pks,~] = findpeaks(TF.doppler_marginal,...
    'MinPeakHeight',0.1*max(TF.doppler_marginal));
f5 = length(pks);

f6 = mean(abs(diff(TF.tf_ridge)));
mid = floor(length(TF.doppler_marginal)/2);
f7 = sum(TF.doppler_marginal(mid:end)) / sum(TF.doppler_marginal(1:mid));

feat = [f1 f2 f3 f4 f5 f6 f7];
end
