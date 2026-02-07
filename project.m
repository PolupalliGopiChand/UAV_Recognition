clc; clear; close all;

%% =========================== GLOBAL PARAMETERS ==========================
fs = 4000;
SNR_dB = 10;
SNR_sweep = -10:2:20;
N_samples = 100;
N_display = N_samples;

%% ========================== (DATASET CREATION) ==========================

X = [];
y = [];

for n = 1:N_samples
    flagPlot = (n == N_display);

    %% BLOCK 1: Signal Generation
    [s_uav, s_bird, t] = B1_generate_signals(fs, SNR_dB);

    if flagPlot
        t_blk1 = t;
        s_uav_blk1  = s_uav;
        s_bird_blk1 = s_bird;
    end

    %% BLOCK 2: Chirp Optimization
    [t_u, x_uav,  alpha_opt_uav,  alpha_vec_uav,  H2_uav ] = B2_chirp_optimizer(s_uav,  fs);
    [t_b, x_bird, alpha_opt_bird, alpha_vec_bird, H2_bird] = B2_chirp_optimizer(s_bird, fs);

    if flagPlot
        t_blk2 = t_u;
        x_alpha_uav = x_uav;
        x_alpha_bird = x_bird;
    end

    %% BLOCK 3: Doppler Enhancement
    [S_orig_uav,  S_enh_uav,  fd_uav,  t_stft_uav ] = B3_doppler_enhancer(x_uav,  fs);
    [S_orig_bird, S_enh_bird, fd_bird, t_stft_bird] = B3_doppler_enhancer(x_bird, fs);

    %% BLOCK 4: TF Signature
    TF_sig_uav  = B4_TF_signature_generator(S_enh_uav,  fd_uav,  t_stft_uav);
    TF_sig_bird = B4_TF_signature_generator(S_enh_bird, fd_bird, t_stft_bird);

    %% BLOCK 5: Feature Extraction
    X = [X; B5_feature_extractor(TF_sig_uav); B5_feature_extractor(TF_sig_bird)];
    y = [y; 1; 0];

    if flagPlot
        feat_uav_n  = B5_feature_extractor(TF_sig_uav);
        feat_bird_n = B5_feature_extractor(TF_sig_bird);
    end
end

fprintf('Dataset created: %d samples × %d features\n', size(X,1), size(X,2));

% === SHOW Nth SAMPLE FEATURES ===
disp('================= Nth SAMPLE FEATURES =================');
disp(['Sample index = ', num2str(N_display)]);
disp('UAV Features:');  disp(feat_uav_n);
disp('Bird Features:'); disp(feat_bird_n);

% === FEATURE NORMALIZATION ===
[X, mu, sigma] = zscore(X);

%% =============================== BLOCK 6 ================================
   
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

% === FINAL MODEL ===
svmFinal = fitcsvm(X, y, ...
    'KernelFunction','rbf', ...
    'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('ShowPlots',false,'Verbose',0));

%% =============================== BLOCK 7 ================================

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

%% ================================ OUTPUTS ===============================

%% BLOCK-1 PLOT
figure;
% UAV Echo signal 
subplot(2,1,1); plot(t_blk1,  real(s_uav_blk1));
xlabel('Time (s)'); ylabel('Amplitude (v)');
title('UAV Echo Signal');  grid on;
% Bird Echo signal 
subplot(2,1,2); plot(t_blk1, real(s_bird_blk1));
xlabel('Time (s)'); ylabel('Amplitude (v)');
title('Bird Echo Signal'); grid on;

%% BLOCK-2 PLOT
figure;
% UAV Entropy vs Chirp Rate
subplot(3,2,1); plot(alpha_vec_uav,  H2_uav,  'LineWidth', 2);
xline(alpha_opt_uav,  '--r', 'Optimal UAV \alpha',  'LineWidth', 2);
xlabel('Chirp Rate (Hz/s)'); ylabel('Renyi Entropy');
title('Entropy vs Chirp Rate for UAV');  grid on;
% Bird Entropy vs Chirp Rate
subplot(3,2,2); plot(alpha_vec_bird, H2_bird, 'LineWidth', 2);
xline(alpha_opt_bird, '--r', 'Optimal Bird \alpha', 'LineWidth', 2);
xlabel('Chirp Rate (Hz/s)'); ylabel('Renyi Entropy');
title('Entropy vs Chirp Rate for Bird'); grid on;
% UAV De-Chirped signal
subplot(3,1,2); plot(t_blk2, real(x_alpha_uav));
xlabel('Time (s)'); ylabel('Amplitude (v)');
title('Dechirped UAV Signal');  grid on;
% Bird De-Chirped signal
subplot(3,1,3); plot(t_blk2, real(x_alpha_bird));
xlabel('Time (s)'); ylabel('Amplitude (v)');
title('Dechirped Bird Signal'); grid on;

%% BLOCK-3 PLOT
figure;
% Original UAV Spectrogram
subplot(2,2,1); imagesc(t_stft_uav,  fd_uav,  10*log10(S_orig_uav+1e-6));  axis xy; c = colorbar; 
ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Original UAV Spectrogram' );
% Original Bird Spectrogram
subplot(2,2,2); imagesc(t_stft_bird, fd_bird, 10*log10(S_orig_bird+1e-6)); axis xy; c = colorbar; 
ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Original Bird Spectrogram');
% Enhanced UAV Spectrogram
subplot(2,2,3); imagesc(t_stft_uav,  fd_uav,  10*log10(S_enh_uav+1e-6));   axis xy; c = colorbar; 
ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Enhanced UAV Spectrogram' );
% Enhanced Bird Spectrogram
subplot(2,2,4); imagesc(t_stft_bird, fd_bird, 10*log10(S_enh_bird+1e-6));  axis xy; c = colorbar; 
ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Enhanced Bird Spectrogram');

%% BLOCK-4 PLOT
figure;
% Normalized UAV Spectrogram
subplot(3,2,1); imagesc(TF_sig_uav.t_stft,  TF_sig_uav.fd,  10*log10(TF_sig_uav.S_normalized  + 1e-6));
axis xy; c = colorbar; ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Normalized UAV Spectrogram' );
% Normalized Bird Spectrogram
subplot(3,2,2); imagesc(TF_sig_bird.t_stft, TF_sig_bird.fd, 10*log10(TF_sig_bird.S_normalized + 1e-6));
axis xy; c = colorbar; ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Normalized Bird Spectrogram');
% UAV Micro-Doppler Ridge
subplot(3,2,3); plot(TF_sig_uav.t_stft,  TF_sig_uav.tf_ridge,  'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('UAV Micro-Doppler Ridge' ); grid on;
% Bird Micro-Doppler Ridge
subplot(3,2,4); plot(TF_sig_bird.t_stft, TF_sig_bird.tf_ridge, 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Bird Micro-Doppler Ridge'); grid on;
% UAV Doppler Marginal
subplot(3,2,5); plot(TF_sig_uav.fd,  TF_sig_uav.doppler_marginal,  'LineWidth', 1.5);
xlabel('Doppler Frequency (Hz)'); ylabel('Normalized Energy');
title('UAV Doppler Marginal'); grid on;
% Bird Doppler Marginal
subplot(3,2,6); plot(TF_sig_bird.fd, TF_sig_bird.doppler_marginal, 'LineWidth', 1.5);
xlabel('Doppler Frequency (Hz)'); ylabel('Normalized Energy');
title('Bird Doppler Marginal'); grid on;

%% BLOCK-5 PLOT
figure;
feature_names = {'Doppler BW','Spectral Entropy','Ridge Mean',...
                 'Ridge Std','Peak Count','Ridge Curvature','Asymmetry'};

for i = 1:size(X,2)
    subplot(3,3,i);
    h1 = histogram(X(y==1,i),25,'Normalization','pdf'); hold on;
    h2 = histogram(X(y==0,i),25,'Normalization','pdf');

    h1.FaceColor = [0.2 0.6 0.9]; % blue-ish for UAV
    h2.FaceColor = [0.9 0.4 0.2]; % orange for Bird
    h1.FaceAlpha = 0.6;
    h2.FaceAlpha = 0.6;

    title(feature_names{i},'FontWeight','bold');
    xlabel('Normalized Value');
    ylabel('Probability Density');
    legend({'UAV','Bird'},'Location','best');
    grid minor;
end

sgtitle('Feature Separability: UAV vs Bird','FontSize',14,'FontWeight','bold');
%% BLOCK-6 PLOT
figure;
% PCA 
[coeff,score,~,~,explained] = pca(X);

gscatter(score(:,1), score(:,2), y, ...
         [0.2 0.6 0.9; 0.9 0.4 0.2], 'ox', 8);

hold on;
xlabel(['PC1  (',num2str(explained(1),3),'%)']);
ylabel(['PC2  (',num2str(explained(2),3),'%)']);
title('Low-Dimensional Feature Space');
legend({'Bird','UAV'},'Location','best');
grid on; box on;
% Learning Curve
train_sizes = 0:20:200;
acc_lc = zeros(size(train_sizes));

for k = 2:length(train_sizes)
    idx = randperm(length(y), train_sizes(k));
    model = fitcsvm(X(idx,:), y(idx),...
        'KernelFunction','rbf','Standardize',true);
    acc_lc(k) = mean(predict(model,X)==y)*100;
end

figure;
plot(train_sizes, acc_lc,'-o','LineWidth',2,'MarkerSize',6);
xlabel('Training Samples');
ylabel('Accuracy (%)');
ylim([0 100]);
title('Model Learning Behavior');
grid minor; box on;
%% BLOCK-7 PLOT
figure;
%Confusion Matrix
cm = confusionchart(C, {'Bird','UAV'});
cm.Title = 'Final Classification Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
cm.FontSize = 11;
cm.GridVisible = 'on';
figure;
% ROC Curve
[~, score] = predict(svmFinal, X);
[Xroc, Yroc, ~, AUC] = perfcurve(y, score(:,2), 1);

plot(Xroc, Yroc, 'LineWidth', 2, 'Color', [0.1 0.4 0.8]); hold on;
plot([0 1],[0 1],'k--','LineWidth',1.5);

xlabel('False Positive Rate'); 
ylabel('True Positive Rate');
title(['ROC Curve  |  AUC = ', num2str(AUC,3)]);
grid on; box on;
acc_snr = zeros(size(SNR_sweep));
%SNR Robustness
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
    acc_snr(k) = mean(predict(svmFinal,Xt)==yt)*100;
end

figure;
plot(SNR_sweep, acc_snr,'-o','LineWidth',2,'MarkerSize',6);
xlabel('SNR (dB)'); 
ylabel('Accuracy (%)');
ylim([0 100]);
title('Robustness of UAV Recognition vs Noise');
grid minor; box on;


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
f7 = sum(TF.doppler_marginal(mid:end)) / ...
     sum(TF.doppler_marginal(1:mid));

feat = [f1 f2 f3 f4 f5 f6 f7];
end
