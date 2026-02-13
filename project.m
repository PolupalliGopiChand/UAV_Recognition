clc; clear; close all;

%% =========================== GLOBAL PARAMETERS ==========================
fs = 4000;
SNR_dB = 10;
SNR_sweep = -10:1:20;
N_samples = 100;
N_display = N_samples;

%% ========================== DATASET CREATION ==========================

X = [];
y = [];

for n = 1:N_samples
    flagPlot = (n == N_display);

    %% -------- BLOCK 1 --------
    [s_uav, s_bird, t] = B1_generate_signals(fs, SNR_dB);

    if flagPlot
        t_blk1 = t;
        s_uav_blk1  = s_uav;
        s_bird_blk1 = s_bird;
    end

    %% -------- BLOCK 2 --------
    [t_u, x_uav,  alpha_opt_uav,  alpha_vec_uav,  H2_uav ] = ...
        B2_chirp_optimizer(s_uav,  fs);
    [t_b, x_bird, alpha_opt_bird, alpha_vec_bird, H2_bird] = ...
        B2_chirp_optimizer(s_bird, fs);

    if flagPlot
        t_blk2 = t_u;
        x_alpha_uav = x_uav;
        x_alpha_bird = x_bird;

        fprintf('\n===== BLOCK-2 CHIRP ESTIMATION =====\n');
        fprintf('UAV Actual Chirp Rate     = 20000 Hz/s\n');
        fprintf('UAV Estimated Chirp Rate  = %.2f Hz/s\n', alpha_opt_uav);
        fprintf('Bird Actual Chirp Rate    = 1000 Hz/s\n');
        fprintf('Bird Estimated Chirp Rate = %.2f Hz/s\n\n', alpha_opt_bird);
    end

    %% -------- BLOCK 3 --------
    [S_orig_uav,  S_enh_uav,  fd_uav,  t_stft_uav, ...
        snr_before_uav, snr_after_uav] = B3_doppler_enhancer(x_uav, fs);

    [S_orig_bird, S_enh_bird, fd_bird, t_stft_bird, ...
        snr_before_bird, snr_after_bird] = B3_doppler_enhancer(x_bird, fs);

    if flagPlot
        fprintf('===== BLOCK-3 HARMONIC SNR =====\n');
        fprintf('UAV  SNR Before = %.2f dB\n', snr_before_uav);
        fprintf('UAV  SNR After  = %.2f dB\n', snr_after_uav);
        fprintf('Bird SNR Before = %.2f dB\n', snr_before_bird);
        fprintf('Bird SNR After  = %.2f dB\n\n', snr_after_bird);
    end

    %% -------- BLOCK 4 --------
    TF_sig_uav  = B4_TF_signature_generator(S_enh_uav,  fd_uav,  t_stft_uav);
    TF_sig_bird = B4_TF_signature_generator(S_enh_bird, fd_bird, t_stft_bird);

    if flagPlot
        fprintf('\n===== BLOCK 4: TIME–FREQUENCY SIGNATURE (SAMPLE %d) =====\n', n);
        fprintf('Spectral Entropy      = %.4f\n', TF_sig_uav.spectral_entropy);
        fprintf('Doppler Bandwidth     = %.2f Hz\n', TF_sig_uav.doppler_bw);
        fprintf('Ridge Mean Doppler    = %.2f Hz\n', mean(TF_sig_uav.tf_ridge));
        fprintf('Ridge Std Doppler     = %.2f Hz\n', std(TF_sig_uav.tf_ridge));
        fprintf('Marginal Energy (sum) = %.4f\n', sum(TF_sig_uav.doppler_marginal));
    end


    %% -------- BLOCK 5 --------
    X = [X;
         B5_feature_extractor(TF_sig_uav);
         B5_feature_extractor(TF_sig_bird)];
    y = [y; 1; 0];

    if flagPlot
        feat_uav_n  = B5_feature_extractor(TF_sig_uav);
        feat_bird_n = B5_feature_extractor(TF_sig_bird);
        
        fprintf('\n===== BLOCK 5: EXTRACTED FEATURES (ONE SAMPLE) =====\n');
        feature_names = {...
            'Spectral Entropy', ...
            'Peak Count', ...
            'Ridge Mean', ...
            'Doppler BW', ...
            'Ridge Std', ...
            'Ridge Curvature', ...
            'Asymmetry'};
    
        for i = 1:length(feat_uav_n)
            fprintf('UAV  %s = %.4f\n', feature_names{i}, feat_uav_n(i));
            fprintf('Bird %s = %.4f\n', feature_names{i}, feat_bird_n(i));
        end
    end
end

fprintf('Dataset created: %d samples × %d features\n', size(X,1), size(X,2));
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

    acc(i) = mean(predict(svmModel,Xtest)==ytest);
end

svmFinal = fitcsvm(X, y, ...
    'KernelFunction','rbf', ...
    'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('ShowPlots',false,'Verbose',0));

fprintf('\n===== BLOCK 6: MODEL PERFORMANCE (CV) =====\n');
fprintf('Mean CV Accuracy = %.2f %%\n', mean(acc)*100);
fprintf('\n');


%% ==============================  BLOCK 7 ================================
acc_snr = zeros(size(SNR_sweep));

for k = 1:length(SNR_sweep)
    Xt = [];
    yt = [];

    for n = 1:30
        [su, sb, ~] = B1_generate_signals(fs, SNR_sweep(k));

        [~, xu] = B2_chirp_optimizer(su, fs);
        [~, xb] = B2_chirp_optimizer(sb, fs);

        [~, Su, fdu, tu, ~, ~] = B3_doppler_enhancer(xu, fs);
        [~, Sb, fdb, tb, ~, ~] = B3_doppler_enhancer(xb, fs);

        Xt = [Xt;
              B5_feature_extractor(B4_TF_signature_generator(Su,fdu,tu));
              B5_feature_extractor(B4_TF_signature_generator(Sb,fdb,tb))];

        yt = [yt; 1; 0];
    end

    Xt = (Xt - mu) ./ sigma;
    acc_snr(k) = mean(predict(svmFinal, Xt) == yt) * 100;
end

% Confusion matrix
y_pred = predict(svmFinal, X);
C = confusionmat(y, y_pred);

% ----- COMPUTE METRICS (THIS IS WHAT YOU WERE MISSING) -----
TP = C(2,2); 
TN = C(1,1);
FP = C(1,2); 
FN = C(2,1);

accuracy  = (TP+TN)/sum(C(:));
precision = TP/(TP+FP+eps);
recall    = TP/(TP+FN+eps);
F1        = 2*(precision*recall)/(precision+recall+eps);

fprintf('\n===== BLOCK 7: FINAL METRICS =====\n');
fprintf('Accuracy  = %.2f %%\n', accuracy*100);
fprintf('Precision = %.4f\n', precision);
fprintf('Recall    = %.4f\n', recall);
fprintf('F1 Score  = %.4f\n', F1);
%% ========================== ALL PLOTS (LABELED) ==========================

%% -------- FIGURE 1 --------
figure;
subplot(2,1,1);
plot(t_blk1, real(s_uav_blk1),'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('Amplitude (V)');
title('UAV Echo Signal');

subplot(2,1,2);
plot(t_blk1, real(s_bird_blk1),'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('Amplitude (V)');
title('Bird Echo Signal');

sgtitle('Figure 1: Raw Echo Signals');

%% -------- FIGURE 2 --------
figure;

subplot(3,2,1);
plot(alpha_vec_uav, H2_uav,'LineWidth',2); hold on;
xline(alpha_opt_uav,'--r','LineWidth',1.5);
xlabel('Chirp Rate (Hz/s)');
ylabel('Rényi Entropy');
title('Entropy vs Chirp Rate — UAV');
grid on;

subplot(3,2,2);
plot(alpha_vec_bird, H2_bird,'LineWidth',2); hold on;
xline(alpha_opt_bird,'--r','LineWidth',1.5);
xlabel('Chirp Rate (Hz/s)');
ylabel('Rényi Entropy');
title('Entropy vs Chirp Rate — Bird');
grid on;

subplot(3,1,2);
plot(t_blk2, real(x_alpha_uav),'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('Amplitude (V)');
title('De-chirped UAV Signal');

subplot(3,1,3);
plot(t_blk2, real(x_alpha_bird),'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('Amplitude (V)');
title('De-chirped Bird Signal');

sgtitle('Figure 2: Chirp Optimization');

%% -------- FIGURE 3 --------
figure;

subplot(2,2,1);
imagesc(t_stft_uav, fd_uav, 10*log10(S_orig_uav+1e-6));
axis xy; colorbar;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Original UAV Spectrogram');

subplot(2,2,2);
imagesc(t_stft_bird, fd_bird, 10*log10(S_orig_bird+1e-6));
axis xy; colorbar;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Original Bird Spectrogram');

subplot(2,2,3);
imagesc(t_stft_uav, fd_uav, 10*log10(S_enh_uav+1e-6));
axis xy; colorbar;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Enhanced UAV Spectrogram');

subplot(2,2,4);
imagesc(t_stft_bird, fd_bird, 10*log10(S_enh_bird+1e-6));
axis xy; colorbar;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Enhanced Bird Spectrogram');

sgtitle('Figure 3: Doppler Enhancement');

%% -------- FIGURE 4 --------
figure;

subplot(3,2,1);
imagesc(TF_sig_uav.t_stft, TF_sig_uav.fd, ...
    10*log10(TF_sig_uav.S_normalized+1e-6));
axis xy; colorbar;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Normalized UAV Spectrogram');

subplot(3,2,2);
imagesc(TF_sig_bird.t_stft, TF_sig_bird.fd, ...
    10*log10(TF_sig_bird.S_normalized+1e-6));
axis xy; colorbar;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Normalized Bird Spectrogram');

subplot(3,2,3);
plot(TF_sig_uav.t_stft, TF_sig_uav.tf_ridge,'LineWidth',1.5); grid on;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('UAV Micro-Doppler Ridge');

subplot(3,2,4);
plot(TF_sig_bird.t_stft, TF_sig_bird.tf_ridge,'LineWidth',1.5); grid on;
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Bird Micro-Doppler Ridge');

subplot(3,2,5);
plot(TF_sig_uav.fd, TF_sig_uav.doppler_marginal,'LineWidth',1.5); grid on;
xlabel('Doppler Frequency (Hz)');
ylabel('Normalized Energy');
title('UAV Doppler Marginal');

subplot(3,2,6);
plot(TF_sig_bird.fd, TF_sig_bird.doppler_marginal,'LineWidth',1.5); grid on;
xlabel('Doppler Frequency (Hz)');
ylabel('Normalized Energy');
title('Bird Doppler Marginal');

sgtitle('Figure 4: Time-Frequency Signatures');

%% -------- FIGURE 5 --------
figure;

feature_names = { 'Spectral Entropy', 'Peak Count', 'Ridge Mean', 'Doppler BW', 'Ridge Std', 'Ridge Curvature', 'Asymmetry' };

for i = 1:size(X,2)
    if (i >= 1 && i <= 4)
        subplot(3,2,i);
    else
        subplot(3,3,i+2);
    end
    histogram(X(y==1,i),25,'Normalization','pdf'); hold on;
    histogram(X(y==0,i),25,'Normalization','pdf');
    grid minor;
    xlabel('Normalized Feature Value');
    ylabel('Probability Density');
    title(feature_names{i});
    legend({'UAV','Bird'});
end

sgtitle('Figure 5: Feature Separability');

%% -------- FIGURE 6 --------
figure;

[~,score,~,~,explained] = pca(X);

h = gscatter(score(:,1), score(:,2), y, ...
    [0.2 0.6 0.9; 0.9 0.4 0.2], 'ox', 8);

% Fix legend names
set(h(1), 'DisplayName', 'Bird');   % y = 0
set(h(2), 'DisplayName', 'UAV');    % y = 1
legend('Location','best');

xlabel(['PC1  (',num2str(explained(1),3),'%)']);
ylabel(['PC2  (',num2str(explained(2),3),'%)']);
title('Low-Dimensional Feature Space (PCA)');
grid on; box on;


%% -------- FIGURE 7 --------
figure;

train_sizes = 0:10:2*N_display;
acc_lc = zeros(size(train_sizes));

for k = 2:length(train_sizes)
    idx = randperm(length(y), train_sizes(k));
    model = fitcsvm(X(idx,:), y(idx),'KernelFunction','rbf');
    acc_lc(k) = mean(predict(model,X)==y)*100;
end

plot(train_sizes, acc_lc,'-o','LineWidth',2,'MarkerSize',6);
xlabel('Number of Training Samples');
ylabel('Classification Accuracy (%)');
ylim([0 100]);
title('Model Learning Behavior');
grid minor; box on;

%% -------- FIGURE 8 --------
figure;
cm1 = confusionchart(C, {'Bird','UAV'});
cm1.Title = 'Confusion Matrix';
cm1.GridVisible = 'on';

%% -------- FIGURE 9 (ROC) --------
figure;

[~, score] = predict(svmFinal, X);
[Xroc, Yroc, ~, AUC] = perfcurve(y, score(:,2), 1);

plot(Xroc, Yroc,'LineWidth',2); hold on;
plot([0 1],[0 1],'k--','LineWidth',1.5);

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve  |  AUC = ', num2str(AUC,3)]);
grid on; box on;

%% -------- FIGURE 10 --------
figure;

plot(SNR_sweep, acc_snr,'-o','LineWidth',2,'MarkerSize',6);
xlabel('SNR (dB)');
ylabel('Classification Accuracy (%)');
ylim([0 100]);
title('Robustness of UAV Recognition vs Noise');
grid minor; box on;

%% ========================== FUNCTIONS ===================================

%% ---- B1 ----
function [s_uav, s_bird, t] = B1_generate_signals(fs, SNR_dB)
    T = 1; t = 0:1/fs:T-1/fs; amp = 1;
    uav_freqs = 200:20:400;  bird_freqs = 2:2:26;
    alpha_uav = 2e4; alpha_bird = 1000;
    
    uav_phase = zeros(size(t));
    for k = 1:length(uav_freqs)
        uav_phase = uav_phase + (amp/k)*sin(2*pi*(uav_freqs(k)*t + 0.5*alpha_uav*t.^2));
    end
    s_uav = exp(1j*2*pi*0.5*uav_phase);
    s_uav = awgn(s_uav,SNR_dB,'measured');
    
    bird_phase = zeros(size(t));
    for k = 1:length(bird_freqs)
        bird_phase = bird_phase + (amp/k)*sin(2*pi*(bird_freqs(k)*t + 0.5*alpha_bird*t.^2));
    end
    s_bird = exp(1j*2*pi*0.5*bird_phase);
    s_bird = awgn(s_bird,SNR_dB,'measured');
end

%% ---- B2 ----
function [t,x_alpha,alpha_opt,alpha_vec,H2] = B2_chirp_optimizer(x,fs)
    t = (0:length(x)-1)/fs;
    alpha_vec = linspace(-5e4,5e4,64);
    win = hamming(512); H2=zeros(length(alpha_vec),1);
    
    for i=1:length(alpha_vec)
        xd = x.*exp(-1j*pi*alpha_vec(i)*t.^2);
        S = abs(spectrogram(xd,win,384,1024,fs)).^2;
        P = S/sum(S(:))+eps;
        H2(i) = -log(sum(P(:).^2));
    end
    [~,idx] = min(H2);
    alpha_opt = alpha_vec(idx);
    x_alpha = x.*exp(-1j*pi*alpha_opt*t.^2);
end

%% ---- B3 ----
function [S_orig,S_enh,fd,t_stft,snr_before,snr_after] = B3_doppler_enhancer(x,fs)
    
    [X,fd,t_stft] = spectrogram(x,hamming(256),192,1024,fs);
    S_orig = abs(X).^2;
    
    fc = 0.05*fs;
    X(abs(fd)<fc,:) = 0;
    
    noise = median(abs(X).^2,2);
    X = X./sqrt(noise+eps);
    
    X = conv2(abs(X),fspecial('gaussian',[11 1],2),'same');
    
    [U,S,V] = svd(X,'econ');
    S_enh = max(X - U(:,1)*S(1,1)*V(:,1)',0);
    
    body = abs(fd)<fc; harm = abs(fd)>=fc;
    snr_before = 10*log10(mean(S_orig(harm,:),'all')/mean(S_orig(body,:),'all'));
    snr_after  = 10*log10(mean(S_enh(harm,:),'all')/mean(S_enh(body,:),'all'));
end

%% ---- B4 ----
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

%% ---- B5 ----
function feat = B5_feature_extractor(TF)
[pks,~] = findpeaks(TF.doppler_marginal, 'MinPeakHeight',0.1*max(TF.doppler_marginal));
feat = [ TF.spectral_entropy, length(pks), mean(TF.tf_ridge), TF.doppler_bw, std(TF.tf_ridge), mean(abs(diff(TF.tf_ridge))), sum(TF.doppler_marginal) ];

end
