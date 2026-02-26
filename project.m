clc; clear; close all;

%% ----- PARAMETERS -----
fs = 4000;                                          % sampling frequency
SNR_dB = 0 + 10*rand;                               % random noise level
SNR_sweep = 0:3:30;                                 % snr range for testing
N_samples = 500;                                    % number of samples per a class

%% =========================== DATASET CREATION ===========================
X = [];                                             % feature matrix
y = [];                                             % labels (1 = UAV, 0 = Bird)

for n = 1:N_samples
    % BLOCK 1
    [s_uav_blk1, s_bird_blk1, alpha_uav, alpha_bird, t_blk1] = B1_generate_signals(fs, SNR_dB);

    % BLOCK 2
    [t_blk2, x_alpha_uav,  alpha_opt_uav,  alpha_vec_uav,  H2_uav ] = B2_chirp_optimizer(s_uav_blk1,  fs);
    [t,      x_alpha_bird, alpha_opt_bird, alpha_vec_bird, H2_bird] = B2_chirp_optimizer(s_bird_blk1, fs);

    % BLOCK 3
    [S_orig_uav,  S_enh_uav,  fd_uav,  t_stft_uav,  snr_before_uav,  snr_after_uav]  = B3_doppler_enhancer(x_alpha_uav,  fs);
    [S_orig_bird, S_enh_bird, fd_bird, t_stft_bird, snr_before_bird, snr_after_bird] = B3_doppler_enhancer(x_alpha_bird, fs);

    % BLOCK 4
    TF_sig_uav  = B4_TF_signature_generator(S_enh_uav,  fd_uav,  t_stft_uav);
    TF_sig_bird = B4_TF_signature_generator(S_enh_bird, fd_bird, t_stft_bird);

    % BLOCK 5
    feat_uav_n  = B5_feature_extractor(TF_sig_uav);
    feat_bird_n = B5_feature_extractor(TF_sig_bird);
    X = [X; feat_uav_n; feat_bird_n];
    y = [y; 1; 0];
end

%% ----- BLOCK 6 -----
K = 5;                                              % number of folds for cross-validation
cv = cvpartition(y,'KFold',K);                      % split data into K folds
acc = zeros(K,1);                                   % store accuracy for each fold

for i = 1:K
    % SPLIT DATA INTO TRAINING & TESTING SETS
    Xtrain = X(training(cv,i),:);
    ytrain = y(training(cv,i));
    Xtest  = X(test(cv,i),:);
    ytest  = y(test(cv,i));

    % NORMALIZE FEATURES USING TRAINED DATA STATISTICS
    mu_fold = mean(Xtrain);
    sigma_fold = std(Xtrain);
    Xtrain = (Xtrain - mu_fold) ./ (sigma_fold + eps);
    Xtest  = (Xtest  - mu_fold) ./ (sigma_fold + eps);

    % TRAIN SVM CLASSIFIER WITH RBF KERNEL
    svmModel = fitcsvm(Xtrain,ytrain,'KernelFunction','rbf','Standardize',false);

    % TEST MODEL & STORE ACCURACY
    acc(i) = mean(predict(svmModel,Xtest)==ytest);
end

% NORMALIZE ENTIRE DATASET
mu = mean(X);
sigma = std(X);
X_norm = (X - mu) ./ (sigma + eps);

% TRAIN FINAL SVM MODEL USING ALL DATA
svmFinal = fitcsvm(X_norm, y,...
        'KernelFunction','rbf',...
        'BoxConstraint',1,...
        'KernelScale','auto');


%% ----- BLOCK 7 -----
acc_snr = zeros(length(SNR_sweep),1);               % store accuracy for each SNR value

for k = 1:length(SNR_sweep)

    X_temp = [];                                    % temporary feature matrix
    y_temp = [];                                    % temporary labels

    for n = 1:50
        % BLOCK 1
        [s_uav_blk1, s_bird_blk1, ~, ~, ~] = B1_generate_signals(fs, SNR_sweep(k));
        
        % BLOCK 2
        [~, x_alpha_uav, ~, ~, ~]  = B2_chirp_optimizer(s_uav_blk1, fs);
        [~, x_alpha_bird, ~, ~, ~] = B2_chirp_optimizer(s_bird_blk1, fs);

        % BLOCK 3
        [S1,~,fd,t_stft] = B3_doppler_enhancer(x_alpha_uav,fs);
        [S2,~,~,~]       = B3_doppler_enhancer(x_alpha_bird,fs);

        % BLOCK 4
        TF1 = B4_TF_signature_generator(S1,fd,t_stft);
        TF2 = B4_TF_signature_generator(S2,fd,t_stft);

        % BLOCK 5
        X_temp = [X_temp; B5_feature_extractor(TF1); B5_feature_extractor(TF2)];
        y_temp = [y_temp;1;0];
    end

    % SPLIT DATA INTO TRAINING & TESTING SETS (70% / 30%)
    cv_snr = cvpartition(y_temp,'HoldOut',0.3);
    Xtrain = X_temp(training(cv_snr),:);
    ytrain = y_temp(training(cv_snr));
    Xtest  = X_temp(test(cv_snr),:);
    ytest  = y_temp(test(cv_snr));
    
    % NORMALIZE FEATURES USING TRAINED DATA STATISTICS
    mu_snr = mean(Xtrain);
    sigma_snr = std(Xtrain);
    Xtrain = (Xtrain-mu_snr)./(sigma_snr+eps);
    Xtest  = (Xtest-mu_snr)./(sigma_snr+eps);
    
    % TRAIN SVM CLASSIFIER
    model_snr = fitcsvm(Xtrain,ytrain,'KernelFunction','rbf');

    % COMPUTE CLASSIFICATION ACCURACY AT THIS SNR
    acc_snr(k) = mean(predict(model_snr,Xtest)==ytest)*100;
end

% FINAL PERFORMANCE EVALUATION
X_eval = (X - mu) ./ (sigma + eps);                 % normalize full dataset using previously computed statistics
[~, score_final] = predict(svmFinal, X_eval);       % get prediction scores for ROC analysis
[~, ~, ~, AUC] = perfcurve(y, score_final(:,2), 1); % compute area under curve (AUC)

% EXTRACT CONFUSION MATRIX VALUES
y_pred = predict(svmFinal, X_eval);                 % predict labels for confusion matrix
C = confusionmat(y, y_pred);                        % build confusion matrix
TP = C(2,2);                                        % correctly detected UAV
TN = C(1,1);                                        % correctly detected Bird
FP = C(1,2);                                        % bird classified as UAV
FN = C(2,1);                                        % UAV classified as bird

% COMPUTE PERFORMANCE METRICS
accuracy  = (TP+TN)/sum(C(:));                      % overall accuracy
precision = TP/(TP+FP+eps);                         % detection precision
recall    = TP/(TP+FN+eps);                         % detection sensitivity
F1 = 2*(precision*recall)/(precision+recall+eps);   % F1 balance score

%% ============================ OUTPUT VALUES =============================
%% ----- BLOCK 1 TO 4 OUTPUT VALUES TABLE -----

Parameter = [ ...
    "Actual Chirp Rate (Hz/s)"; ...
    "Estimated Chirp Rate (Hz/s)"; ...
    "SNR Before (dB)"; ...
    "SNR After (dB)"; ...
    "SNR Gain (dB)"; ...
    "Spectral Entropy"; ...
    "Doppler Bandwidth (Hz)"; ...
    "Doppler Marginal Energy" ];

UAV_values = [ ...
    alpha_uav; ...
    alpha_opt_uav; ...
    snr_before_uav; ...
    snr_after_uav; ...
    snr_after_uav - snr_before_uav; ...
    TF_sig_uav.spectral_entropy; ...
    TF_sig_uav.doppler_bw; ...
    TF_sig_uav.doppler_energy ];

Bird_values = [ ...
    alpha_bird; ...
    alpha_opt_bird; ...
    snr_before_bird; ...
    snr_after_bird; ...
    snr_after_bird - snr_before_bird; ...
    TF_sig_bird.spectral_entropy; ...
    TF_sig_bird.doppler_bw; ...
    TF_sig_bird.doppler_energy ];

SignalTable = table(Parameter, round(UAV_values,3), round(Bird_values,3), ...
                    'VariableNames', {'Parameter','UAV','Bird'});

disp('BLOCK 1 TO 4 OUTPUT VALUES TABLE')
disp(SignalTable)

%% ----- BLOCK 5 OUTPUT VALUES TABLE -----

FeatureNames = [ ...
    "Spectral Entropy"; ...
    "Doppler Bandwidth (Hz)"; ...
    "Peak Count"; ...
    "Skewness"; ...
    "Ridge Mean (Hz)"; ...
    "Ridge STD (Hz)"; ...
    "Ridge Curvature (Hz)" ];

FeatureTable = table(FeatureNames, round(feat_uav_n(:),3), round(feat_bird_n(:),3), ...
                     'VariableNames', {'Feature','UAV','Bird'});

disp('BLOCK 5 OUTPUT VALUES TABLE')
disp(FeatureTable)

%% ----- BLOCK 6 AND 7 OUTPUT VALUES TABLE -----

Metric = [ ...
    "Mean CV Accuracy (%)"; ...
    "Final Accuracy (%)"; ...
    "Precision"; ...
    "Recall"; ...
    "F1 Score"; ...
    "AUC" ];

MetricValues = [ ...
    mean(acc)*100; ...
    accuracy*100; ...
    precision; ...
    recall; ...
    F1; ...
    AUC ];

PerformanceTable = table(Metric, round(MetricValues,4), ...
                         'VariableNames', {'Metric','Value'});

disp('BLOCK 6 AND 7 OUTPUT VALUES TABLE')
disp(PerformanceTable)

%% ============================= OUTPUT PLOTS =============================
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

sgtitle('ECHO SIGNALS GENERATION');

%% -------- FIGURE 2 --------
figure;

subplot(3,2,1);
plot(alpha_vec_uav, H2_uav,'LineWidth',2); hold on;
xline(alpha_opt_uav,'--r','LineWidth',1.5);
xlabel('Chirp Rate (Hz/s)');
ylabel('Rényi Entropy');
title('Entropy vs Chirp Rate (UAV)');
grid on;

subplot(3,2,2);
plot(alpha_vec_bird, H2_bird,'LineWidth',2); hold on;
xline(alpha_opt_bird,'--r','LineWidth',1.5);
xlabel('Chirp Rate (Hz/s)');
ylabel('Rényi Entropy');
title('Entropy vs Chirp Rate (Bird)');
grid on;

subplot(3,1,2);
plot(t_blk2, real(x_alpha_uav),'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('Amplitude (V)');
title('De-chirped UAV Signal');

subplot(3,1,3);
plot(t_blk2, real(x_alpha_bird),'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('Amplitude (V)');
title('De-chirped Bird Signal');

sgtitle('CHIRP RATE OPTIMIZATION');

%% -------- FIGURE 3 --------
figure;

subplot(2,2,1);
imagesc(t_stft_uav, fd_uav, 10*log10(S_orig_uav+1e-6));
axis xy; h = colorbar;
ylabel(h,'Power (dB)');
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Original UAV Spectrogram');

subplot(2,2,2);
imagesc(t_stft_bird, fd_bird, 10*log10(S_orig_bird+1e-6));
axis xy; h = colorbar;
ylabel(h,'Power (dB)');
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Original Bird Spectrogram');

subplot(2,2,3);
imagesc(t_stft_uav, fd_uav, 10*log10(S_enh_uav+1e-6));
axis xy; h = colorbar;
ylabel(h,'Power (dB)');
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Enhanced UAV Spectrogram');

subplot(2,2,4);
imagesc(t_stft_bird, fd_bird, 10*log10(S_enh_bird+1e-6));
axis xy; h = colorbar;
ylabel(h,'Power (dB)');
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Enhanced Bird Spectrogram');

sgtitle('DOPPLER ENHANCEMENT');

%% -------- FIGURE 4 --------
figure;

subplot(3,2,1);
imagesc(TF_sig_uav.t_stft, TF_sig_uav.fd, ...
    10*log10(TF_sig_uav.S_normalized+1e-6));
axis xy; h = colorbar;
ylabel(h,'Power (dB)');
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Normalized UAV Spectrogram');

subplot(3,2,2);
imagesc(TF_sig_bird.t_stft, TF_sig_bird.fd, ...
    10*log10(TF_sig_bird.S_normalized+1e-6));
axis xy; h = colorbar;
ylabel(h,'Power (dB)');
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

sgtitle('Time-Frequency Signatures');
sgtitle('TIME-FREQUENCY SIGNATURE GENERATION');

%% -------- FIGURE 5 --------
figure;

feature_names = { ...
    'Spectral Entropy', ...
    'Doppler Bandwidth', ...
    'Peak Count', ...
    'Asymmetry', ...
    'Ridge Mean', ...
    'Ridge STD', ...
    'Ridge Curvature' };

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

sgtitle('FEATURE EXTACTION');

%% -------- FIGURE 6 --------
figure;

[~,score,~,~,explained] = pca(X);
h = gscatter(score(:,1), score(:,2), y, ...
    [0.2 0.6 0.9; 0.9 0.4 0.2], 'ox', 8);
set(h(1), 'DisplayName', 'Bird');
set(h(2), 'DisplayName', 'UAV');
legend('Location','best');
xlabel(['PC1  (',num2str(explained(1),3),'%)']);
ylabel(['PC2  (',num2str(explained(2),3),'%)']);
title('LOW-DIMENSIONAL FEATURE SPACE (PCA)');
grid on; box on;

%% -------- FIGURE 7 --------
figure;

cv_hold = cvpartition(y,'HoldOut',0.3);
Xtrain_full = X(training(cv_hold),:);
ytrain_full = y(training(cv_hold));
Xval = X(test(cv_hold),:);
yval = y(test(cv_hold));
Ntrain = length(ytrain_full);

train_sizes = round(linspace(20, Ntrain-10, 10));
acc_lc = zeros(size(train_sizes));

for k = 1:length(train_sizes)
    n_samples = train_sizes(k);
    idx = randperm(Ntrain, n_samples);
    Xsub = Xtrain_full(idx,:);
    ysub = ytrain_full(idx);
    mu_lc = mean(Xsub);
    sigma_lc = std(Xsub);
    Xsub = (Xsub - mu_lc)./(sigma_lc + eps);
    Xval_n = (Xval - mu_lc)./(sigma_lc + eps);
    model = fitcsvm(Xsub, ysub,'KernelFunction','rbf');
    acc_lc(k) = mean(predict(model,Xval_n)==yval)*100;
end

plot(train_sizes, acc_lc,'-o','LineWidth',2);
xlabel('Training Samples');
ylabel('Validation Accuracy (%)');
title('TRUE LEARNING CURVE');
grid on;

%% -------- FIGURE 8 --------
figure;
cm1 = confusionchart(C, {'Bird','UAV'});
cm1.Title = 'CONFUSION MATRIX';
cm1.GridVisible = 'on';

%% -------- FIGURE 9 --------
figure;

X_eval = (X - mu) ./ (sigma + eps);
[~, score] = predict(svmFinal, X_eval);

[Xroc, Yroc, ~, ~] = perfcurve(y, score(:,2), 1);

plot(Xroc, Yroc,'LineWidth',2); hold on;
plot([0 1],[0 1],'k--','LineWidth',1.5);

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC CURVE  |  AUC = ', num2str(AUC,3)]);
grid on; box on;

%% -------- FIGURE 10 --------
figure;

plot(SNR_sweep, acc_snr,'-o','LineWidth',2,'MarkerSize',6);
xlabel('SNR (dB)');
ylabel('Classification Accuracy (%)');
ylim([0 100]);
title('ROBUSTNESS OF UAV RECOGNITION VS NOISE');
grid minor; box on;

%% ============================== FUNCTIONS ===============================
%% ----- BLOCK 1 -----
function [s_uav, s_bird, alpha_uav, alpha_bird, t] = B1_generate_signals(fs, SNR_dB)
    % PARAMETERS
    T = 1;                                          % signal duration (seconds)
    t = 0:1/fs:T-1/fs;                              % time vector
    amp = 0.8 + 0.4*rand;                           % random amplitude
    uav_freqs  = 200 + 20*(0:10) + randn(1,11)*3;   % ~200–400 Hz harmonics (uav rotor blade)
    bird_freqs = 2 + 2*(0:49) + randn(1,50)*0.5;    % ~2–100 Hz harmonics (bird wing flap)
    alpha_uav  = 20000 + 4000*randn;                % 20k Hz (high frequency chirp rate)
    alpha_bird = 1000  + 200*randn;                 % 1k Hz (low frequency chirp rate)
    
    % UAV SIGNAL GENERATION
    s_uav = zeros(size(t));                         % uav signal initialization
    for k = 1:length(uav_freqs)
        s_uav = s_uav + (amp/k) * exp(1j*2*pi*(uav_freqs(k)*t + 0.5*alpha_uav*t.^2)); 
    end
    s_uav = awgn(s_uav,SNR_dB,'measured');          % add noise to uav signal

    % BIRD SIGNAL GENERATION    
    s_bird = zeros(size(t));                        % bird signal initialization
    for k = 1:length(bird_freqs)
        s_bird = s_bird + (amp/k) * exp(1j*2*pi*(bird_freqs(k)*t + 0.5*alpha_bird*t.^2)); 
    end
    s_bird = awgn(s_bird,SNR_dB,'measured');        % add noise to uav signal
end

%% ----- BLOCK 2 -----
function [t,x_alpha,alpha_opt,alpha_vec,H2] = B2_chirp_optimizer(x,fs)
    % PARAMETERS
    t = (0:length(x)-1)/fs;                         % time vector
    alpha_vec = linspace(-3e4,3e4,100);             % chirp rate range
    win = hamming(256);                             % spectrogram window
    
    % DE-CHIRP THE ECHO SIGNAL
    H2 = zeros(length(alpha_vec),1);                % entropy initialization
    for i = 1:length(alpha_vec)
        xd = x .* exp(-1j*pi*alpha_vec(i)*t.^2);    % de-chirp the signal
        S = abs(spectrogram(xd,win,192,512,fs)).^2; % compute spectrogram energy
        P = S / (sum(S(:)) + eps);                  % normalize energy distribution
        H2(i) = -log(sum(P(:).^2));                 % compute Rényi entropy
    end
    [~, idx] = min(H2); alpha_opt = alpha_vec(idx); % select min chirp rate                     
    x_alpha = x .* exp(-1j*pi*alpha_opt*t.^2);      % de-chirped signal
end

%% ----- BLOCK 3 -----
function [S_orig,S_enh,fd,t_stft,snr_before,snr_after] = B3_doppler_enhancer(x,fs)
    % SPECTROGRAM GENERATION
    [X,fd,t_stft] = spectrogram(x,hamming(256),192,1024,fs);
    S_orig = abs(X).^2;                             % original power spectrogram

    % BODY CLUTTER REMOVAL
    fc = 0.05*fs;                                   % cutoff frequency to remove body component
    X(abs(fd)<fc,:) = 0;                            % suppress low-frequency body motion

    % NOISE REMOVAL
    noise = median(abs(X).^2,2);                    % estimate noise
    X = X./sqrt(noise+eps);                         % normalize spectrum using noise level
    
    % GAUSSIAN SMOOTHING
    X = conv2(abs(X),fspecial('gaussian',[11 1],2),'same');
    
    % BACKGROUND SUPPRESSION
    X_mean = mean(X,2); S_enh = max(X - X_mean,0);

    % SNR IMPROVEMENT MEASUREMENT
    body = abs(fd)<fc; harm = abs(fd)>=fc;          % define body and harmonic regions
    snr_before = 10*log10 ...                       % compute SNR before enhancement
        (mean(S_orig(harm,:),'all')/mean(S_orig(body,:),'all'));
    snr_after  = 10*log10 ...                       % compute SNR after enhancement
        (mean(S_enh(harm,:),'all')/mean(S_enh(body,:),'all'));
end

%% ----- BLOCK 4 -----
function TF = B4_TF_signature_generator(S,fd,t)
    % NORMALIZE SPECTROGRAM COLUMN-WISE
    S = S ./ (sum(S,1) + eps);

    % ALLOW POSITIVE DOPPLER FREQUENCIES ONLY
    mask = fd > 0;
    S_pos = S(mask,:);
    TF.S_normalized = S_pos;                        % store normalized spectrogram
    fd_pos = fd(mask);
    TF.fd = fd_pos;                                 % store positive Doppler frequency vector

    % COMPUTE DOPPLER MARGINAL
    dop = sum(S_pos,2);
    p = dop / (sum(dop) + eps);
    TF.doppler_marginal = p;                        % store normalized Doppler marginal
    TF.doppler_energy = sum(dop);                   % store doppler marginal

    % COMPUTE SPECTRAL ENTROPY
    TF.spectral_entropy = -sum(p .* log(p + eps));  % store spectral entropy

    % COMPUTE DOPPLER BANDWIDTH
    cdf = cumsum(p);                                % compute cdf of Doppler energy
    idx_low  = find(cdf > 0.05,1,'first');          % find lower cutoff frequency
    idx_high = find(cdf > 0.95,1,'first');          % find higher cutoff frequency
    if isempty(idx_low) || isempty(idx_high)
        TF.doppler_bw = 0;
    else
        TF.doppler_bw = fd_pos(idx_high) - fd_pos(idx_low);
    end

    % COMPUTE RIDGE VALUES
    [~,idx] = max(S_pos,[],1);
    TF.tf_ridge = fd_pos(idx);                      % store extracted ridge value
    TF.t_stft = t;                                  % store time axis of spectrogram
end

%% ----- BLOCK 5 -----
function feat = B5_feature_extractor(TF)
    % DOPPLER ENERGY DISTRIBUTION
    dop = TF.doppler_marginal(:);
    fd  = TF.fd(:);

    % NORMALIZE ENERGY DISTRIBUTION
    dop = dop / (sum(dop) + eps);
    
    % PEAK DETECTION
    peak_count = 0;                                 % peak count initialization
    if ~isempty(dop) && any(dop > 0)
        max_val = max(dop);
        if max_val > 0
            % detect peaks with prominence level
            prominence_level = 0.05 * max_val;
            [pks,~] = findpeaks(dop, ...
                'MinPeakProminence', prominence_level, ...
                'MinPeakDistance', 10);
            peak_count = numel(pks);
        end
    end
    
    % DOPPLER STATISTICS
    mean_fd = sum(fd .* dop);                       % mean of doppler frequency
    std_fd  = sqrt(sum(((fd - mean_fd).^2).*dop));  % spread of doppler frequencies
    if std_fd > 1e-12                               % skewness of doppler frequencies
        skew_fd = sum(((fd - mean_fd).^3).*dop)/(std_fd^3);
    else
        skew_fd = 0;
    end
    
    % RIGDE STATISTICS
    ridge_mean = mean(TF.tf_ridge);                 % mean of ridge
    ridge_std  = std(TF.tf_ridge);                  % standard deviation of ridge
    ridge_curv = mean(abs(diff(TF.tf_ridge)));      % ridge curvature

    % FEATURE VECTOR
    feat = [ ...
        TF.spectral_entropy, ...                    % SPECTRAL ENTROPY
        TF.doppler_bw, ...                          % DOPPLER BANDWIDTH
        peak_count, ...                             % PEAK COUNT
        skew_fd, ...                                % SKEWNESS
        ridge_mean, ...                             % RIDGE MEAN
        ridge_std, ...                              % RIGDE STANDARD DEVIATION
        ridge_curv ...                              % RIGDE CURVATURE
        ];
end
