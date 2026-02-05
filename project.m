clc; clear; close all;

%% ============================== BLOCK-1 =================================
% --------------- Radar Echo Signal Generation (UAV / Bird) ---------------

fs = 4000; 
T  = 1; 
t  = 0:1/fs:T-1/fs; 
SNR_dB = 10; 
amp = 1.0; 

uav_prop_freqs  = [200 220 240 260 280 300 320 340 360 380 400]; 
bird_flap_freqs = [2 4 6 8 10 12 14 16 18 20 22 24 26]; 

alpha_uav_true  = 2e4; 
alpha_bird_true = 100; 

% UAV Signal
uav_phase = zeros(size(t));
for k = 1:length(uav_prop_freqs)
    f0 = uav_prop_freqs(k);
    uav_phase = uav_phase + (amp/k) * ...
        sin(2*pi*( f0*t + 0.5*alpha_uav_true*t.^2 ));
end

s_uav = exp(1j * 2*pi*0.5 * uav_phase);
s_uav = awgn(s_uav, SNR_dB, 'measured');

% Bird Signal
bird_phase = zeros(size(t));
for k = 1:length(bird_flap_freqs)
    f0 = bird_flap_freqs(k);
    bird_phase = bird_phase + (amp/k) * ...
        sin(2*pi*( f0*t + 0.5*alpha_bird_true*t.^2 ));
end

s_bird = exp(1j * 2*pi*0.5 * bird_phase);
s_bird = awgn(s_bird, SNR_dB, 'measured');

% Block-1 Output Plot
figure;
% UAV Signal Plot
subplot(2,1,1);
plot(t, real(s_uav));
title('Block-1: UAV Radar Echo (Real Part)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;
% Bird Signal Plot
subplot(2,1,2);
plot(t, real(s_bird));
title('Block-1: Bird Radar Echo (Real Part)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

%% ============================ BLOCK-2 ===================================
% ------------------ Adaptive Chirp-Rate Optimizer ------------------------

x = s_uav;     
t = (0:length(x)-1)/fs;

alpha_min = -5e4;
alpha_max =  5e4;
N_alpha   = 64;
alpha_vec = linspace(alpha_min, alpha_max, N_alpha);

win_len  = 256;
win      = hamming(win_len);
noverlap = round(0.75*win_len);
nfft     = 1024;

H2 = zeros(length(alpha_vec),1);

for i = 1:length(alpha_vec)
    alpha = alpha_vec(i);
    chirp_comp = exp(-1j*pi*alpha*t.^2);
    x_dechirp  = x .* chirp_comp;

    [S,~,~] = spectrogram(x_dechirp, win, noverlap, nfft, fs);
    S_mag = abs(S).^2;

    P = S_mag / sum(S_mag(:)) + eps;
    H2(i) = -log(sum(P(:).^2));
end

[~, idx_opt] = min(H2);
alpha_opt = alpha_vec(idx_opt);

fprintf('\nBlock-2 Output:\n');
fprintf('True UAV Chirp Rate  = %.2f Hz/s\n', alpha_uav_true);
fprintf('Estimated Chirp Rate = %.2f Hz/s\n', alpha_opt);

chirp_comp_opt = exp(-1j*pi*alpha_opt*t.^2);
x_alpha = x .* chirp_comp_opt;

% Block-2 Output Plot
figure;
plot(alpha_vec, H2, 'LineWidth', 2);
xlabel('Chirp Rate (Hz/s)');
ylabel('Renyi Entropy');
title('Block-2: Entropy vs Chirp Rate');
grid on;

%% =============================== BLOCK-3 ================================
% ----------------------- Adaptive Doppler Enhancer -----------------------

PRF = fs;

winLen   = 256;
hop      = 64;
noverlap = winLen - hop;
nfft     = 1024;
win      = hamming(winLen);

[X, fd, t_stft] = spectrogram(x_alpha, win, noverlap, nfft, fs);
S = abs(X).^2;

% Body removal
fc = 0.05 * PRF;
HPmask = abs(fd) >= fc;
X_hp = X .* HPmask;

% Spectral whitening
P_noise = median(abs(X_hp).^2, 2);
eps_val = 1e-6;
X_white = X_hp ./ sqrt(P_noise + eps_val);

% Ridge sharpening
sigma_fd = 2;
h = fspecial('gaussian', [11 1], sigma_fd);
h = h / sum(h);
X_sharp = conv2(X_white, h, 'same');

% SVD background suppression
S_mag = abs(X_sharp);
[U, Ssvd, V] = svd(S_mag, 'econ');
K = 1;
S_bg = U(:,1:K) * Ssvd(1:K,1:K) * V(:,1:K)';
S_enh = max(S_mag - S_bg, 0);

% Metrics
bodyBand = abs(fd) < fc;
harmBand = abs(fd) >= fc;

bodyPower = mean(S(bodyBand,:), 'all');
harmPower = mean(S(harmBand,:), 'all');

bodyPower_enh = mean(S_enh(bodyBand,:), 'all');
harmPower_enh = mean(S_enh(harmBand,:), 'all');

Harmonic_SNR_before = 10*log10(harmPower / bodyPower);
Harmonic_SNR_after  = 10*log10(harmPower_enh / bodyPower_enh);

fprintf('\nBlock-3 Output:\n');
fprintf('Harmonic SNR Before = %.2f dB\n', Harmonic_SNR_before);
fprintf('Harmonic SNR After  = %.2f dB\n', Harmonic_SNR_after);

% Block-3 Output Plot
figure;
% Original Spectrogram
subplot(1,2,1);
imagesc(t_stft, fd, 10*log10(S + eps_val));
axis xy;
shading interp;
colormap(jet);
c1 = colorbar;
ylabel(c1, 'Power (dB)');
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Original Spectrogram');
% Enhanced Spectrogram
subplot(1,2,2);
imagesc(t_stft, fd, 10*log10(S_enh + eps_val));
axis xy;
shading interp;
colormap(jet);
c2 = colorbar;
ylabel(c2, 'Power (dB)');
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Enhanced Spectrogram');

%% =============================== BLOCK-4 ================================
% ------------------ Time–Frequency Signature Generator -------------------

S_norm = S_enh ./ (sum(S_enh,1) + eps_val);

doppler_marginal = sum(S_norm, 2);
time_marginal = sum(S_norm, 1);

[~, ridge_idx] = max(S_norm, [], 1);
tf_ridge = fd(ridge_idx);

cdf_dopp = cumsum(doppler_marginal);
cdf_dopp = cdf_dopp / max(cdf_dopp);

f_low  = fd(find(cdf_dopp >= 0.05, 1, 'first'));
f_high = fd(find(cdf_dopp >= 0.95, 1, 'first'));
doppler_bw = f_high - f_low;

f_min = 30;
harm_mask = abs(fd) > f_min;
harmonic_env = sum(S_norm(harm_mask,:), 1);

tm = time_marginal - mean(time_marginal);
acf = xcorr(tm, 'coeff');
acf = acf(length(tm):end);

ridge_mean = mean(abs(tf_ridge));
ridge_std  = std(tf_ridge);

features = [
    ridge_mean;
    ridge_std;
    doppler_bw;
    mean(harmonic_env);
    std(harmonic_env);
    max(acf(2:end));
    skewness(doppler_marginal);
    kurtosis(doppler_marginal)
];

TF_sig = struct();
TF_sig.features = features(:).';

% Block-4 Output Plot
figure;
% Normalized TF Signature
subplot(2,2,1);
imagesc(t_stft, fd, 10*log10(S_norm + eps_val));
axis xy;
shading interp;
colormap(jet);
c3 = colorbar;
ylabel(c3, 'Normalized Power (dB)');
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Normalized TF Signature');
% Micro-Doppler Ridge
subplot(2,2,2);
plot(t_stft, tf_ridge, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Doppler Frequency (Hz)');
title('Micro-Doppler Ridge');
grid on;
% Doppler Marginal
subplot(2,2,3);
plot(fd, doppler_marginal, 'LineWidth', 1.5);
xlabel('Doppler Frequency (Hz)');
ylabel('Normalized Energy');
title('Doppler Marginal Signature');
grid on;
% Periodicity (ACF)
subplot(2,2,4);
plot(acf, 'LineWidth', 1.5);
xlabel('Lag Index');
ylabel('Autocorrelation');
title('Micro-Motion Periodicity');
grid on;

%% ======================= BLOCK-5: FEATURE EXTRACTOR =======================

% ---------------- A) Z-SCORE NORMALIZATION ----------------
X = TF_sig.features;                 % [1 x 8] feature vector
mu = mean(X,1);
sigma = std(X,0,1) + 1e-9;           % numerical safety
X_norm = (X - mu) ./ sigma;

% Boxplot: before vs after normalization
figure('Name','Feature Normalization');
subplot(1,2,1); boxplot(X); 
title('Raw Features'); ylabel('Feature Value'); grid on;
subplot(1,2,2); boxplot(X_norm);
title('Z-score Normalized Features'); ylabel('Standard Units'); grid on;

% ---------------- C) PHYSICS-AWARE FEATURE AUGMENTATION ----------------

% (1) Ridge smoothness: variance of first difference
ridge_diff   = diff(tf_ridge);
ridge_smooth = var(ridge_diff(:));

% (2) High-Doppler energy ratio
fd_norm = linspace(-1,1,length(doppler_marginal)); % normalized Doppler axis
thr = 0.2 * max(abs(fd_norm));
high_idx = abs(fd_norm) >= thr;

E_high = sum(doppler_marginal(high_idx).^2);
E_tot  = sum(doppler_marginal.^2);
high_doppler_ratio = E_high / (E_tot + 1e-9);

% (3) Periodicity peak sharpness (ACF contrast)
acf_peak_sharp = max(acf) / (mean(acf(2:end)) + 1e-9);

% Append new features to normalized set
X_aug = [X_norm, ridge_smooth, high_doppler_ratio, acf_peak_sharp];

feature_names = {
 'RidgeMean',...
 'RidgeStd',...
 'DopplerBW',...
 'HarmEnvMean',...
 'HarmEnvStd',...
 'ACF_MaxLag',...
 'Dopp_Skew',...
 'Dopp_Kurt',...
 'RidgeSmoothness',...
 'HighDopplerEnergyRatio',...
 'ACF_PeakSharpness'
};

% ---------------- B) PCA (ROBUST TO SINGLE SAMPLE) ----------------
N = size(X_aug,1);

if N < 2
    % ======== SINGLE-SAMPLE SAFE MODE ========
    warning('Only one sample available → skipping PCA (not mathematically defined).');

    X_pca = X_aug;          
    pca_coeff = eye(size(X_aug,2)); 
    explained_variance = ones(size(X_aug,2),1) * (100/size(X_aug,2));

    % Visualization instead of PC scatter
    figure('Name','Feature Space (Single Sample)');
    bar(X_aug);
    xticklabels(feature_names);
    xtickangle(45);
    title('Augmented Feature Vector (No PCA — single sample)');
    grid on;

else
    % ======== NORMAL PCA MODE (FOR DATASETS) ========
    [coeff, score, ~, ~, explained] = pca(X_aug);

    cumVar = cumsum(explained);
    k = find(cumVar >= 95,1,'first');

    X_pca = score(:,1:k);
    pca_coeff = coeff(:,1:k);
    explained_variance = explained;

    % ---- Scree plot ----
    figure('Name','PCA Explained Variance');
    plot(cumVar,'LineWidth',2); grid on;
    yline(95,'r--','95% threshold');
    xlabel('Number of PCs');
    ylabel('Cumulative Variance (%)');
    title('PCA Variance Retention');

    % ---- PC1–PC2 scatter ----
    figure('Name','PC1 vs PC2');
    scatter(score(:,1), score(:,2), 90, 'filled');
    xlabel('PC1'); ylabel('PC2');
    grid on; title('Latent Feature Space');
end

% ---------------- D) CLEAN OUTPUT INTERFACE ----------------
F_out = struct();
F_out.X_norm = X_norm;
F_out.X_pca = X_pca;
F_out.pca_coeff = pca_coeff;
F_out.explained_variance = explained_variance;
F_out.mu = mu;
F_out.sigma = sigma;
F_out.feature_names = feature_names;

disp('Block-5 complete: F_out ready for SVM (Block-6).');
