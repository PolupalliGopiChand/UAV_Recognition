clc; clear; close all;

fs = 4000;
SNR_dB = 10;

%% ======================= BLOCK 1 ===============================
[s_uav, s_bird, t] = B1_generate_signals(fs, SNR_dB);

figure;
% UAV Echo signal 
subplot(2,1,1); plot(t, real(s_uav));
title('UAV Echo Signal');
xlabel('Time (s)'); ylabel('Amplitude (v)'); grid on;
% Bird Echo signal
subplot(2,1,2); plot(t, real(s_bird));
title('Bird Echo Signal');
xlabel('Time (s)'); ylabel('Amplitude (v)'); grid on;
%% ======================= BLOCK 2 ===============================
[t, x_alpha_uav, alpha_opt_uav, alpha_vec_uav, H2_uav]     = B2_chirp_optimizer(s_uav, fs);
[t, x_alpha_bird, alpha_opt_bird, alpha_vec_bird, H2_bird] = B2_chirp_optimizer(s_bird, fs);

fprintf('\nBlock-2 Output:\n');
fprintf('UAV Actual Chirp Rate     = %.2f Hz/s\n', 2e4);
fprintf('UAV Estimated Chirp Rate  = %.2f Hz/s\n', alpha_opt_uav);
fprintf('Bird Actual Chirp Rate    = %.2f Hz/s\n', 1000);
fprintf('Bird Estimated Chirp Rate  = %.2f Hz/s\n', alpha_opt_bird);

figure;
% UAV Entropy vs Chirp Rate
subplot(3,2,1); plot(alpha_vec_uav, H2_uav, 'LineWidth', 2);
title('Entropy vs Chirp Rate for UAV');
xlabel('Chirp Rate (Hz/s)'); ylabel('Renyi Entropy'); grid on;
% Bird Entropy vs Chirp Rate
subplot(3,2,2); plot(alpha_vec_bird, H2_bird, 'LineWidth', 2);
title('Entropy vs Chirp Rate for Bird'); 
xlabel('Chirp Rate (Hz/s)'); ylabel('Renyi Entropy'); grid on;
% UAV De-Chirped signal
subplot(3,1,2); plot(t, real(x_alpha_uav));
title('Dechirped UAV Signal');
xlabel('Time (s)'); ylabel('Amplitude (v)'); grid on;
% Bird De-Chirped signal
subplot(3,1,3); plot(t, real(x_alpha_bird));
title('Dechirped Bird Signal');
xlabel('Time (s)'); ylabel('Amplitude (v)'); grid on;

%% ======================= BLOCK 3 ===============================
[S_orig_uav, S_enh_uav, fd_uav, t_stft_uav, snr_before_uav, snr_after_uav] = B3_doppler_enhancer(x_alpha_uav, fs);
[S_orig_bird, S_enh_bird, fd_bird, t_stft_bird, snr_before_bird, snr_after_bird] = B3_doppler_enhancer(x_alpha_bird, fs);

fprintf('\nBlock-3 Output:\n');
fprintf('Harmonic SNR Before (UAV) = %.2f dB\n', snr_before_uav);
fprintf('Harmonic SNR After  (UAV) = %.2f dB\n', snr_after_uav);
fprintf('Harmonic SNR Before (UAV) = %.2f dB\n', snr_before_bird);
fprintf('Harmonic SNR After  (UAV) = %.2f dB\n', snr_after_bird);

figure;
% Original UAV Spectrogram
subplot(2,2,1); imagesc(t_stft_uav, fd_uav, 10*log10(S_orig_uav+1e-6));
axis xy; c = colorbar; ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Original UAV Spectrogram');
% Original Bird Spectrogram
subplot(2,2,2); imagesc(t_stft_bird, fd_bird, 10*log10(S_orig_bird+1e-6));
axis xy; c = colorbar; ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Original Bird Spectrogram');
% Enhanced UAV Spectrogram
subplot(2,2,3); imagesc(t_stft_uav, fd_uav, 10*log10(S_enh_uav+1e-6));
axis xy; c = colorbar; ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Enhanced UAV Spectrogram');
% Enhanced Bird Spectrogram
subplot(2,2,4); imagesc(t_stft_bird, fd_bird, 10*log10(S_enh_bird+1e-6));
axis xy; c = colorbar; ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Enhanced Bird Spectrogram');

%% ======================= BLOCK 4 ===============================
TF_uav  = B4_TF_signature(S_enh_uav, fd_uav);
TF_bird = B4_TF_signature(S_enh_bird, fd_bird);

fprintf('\nBlock-4 Output:\n');
fprintf('UAV Doppler Bandwidth  = %.2f Hz\n', TF_uav.doppler_bw);
fprintf('Bird Doppler Bandwidth = %.2f Hz\n', TF_bird.doppler_bw);

figure;

subplot(4,2,1);
imagesc(t_stft_uav, fd_uav, 10*log10(S_enh_uav+1e-6));
axis xy; title('Enhanced UAV Spectrogram');

subplot(4,2,3);
imagesc(t_stft_bird, fd_bird, 10*log10(S_enh_bird+1e-6));
axis xy; title('Enhanced Bird Spectrogram');

subplot(4,2,2);
imagesc(t_stft_uav, fd_uav, 10*log10(TF_uav.S_norm+1e-6));
axis xy; title('Enhanced UAV Spectrogram');

subplot(4,2,4);
imagesc(t_stft_bird, fd_bird, 10*log10(TF_bird.S_norm+1e-6));
axis xy; title('Enhanced Bird Spectrogram');

subplot(4,2,5);
plot(fd_uav, TF_uav.doppler_marginal);
title('UAV Doppler Marginal'); grid on;

subplot(4,2,7);
plot(fd_bird, TF_bird.doppler_marginal);
title('Bird Doppler Marginal'); grid on;

subplot(4,2,6);
plot(t_stft_uav, TF_uav.tf_ridge); title('UAV Micro-Doppler Ridge'); grid on;

subplot(4,2,8);
plot(t_stft_bird, TF_bird.tf_ridge); title('Bird Micro-Doppler Ridge'); grid on;

fprintf('Block-4 complete: TF signatures and micro-Doppler extracted.\n');

%% ======================= BLOCK 5 ===============================
fprintf('\n******** BLOCK-5: FEATURE EXTRACTION ********\n');

F_uav  = B5_feature_extractor(TF_uav, fd_uav);
F_bird = B5_feature_extractor(TF_bird, fd_bird);

fprintf('\nBlock-5 Output:\n');
disp('Normalized feature vectors generated for UAV and Bird.');
disp('These will be used as inputs to the SVM classifier.');

figure;
bar([F_uav.X_norm; F_bird.X_norm]');
legend({'UAV','Bird'});
title('Normalized Feature Comparison');
xlabel('Feature Index');
ylabel('Z-score Value');
grid on;

fprintf('Block-5 complete: Features ready for classification.\n');

%% ======================= DATASET FOR BLOCK-6 ===============================
X_all = [F_uav.X_pca; F_bird.X_pca];
labels = [1; 0];

fprintf('\nDataset prepared for Block-6 (SVM).\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ======================== FUNCTIONS (B1–B5) =============================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% -------- B1 --------
function [s_uav, s_bird, t] = B1_generate_signals(fs, SNR_dB)

T = 1; 
t = 0:1/fs:T-1/fs;
amp = 1.0; 

uav_freqs = 200:20:400;
bird_freqs = 2:2:26;

alpha_uav = 2e4; 
alpha_bird = 1000;

% UAV Signal
uav_phase = zeros(size(t));
for k = 1:length(uav_freqs)
    f0 = uav_freqs(k);
    uav_phase = uav_phase + (amp/k) * sin(2*pi*( f0*t + 0.5*alpha_uav*t.^2 ));
end
s_uav = exp(1j * 2*pi*0.5 * uav_phase);
s_uav = awgn(s_uav, SNR_dB, 'measured');

% Bird Signal
bird_phase = zeros(size(t));
for k = 1:length(bird_freqs)
    f0 = bird_freqs(k);
    bird_phase = bird_phase + (amp/k) * sin(2*pi*( f0*t + 0.5*alpha_bird*t.^2 ));
end
s_bird = exp(1j * 2*pi*0.5 * bird_phase);
s_bird = awgn(s_bird, SNR_dB, 'measured');

end

%% -------- B2 --------
function [t, x_alpha, alpha_opt, alpha_vec, H2] = B2_chirp_optimizer(x, fs)

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

chirp_comp_opt = exp(-1j*pi*alpha_opt*t.^2);
x_alpha = x .* chirp_comp_opt;

end

%% -------- B3 --------
function [S_orig, S_enh, fd, t_stft, snr_before, snr_after] = B3_doppler_enhancer(x_alpha, fs)

PRF = fs;
winLen   = 256;
hop      = 64;
noverlap = winLen - hop;
nfft     = 1024;
win      = hamming(winLen);

% De-Chirped signal to Spectrogram representation
[X, fd, t_stft] = spectrogram(x_alpha, win, noverlap, nfft, fs);
S_orig = abs(X).^2;

% Body removal
fc = 0.05 * PRF;
HPmask = abs(fd) >= fc;
X_hp = X .* HPmask;

% Spectral whitening
P_noise = median(abs(X_hp).^2, 2);
eps_val = 1e-6;
X_white = X_hp ./ sqrt(P_noise + eps_val);

% Doppler Ridge sharpening
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

% SNR Metric Calculations (Hormonic to Body Ratio)
bodyBand = abs(fd) < fc;
harmBand = abs(fd) >= fc;

bodyPower = mean(S_orig(bodyBand,:), 'all');
harmPower = mean(S_orig(harmBand,:), 'all');
bodyPower_enh = mean(S_enh(bodyBand,:), 'all');
harmPower_enh = mean(S_enh(harmBand,:), 'all');

snr_before = 10*log10(harmPower / bodyPower);
snr_after  = 10*log10(harmPower_enh / bodyPower_enh);

end

%% -------- B4 --------
function TF = B4_TF_signature(S_enh, fd)

eps_val = 1e-6;
S_norm = S_enh./(sum(S_enh,1)+eps_val);
TF.S_norm = S_enh./(sum(S_enh,1)+eps_val)

TF.doppler_marginal = sum(S_norm,2);
TF.time_marginal = sum(S_norm,1);

[~,idx] = max(S_norm,[],1);
TF.tf_ridge = fd(idx);

cdf = cumsum(TF.doppler_marginal);
cdf = cdf/max(cdf);
TF.doppler_bw = fd(find(cdf>=0.95,1)) - fd(find(cdf>=0.05,1));

end

%% -------- B5 --------
function F = B5_feature_extractor(TF, fd)

X = [mean(abs(TF.tf_ridge)), std(TF.tf_ridge), ...
     TF.doppler_bw, mean(TF.doppler_marginal), ...
     std(TF.doppler_marginal)];

mu = mean(X); 
sigma = std(X)+1e-9;

F.X_norm = (X-mu)./sigma;
F.X_pca = F.X_norm;
F.mu = mu; 
F.sigma = sigma;

end
