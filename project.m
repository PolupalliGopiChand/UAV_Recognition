clc; clear; close all;

fs = 4000;
SNR_dB = 10;

%% ============================ BLOCK 1 ===================================
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

%% ============================ BLOCK 2 ===================================
[t, x_alpha_uav,  alpha_opt_uav,  alpha_vec_uav,  H2_uav ] = B2_chirp_optimizer(s_uav,  fs);
[t, x_alpha_bird, alpha_opt_bird, alpha_vec_bird, H2_bird] = B2_chirp_optimizer(s_bird, fs);

fprintf('Block-2 Output:\n');
fprintf('UAV Actual Chirp Rate      = %.2f Hz/s\n', 2e4);
fprintf('UAV Estimated Chirp Rate   = %.2f Hz/s\n', alpha_opt_uav);
fprintf('Bird Actual Chirp Rate     = %.2f Hz/s\n', 1000);
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

%% ============================ BLOCK 3 ===================================
[S_orig_uav,  S_enh_uav,  fd_uav,  t_stft_uav,  snr_before_uav,  snr_after_uav ] = B3_doppler_enhancer(x_alpha_uav,  fs);
[S_orig_bird, S_enh_bird, fd_bird, t_stft_bird, snr_before_bird, snr_after_bird] = B3_doppler_enhancer(x_alpha_bird, fs);

fprintf('\nBlock-3 Output:\n');
fprintf('Harmonic SNR Before (UAV)  = %.2f dB\n', snr_before_uav);
fprintf('Harmonic SNR After  (UAV)  = %.2f dB\n', snr_after_uav);
fprintf('Harmonic SNR Before (UAV)  = %.2f dB\n', snr_before_bird);
fprintf('Harmonic SNR After  (UAV)  = %.2f dB\n', snr_after_bird);

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

%% ============================ BLOCK 4 ===================================
TF_sig_uav  = B4_TF_signature_generator(S_enh_uav,  fd_uav,  t_stft_uav );
TF_sig_bird = B4_TF_signature_generator(S_enh_bird, fd_bird, t_stft_bird);

fprintf('\nBlock-4 Output:\n');
fprintf('UAV Doppler Bandwidth      = %.2f Hz\n', TF_sig_uav.doppler_bw);
fprintf('Bird Doppler Bandwidth     = %.2f Hz\n', TF_sig_bird.doppler_bw);
fprintf('UAV Spectral Entropy       = %.4f\n', TF_sig_uav.spectral_entropy);
fprintf('Bird Spectral Entropy      = %.4f\n', TF_sig_bird.spectral_entropy);

figure;
% Normalized UAV Spectrogram
subplot(3,2,1); imagesc(TF_sig_uav.t_stft, TF_sig_uav.fd, 10*log10(TF_sig_uav.S_normalized + 1e-6));
axis xy; c = colorbar; ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Normalized UAV Spectrogram');
% Normalized Bird Spectrogram
subplot(3,2,2); imagesc(TF_sig_bird.t_stft, TF_sig_bird.fd, 10*log10(TF_sig_bird.S_normalized + 1e-6));
axis xy; c = colorbar; ylabel(c, 'Energy (dB)'); xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Normalized UAV Spectrogram');
% UAV Micro-Doppler Ridge
subplot(3,2,3); plot(TF_sig_uav.t_stft, TF_sig_uav.tf_ridge, 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Micro-Doppler Ridge'); grid on;
% Bird Micro-Doppler Ridge
subplot(3,2,4); plot(TF_sig_bird.t_stft, TF_sig_bird.tf_ridge, 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Doppler Frequency (Hz)');
title('Micro-Doppler Ridge'); grid on;
% UAV Doppler Marginal
subplot(3,2,5); plot(TF_sig_uav.fd, TF_sig_uav.doppler_marginal, 'LineWidth', 1.5);
xlabel('Doppler Frequency (Hz)'); ylabel('Normalized Energy');
title('Doppler Marginal Signature'); grid on;
% Bird Doppler Marginal
subplot(3,2,6); plot(TF_sig_bird.fd, TF_sig_bird.doppler_marginal, 'LineWidth', 1.5);
xlabel('Doppler Frequency (Hz)'); ylabel('Normalized Energy');
title('Doppler Marginal Signature'); grid on;



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

win_len  = 512;
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
function TF_sig = B4_TF_signature_generator(S_enh, fd, t_stft)

eps_val = 1e-6;
S_norm  = S_enh  ./ (sum(S_enh,1)  + eps_val); % Normalize spectrograms
dop_marg  = sum(S_norm,2); % Doppler marginals

% Micro-Doppler ridge
[~, ridge_idx]  = max(S_norm,[],1);
tf_ridge  = fd(ridge_idx);

% Spectral entropy (spread of micro-Doppler energy)
p_dop = dop_marg / (sum(dop_marg)+eps_val);
spec_entropy = -sum(p_dop .* log(p_dop + eps_val));

% Doppler bandwidth
cdf = cumsum(dop_marg);
cdf = cdf / max(cdf);
f_low  = fd(find(cdf>=0.05,1));
f_high = fd(find(cdf>=0.95,1));
doppler_bw = f_high - f_low;

% Store outputs
TF_sig.S_normalized     = S_norm;
TF_sig.doppler_marginal = dop_marg;
TF_sig.tf_ridge         = tf_ridge;
TF_sig.spectral_entropy = spec_entropy;
TF_sig.doppler_bw       = doppler_bw;
TF_sig.fd               = fd;
TF_sig.t_stft           = t_stft;

end
