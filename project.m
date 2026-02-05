clc; clear; close all;

%% ======================= MAIN SCRIPT =======================

fs = 4000;
SNR_dB = 10;

%% ---------------- BLOCK 1 ----------------
[s_uav, s_bird, t] = B1_generate_signals(fs, SNR_dB);

figure('Name','Block-1 Comparison');
subplot(2,1,1);
plot(t, real(s_uav));
title('Block-1: UAV Radar Echo (Real Part)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(2,1,2);
plot(t, real(s_bird));
title('Block-1: Bird Radar Echo (Real Part)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

%% ---------------- BLOCK 2 ----------------
[x_alpha_uav, fd_uav, t_stft_uav, H2_uav, alpha_opt_uav] = ...
        B2_chirp_optimizer(s_uav, fs);

[x_alpha_bird, fd_bird, t_stft_bird, H2_bird, alpha_opt_bird] = ...
        B2_chirp_optimizer(s_bird, fs);

fprintf('\nBlock-2 Output:\n');
fprintf('True UAV Chirp Rate  = %.2f Hz/s\n', 2e4);
fprintf('Estimated UAV Chirp  = %.2f Hz/s\n', alpha_opt_uav);

figure('Name','Block-2 Comparison');

subplot(2,2,1);
plot(t, real(x_alpha_uav));
title('Dechirped UAV'); xlabel('Time'); ylabel('Amplitude'); grid on;

subplot(2,2,2);
plot(alpha_opt_uav, H2_uav, 'LineWidth',2);
title('UAV: Entropy vs Chirp'); xlabel('\alpha'); ylabel('H_2'); grid on;

subplot(2,2,3);
plot(t, real(x_alpha_bird));
title('Dechirped Bird'); xlabel('Time'); ylabel('Amplitude'); grid on;

subplot(2,2,4);
plot(alpha_opt_bird, H2_bird, 'LineWidth',2);
title('Bird: Entropy vs Chirp'); xlabel('\alpha'); ylabel('H_2'); grid on;

%% ---------------- BLOCK 3 ----------------
[S_orig_uav, S_enh_uav, fd_uav, t_stft_uav, snr_before_uav, snr_after_uav] ...
    = B3_doppler_enhancer(x_alpha_uav, fs);

[S_orig_bird, S_enh_bird, fd_bird, t_stft_bird, snr_before_bird, snr_after_bird] ...
    = B3_doppler_enhancer(x_alpha_bird, fs);

fprintf('\nBlock-3 Output:\n');
fprintf('Harmonic SNR Before (UAV) = %.2f dB\n', snr_before_uav);
fprintf('Harmonic SNR After  (UAV) = %.2f dB\n', snr_after_uav);

figure('Name','Block-3 Comparison');

subplot(2,2,1);
imagesc(t_stft_uav, fd_uav, 10*log10(S_orig_uav+1e-6));
axis xy; title('Original UAV'); colorbar;

subplot(2,2,2);
imagesc(t_stft_uav, fd_uav, 10*log10(S_enh_uav+1e-6));
axis xy; title('Enhanced UAV'); colorbar;

subplot(2,2,3);
imagesc(t_stft_bird, fd_bird, 10*log10(S_orig_bird+1e-6));
axis xy; title('Original Bird'); colorbar;

subplot(2,2,4);
imagesc(t_stft_bird, fd_bird, 10*log10(S_enh_bird+1e-6));
axis xy; title('Enhanced Bird'); colorbar;

%% ---------------- BLOCK 4 ----------------
TF_uav  = B4_TF_signature(S_enh_uav, fd_uav);
TF_bird = B4_TF_signature(S_enh_bird, fd_bird);

fprintf('\nBlock-4 Output:\n');
fprintf('UAV Doppler BW = %.2f Hz\n', TF_uav.doppler_bw);
fprintf('Bird Doppler BW = %.2f Hz\n', TF_bird.doppler_bw);

figure('Name','Block-4 Comparison');

subplot(2,2,1);
imagesc(t_stft_uav, fd_uav, 10*log10(S_enh_uav+1e-6));
axis xy; title('Enhanced UAV');

subplot(2,2,2);
imagesc(t_stft_bird, fd_bird, 10*log10(S_enh_bird+1e-6));
axis xy; title('Enhanced Bird');

subplot(2,2,3);
plot(fd_uav, TF_uav.doppler_marginal);
title('UAV Doppler Marginal'); grid on;

subplot(2,2,4);
plot(fd_bird, TF_bird.doppler_marginal);
title('Bird Doppler Marginal'); grid on;

figure('Name','Block-4 Ridge');

subplot(2,1,1);
plot(t_stft_uav, TF_uav.tf_ridge); title('UAV Ridge'); grid on;

subplot(2,1,2);
plot(t_stft_bird, TF_bird.tf_ridge); title('Bird Ridge'); grid on;

%% ---------------- BLOCK 5 ----------------
F_uav  = B5_feature_extractor(TF_uav, fd_uav);
F_bird = B5_feature_extractor(TF_bird, fd_bird);

fprintf('\nBlock-5 Output:\n');
disp('Normalized feature vectors stored in F_uav.X_norm and F_bird.X_norm');

figure('Name','Block-5 Comparison');
bar([F_uav.X_norm; F_bird.X_norm]');
legend({'UAV','Bird'});
title('Normalized Features'); grid on;

%% ---------------- DATASET FOR BLOCK-6 ----------------
X_all = [F_uav.X_pca; F_bird.X_pca];
labels = [1; 0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ====================== FUNCTIONS (B1–B5) ==============================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% -------- B1 --------
function [s_uav, s_bird, t] = B1_generate_signals(fs, SNR_dB)
T = 1; t = 0:1/fs:T-1/fs;

uav_prop_freqs = 200:20:400;
bird_flap_freqs = 2:2:26;

alpha_uav_true = 2e4; alpha_bird_true = 100;

uav_phase = zeros(size(t));
for f0 = uav_prop_freqs
    uav_phase = uav_phase + sin(2*pi*(f0*t + 0.5*alpha_uav_true*t.^2));
end
s_uav = awgn(exp(1j*2*pi*0.5*uav_phase), SNR_dB, 'measured');

bird_phase = zeros(size(t));
for f0 = bird_flap_freqs
    bird_phase = bird_phase + sin(2*pi*(f0*t + 0.5*alpha_bird_true*t.^2));
end
s_bird = awgn(exp(1j*2*pi*0.5*bird_phase), SNR_dB, 'measured');
end

%% -------- B2 --------
function [x_alpha, fd, t_stft, H2, alpha_opt] = B2_chirp_optimizer(x, fs)

t = (0:length(x)-1)/fs;
alpha_vec = linspace(-5e4,5e4,64);
win = hamming(256); noverlap = 192; nfft = 1024;
H2 = zeros(length(alpha_vec),1);

for i = 1:length(alpha_vec)
    chirp = exp(-1j*pi*alpha_vec(i)*t.^2);
    x_d = x.*chirp;
    [S,fd,t_stft] = spectrogram(x_d,win,noverlap,nfft,fs);
    P = abs(S).^2; P = P/sum(P(:))+eps;
    H2(i) = -log(sum(P(:).^2));
end

[~,idx] = min(H2);
alpha_opt = alpha_vec(idx);
x_alpha = x.*exp(-1j*pi*alpha_opt*t.^2);
end

%% -------- B3 --------
function [S_orig, S_enh, fd, t_stft, snr_before, snr_after] = ...
         B3_doppler_enhancer(x_alpha, fs)

win = hamming(256); noverlap = 192; nfft = 1024;
[X,fd,t_stft] = spectrogram(x_alpha,win,noverlap,nfft,fs);
S_orig = abs(X).^2;

fc = 0.05*fs;
HP = abs(fd)>=fc;
X_hp = X.*HP;
P_noise = median(abs(X_hp).^2,2);
X_white = X_hp./sqrt(P_noise+1e-6);

h = fspecial('gaussian',[11 1],2);
X_sharp = conv2(X_white,h,'same');

S_mag = abs(X_sharp);
[U,Ssvd,V] = svd(S_mag,'econ');
S_bg = U(:,1)*Ssvd(1,1)*V(:,1)';
S_enh = max(S_mag-S_bg,0);

body = abs(fd)<fc; harm = abs(fd)>=fc;
snr_before = 10*log10(mean(S_orig(harm,:),'all')/mean(S_orig(body,:),'all'));
snr_after  = 10*log10(mean(S_enh(harm,:),'all')/mean(S_enh(body,:),'all'));
end

%% -------- B4 --------
function TF = B4_TF_signature(S_enh, fd)

eps_val = 1e-6;
S_norm = S_enh./(sum(S_enh,1)+eps_val);

TF.doppler_marginal = sum(S_norm,2);
TF.time_marginal = sum(S_norm,1);

[~,idx] = max(S_norm,[],1);
TF.tf_ridge = fd(idx);

cdf = cumsum(TF.doppler_marginal);
cdf = cdf/max(cdf);
TF.doppler_bw = fd(find(cdf>=0.95,1)) - fd(find(cdf>=0.05,1));

tm = TF.time_marginal-mean(TF.time_marginal);
TF.acf = xcorr(tm,'coeff'); TF.acf = TF.acf(length(tm):end);
end

%% -------- B5 --------
function F = B5_feature_extractor(TF, fd)

X = [mean(abs(TF.tf_ridge)), std(TF.tf_ridge), ...
     TF.doppler_bw, mean(TF.doppler_marginal), ...
     std(TF.doppler_marginal)];

mu = mean(X); sigma = std(X)+1e-9;
F.X_norm = (X-mu)./sigma;

F.X_pca = F.X_norm;      % single-sample safe
F.mu = mu; F.sigma = sigma;
end
